defmodule WolframModel.EventTest do
  use ExUnit.Case, async: true
  alias Hypergraph
  alias WolframModel

  test "evolve_step creates a structured event with deltas and indices" do
    hg = Hypergraph.new() |> Hypergraph.add_hyperedge([1, 2])

    rule = %{
      pattern: [MapSet.new([:a, :b])],
      replacement: [MapSet.new([:a, :b, :c])],
      name: "add_c"
    }

    model = WolframModel.new(hg, [rule])
    model2 = WolframModel.evolve_step(model)

    assert model2.generation == 1

    [event] = model2.causal_network

    # Event fields
    assert event.id == 1
    assert event.generation == 1
    assert event.rule.name == "add_c"

    # Deltas
    assert length(event.removed) == 1
    assert length(event.added) == 1

    # Removed and added disjoint
    assert Enum.empty?(
             Enum.filter(event.removed ++ event.added, fn he ->
               he in event.removed and he in event.added
             end)
           )

    # Affected vertices include 1,2 and new vertex from substitution
    affected = event.affected_vertices |> MapSet.to_list()
    assert Enum.member?(affected, 1) and Enum.member?(affected, 2)

    # Indices updated
    Enum.each(affected, fn v ->
      ids = Map.get(model2.event_index, v) |> MapSet.to_list()
      assert Enum.member?(ids, event.id)
    end)

    # next_event_id advanced
    assert model2.next_event_id == 2
  end
end
