defmodule WolframModel.MultiwayDedupeTest do
  use ExUnit.Case, async: true
  alias Hypergraph
  alias WolframModel

  test "multiway_step deduplicates identical resulting hypergraphs" do
    hg = Hypergraph.new() |> Hypergraph.add_hyperedge([1, 2]) |> Hypergraph.add_hyperedge([3, 4])

    rule = %{
      pattern: [MapSet.new([:a, :b])],
      replacement: [MapSet.new([:a, :b])],
      name: "noop"
    }

    model = WolframModel.new(hg, [rule])
    next_states = WolframModel.multiway_step(model)

    # Applying the noop rule to either hyperedge results in identical hypergraphs
    assert length(next_states) == 1
  end
end
