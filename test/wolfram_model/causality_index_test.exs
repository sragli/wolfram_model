defmodule WolframModel.CausalityIndexTest do
  use ExUnit.Case, async: true
  alias Hypergraph
  alias WolframModel

  test "parent links detect causal edges" do
    # Build hg with two hyperedges sharing vertex 2
    hg = Hypergraph.new() |> Hypergraph.add_hyperedge([1, 2]) |> Hypergraph.add_hyperedge([2, 3])

    # Rule that removes [1,2] and adds [1,4]
    rule1 = %{
      pattern: [MapSet.new([:a, :b])],
      replacement: [MapSet.new([:a, :new])],
      name: "r1"
    }

    # Rule that removes [2,3] and adds [3,5]
    rule2 = %{
      pattern: [MapSet.new([:a, :b])],
      replacement: [MapSet.new([:a, :new])],
      name: "r2"
    }

    model = WolframModel.new(hg, [rule1, rule2])

    # Apply first rule (should match first hyperedge)
    m1 = WolframModel.evolve_step(model)

    # Apply second rule; because vertex 2 is shared, the second event should list first event as a parent
    m2 = WolframModel.evolve_step(m1)

    [e2, e1] = m2.causal_network

    assert e2.parent_ids != []
    assert Enum.member?(e2.parent_ids, e1.id)

    stats = WolframModel.Analytics.analyze_causality(m2)
    assert stats.causal_edges >= 1
  end
end
