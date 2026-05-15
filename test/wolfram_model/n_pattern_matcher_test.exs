defmodule WolframModel.NPatternMatcherTest do
  use ExUnit.Case, async: true
  alias WolframModel
  alias WolframModel.Matcher
  alias Hypergraph

  test "three-pattern rule matches three connected hyperedges" do
    # Pattern: [[1,2],[2,3],[3,4]] — a path of length 3
    hyperedges = [[1, 2], [2, 3], [3, 4]]
    pattern = [[1, 2], [2, 3], [3, 4]]
    results = Matcher.match(hyperedges, pattern)
    assert length(results) >= 1

    # The mapping must consistently bind 1->1, 2->2, 3->3, 4->4
    mapping = hd(results).mapping
    assert mapping[1] == 1
    assert mapping[2] == 2
    assert mapping[3] == 3
    assert mapping[4] == 4
  end

  test "three-pattern rule produces no match when no consistent binding exists" do
    # Three disconnected edges — no shared variables, but the rule requires
    # consistent binding for shared variable positions.
    hyperedges = [[1, 2], [10, 11], [20, 21]]

    # Pattern shares variable 2 (position 2 of p1 == position 1 of p2)
    pattern = [[1, 2], [2, 3], [3, 4]]
    results = Matcher.match(hyperedges, pattern)
    assert results == []
  end

  test "three-pattern rule evolution via WolframModel" do
    hg =
      Hypergraph.new()
      |> Hypergraph.add_hyperedge([1, 2])
      |> Hypergraph.add_hyperedge([2, 3])
      |> Hypergraph.add_hyperedge([3, 4])

    rule = %{
      name: "three_to_one",
      pattern: [[1, 2], [2, 3], [3, 4]],
      replacement: [[1, 4]]
    }

    model = WolframModel.new(hg, [rule])
    evolved = WolframModel.evolve_step(model)

    assert evolved.generation == 1
    assert Enum.any?(Hypergraph.hyperedges(evolved.hypergraph), fn he -> he == [1, 4] end)
  end

  test "same-tag atoms in different replacement hyperedges share one fresh vertex" do
    hg = Hypergraph.new() |> Hypergraph.add_hyperedge([1, 2])

    rule = %{
      name: "star",
      pattern: [[1, 2]],
      replacement: [[1, :new], [2, :new], [:new, :new]]
    }

    model = WolframModel.new(hg, [rule])
    evolved = WolframModel.evolve_step(model)

    edges = Hypergraph.hyperedges(evolved.hypergraph)

    # Find the fresh vertex — it should be shared across all three replacement edges
    new_vertex =
      Enum.find_value(edges, fn
        [1, v] -> v
        _ -> nil
      end)

    assert new_vertex != nil

    assert Enum.any?(edges, fn he -> he == [2, new_vertex] end),
           "edge [2, new] missing; edges: #{inspect(edges)}"

    assert Enum.any?(edges, fn he -> he == [new_vertex, new_vertex] end),
           "self-loop [new, new] missing; edges: #{inspect(edges)}"
  end

  test "wolfram rule_1 matches initial hypergraph" do
    [rule] = WolframModel.RuleSet.rule_set(:wolfram, :rule_1)

    hg =
      Hypergraph.new()
      |> Hypergraph.add_hyperedge([1, 2])
      |> Hypergraph.add_hyperedge([1, 3])

    results = Matcher.match(Hypergraph.hyperedges(hg), rule.pattern)
    assert length(results) >= 1
  end
end
