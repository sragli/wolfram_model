defmodule WolframModel.UpdateOrderingTest do
  use ExUnit.Case, async: true
  alias WolframModel
  alias Hypergraph

  # A hypergraph with two potential matches for the same rule.
  # Edges: [1,2], [2,3], [3,4]
  # Rule: [[a,b]] -> [[a,b,b]]  (single-edge match, three possible)
  defp triple_model do
    hg =
      Hypergraph.new()
      |> Hypergraph.add_hyperedge([1, 2])
      |> Hypergraph.add_hyperedge([2, 3])
      |> Hypergraph.add_hyperedge([3, 4])

    rule = %{name: "tag", pattern: [[1, 2]], replacement: [[1, 2, :t]]}
    WolframModel.new(hg, [rule])
  end

  test ":first ordering applies the first available match" do
    m = triple_model()
    evolved = WolframModel.evolve_step(m, ordering: :first)
    assert evolved.generation == 1
  end

  test ":leftmost ordering picks the match with lowest vertex keys" do
    m = triple_model()
    evolved = WolframModel.evolve_step(m, ordering: :leftmost)
    assert evolved.generation == 1

    # The leftmost match should be [1,2] — vertex 1 is smallest
    assert Enum.any?(Hypergraph.hyperedges(evolved.hypergraph), fn
             [1, 2, _] -> true
             _ -> false
           end)
  end

  test ":random ordering always produces an evolved model" do
    m = triple_model()
    evolved = WolframModel.evolve_step(m, ordering: :random)
    assert evolved.generation == 1
  end

  test "evolve_parallel applies all non-conflicting matches in one step" do
    # Three completely disjoint edges — all can be matched simultaneously.
    hg =
      Hypergraph.new()
      |> Hypergraph.add_hyperedge([1, 2])
      |> Hypergraph.add_hyperedge([3, 4])
      |> Hypergraph.add_hyperedge([5, 6])

    rule = %{name: "dup", pattern: [[1, 2]], replacement: [[1, 2], [1, :new]]}
    model = WolframModel.new(hg, [rule])

    evolved = WolframModel.evolve_parallel(model)
    edges = Hypergraph.hyperedges(evolved.hypergraph)

    # Three original edges should still be present, and three new ones added.
    # 6 original endpoints → 6 edges total (3 original + 3 new)
    assert length(edges) >= 6
  end

  test "fixpoint?/1 returns true when no rules match" do
    hg = Hypergraph.new() |> Hypergraph.add_hyperedge([1, 2, 3])
    # Rule only matches 2-edges; the hypergraph has a 3-edge
    rule = %{name: "binary", pattern: [[1, 2]], replacement: [[1], [2]]}
    model = WolframModel.new(hg, [rule])
    assert WolframModel.fixpoint?(model)
  end

  test "fixpoint?/1 returns false when a rule can match" do
    hg = Hypergraph.new() |> Hypergraph.add_hyperedge([1, 2])
    rule = %{name: "binary", pattern: [[1, 2]], replacement: [[1], [2]]}
    model = WolframModel.new(hg, [rule])
    refute WolframModel.fixpoint?(model)
  end

  test "evolve_until_fixpoint/2 terminates at fixpoint" do
    hg = Hypergraph.new() |> Hypergraph.add_hyperedge([1, 2])
    # After one step [1,2] → [[1],[2]]; no 2-edges left — fixpoint.
    rule = %{name: "split", pattern: [[1, 2]], replacement: [[1], [2]]}
    model = WolframModel.new(hg, [rule])

    final = WolframModel.evolve_until_fixpoint(model)
    assert WolframModel.fixpoint?(final)
  end

  test "evolve_until_fixpoint/2 respects max_steps" do
    hg = Hypergraph.new() |> Hypergraph.add_hyperedge([1, 2])
    # Rule always produces a 2-edge — never terminates naturally.
    rule = %{name: "loop", pattern: [[1, 2]], replacement: [[1, 2], [2, :new]]}
    model = WolframModel.new(hg, [rule])

    result = WolframModel.evolve_until_fixpoint(model, 5)
    assert result.generation <= 5
  end
end
