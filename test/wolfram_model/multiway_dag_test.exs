defmodule WolframModel.MultiwaDAGTest do
  use ExUnit.Case, async: true
  alias WolframModel
  alias Hypergraph

  defp simple_model do
    hg =
      Hypergraph.new()
      |> Hypergraph.add_hyperedge([1, 2])
      |> Hypergraph.add_hyperedge([2, 3])

    rules = [
      %{name: "a", pattern: [[1, 2]], replacement: [[1, :x], [:x, 2]]},
      %{name: "b", pattern: [[1, 2]], replacement: [[1, 2, :y]]}
    ]

    WolframModel.new(hg, rules)
  end

  test "multiway_explore_dag/2 returns a root node" do
    model = simple_model()
    dag = WolframModel.multiway_explore_dag(model, 2)
    assert Map.has_key?(dag, :root)
    assert Map.has_key?(dag, :nodes)
    assert Map.has_key?(dag, :edges)
    assert Map.has_key?(dag.nodes, dag.root)
  end

  test "multiway_explore_dag/2 depth 0 has only root with no edges" do
    model = simple_model()
    dag = WolframModel.multiway_explore_dag(model, 0)
    assert map_size(dag.nodes) == 1
    assert MapSet.size(dag.edges) == 0
  end

  test "multiway_explore_dag/2 explores multiple states" do
    model = simple_model()
    dag = WolframModel.multiway_explore_dag(model, 2)
    assert map_size(dag.nodes) > 1
    assert MapSet.size(dag.edges) > 0
  end

  test "multiway_explore_dag/2 edges reference existing nodes" do
    model = simple_model()
    dag = WolframModel.multiway_explore_dag(model, 2)

    Enum.each(dag.edges, fn {from, to} ->
      assert Map.has_key?(dag.nodes, from), "source #{inspect(from)} not in nodes"
      assert Map.has_key?(dag.nodes, to), "target #{inspect(to)} not in nodes"
    end)
  end

  test "multiway_explore_dag/2 does not revisit already-seen states" do
    # Build a rule that always produces the same canonical result.
    hg = Hypergraph.new() |> Hypergraph.add_hyperedge([1, 2])
    rule = %{name: "id", pattern: [[1, 2]], replacement: [[1, 2]]}
    model = WolframModel.new(hg, [rule])

    dag = WolframModel.multiway_explore_dag(model, 3)
    # The state never changes — DAG should have at most 1 node.
    assert map_size(dag.nodes) == 1
  end
end
