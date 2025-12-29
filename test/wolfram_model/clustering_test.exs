defmodule WolframModel.ClusteringTest do
  use ExUnit.Case, async: true
  alias Hypergraph
  alias WolframModel

  test "triangle of pairwise edges has clustering 1.0" do
    hg =
      Hypergraph.new()
      |> Hypergraph.add_hyperedge([1, 2])
      |> Hypergraph.add_hyperedge([2, 3])
      |> Hypergraph.add_hyperedge([1, 3])

    model = WolframModel.new(hg, [])
    em = WolframModel.analyze_emergence(model)

    assert Float.round(em.clustering_coefficient, 6) == 1.0
  end

  test "single hyperedge of size 3 has clustering 1.0" do
    hg = Hypergraph.new() |> Hypergraph.add_hyperedge([:a, :b, :c])
    model = WolframModel.new(hg, [])
    em = WolframModel.analyze_emergence(model)

    assert Float.round(em.clustering_coefficient, 6) == 1.0
  end

  test "line graph has clustering 0.0" do
    hg = Hypergraph.new() |> Hypergraph.add_hyperedge([1, 2]) |> Hypergraph.add_hyperedge([2, 3])
    model = WolframModel.new(hg, [])
    em = WolframModel.analyze_emergence(model)

    assert Float.round(em.clustering_coefficient, 6) == 0.0
  end
end
