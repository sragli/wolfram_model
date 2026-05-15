defmodule WolframModel.AnalyticsTest do
  use ExUnit.Case, async: true
  alias Hypergraph
  alias WolframModel.Analytics

  test "build_adjacency_map produces correct neighbor sets" do
    hg =
      Hypergraph.new()
      |> Hypergraph.add_hyperedge([1, 2])
      |> Hypergraph.add_hyperedge([2, 3])

    adj = Analytics.build_adjacency_map(hg)

    assert Map.get(adj, 1) == MapSet.new([2])
    assert Map.get(adj, 2) == MapSet.new([1, 3])
    assert Map.get(adj, 3) == MapSet.new([2])
  end

  test "clustering coefficient: triangle, 3-hyperedge, line" do
    triangle =
      Hypergraph.new()
      |> Hypergraph.add_hyperedge([1, 2])
      |> Hypergraph.add_hyperedge([2, 3])
      |> Hypergraph.add_hyperedge([1, 3])

    em = Analytics.calculate_clustering_coefficient(Analytics.build_adjacency_map(triangle))
    assert Float.round(em, 6) == 1.0

    single3 = Hypergraph.new() |> Hypergraph.add_hyperedge([:a, :b, :c])
    em2 = Analytics.calculate_clustering_coefficient(Analytics.build_adjacency_map(single3))
    assert Float.round(em2, 6) == 1.0

    line =
      Hypergraph.new() |> Hypergraph.add_hyperedge([1, 2]) |> Hypergraph.add_hyperedge([2, 3])

    el = Analytics.calculate_clustering_coefficient(Analytics.build_adjacency_map(line))
    assert Float.round(el, 6) == 0.0
  end

  test "estimate_diameter for triangle, line, and isolated vertex" do
    triangle =
      Hypergraph.new()
      |> Hypergraph.add_hyperedge([1, 2])
      |> Hypergraph.add_hyperedge([2, 3])
      |> Hypergraph.add_hyperedge([1, 3])

    assert Analytics.estimate_diameter(Analytics.build_adjacency_map(triangle)) == 1

    line =
      Hypergraph.new() |> Hypergraph.add_hyperedge([1, 2]) |> Hypergraph.add_hyperedge([2, 3])

    assert Analytics.estimate_diameter(Analytics.build_adjacency_map(line)) == 2

    isolated = Hypergraph.new() |> Hypergraph.add_vertex(:a)
    assert Analytics.estimate_diameter(Analytics.build_adjacency_map(isolated)) == 1
  end

  test "calculate_growth_rate computes recent/previous change" do
    hg1 = Hypergraph.new() |> Hypergraph.add_hyperedge([:a])
    hg2 = Hypergraph.new() |> Hypergraph.add_hyperedge([1, 2]) |> Hypergraph.add_hyperedge([2, 3])

    vertex_count = fn hg ->
      hg |> Hypergraph.hyperedges() |> Enum.flat_map(& &1) |> MapSet.new() |> MapSet.size()
    end

    model = WolframModel.new(hg1, [])
    model = %{model | evolution_history: [hg2, hg1]}

    assert Analytics.calculate_growth_rate(model) ==
             (vertex_count.(hg2) - vertex_count.(hg1)) /
               vertex_count.(hg1)
  end

  test "calculate_complexity returns 0 when CorrelationLength absent" do
    hg = Hypergraph.new() |> Hypergraph.add_hyperedge([1, 2])
    assert Analytics.calculate_complexity(hg) == 0
  end
end
