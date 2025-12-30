defmodule WolframModel.DiameterTest do
  use ExUnit.Case, async: true
  alias Hypergraph
  alias WolframModel

  test "triangle has diameter 1" do
    hg =
      Hypergraph.new()
      |> Hypergraph.add_hyperedge([1, 2])
      |> Hypergraph.add_hyperedge([2, 3])
      |> Hypergraph.add_hyperedge([1, 3])

    model = WolframModel.new(hg, [])
    em = WolframModel.Analytics.analyze_emergence(model)

    assert em.estimated_diameter == 1
  end

  test "line of three has diameter 2" do
    hg = Hypergraph.new() |> Hypergraph.add_hyperedge([1, 2]) |> Hypergraph.add_hyperedge([2, 3])
    model = WolframModel.new(hg, [])
    em = WolframModel.Analytics.analyze_emergence(model)

    assert em.estimated_diameter == 2
  end

  test "isolated vertex returns diameter 1" do
    hg = Hypergraph.new() |> Hypergraph.add_vertex(:a)
    model = WolframModel.new(hg, [])
    em = WolframModel.Analytics.analyze_emergence(model)

    assert em.estimated_diameter == 1
  end
end
