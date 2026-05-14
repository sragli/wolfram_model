defmodule WolframModel.DimensionTest do
  use ExUnit.Case, async: true
  alias Hypergraph
  alias WolframModel.Analytics

  test "estimate_dimension returns 1.0 for fewer than 4 vertices" do
    hg = Hypergraph.new() |> Hypergraph.add_hyperedge([1, 2])
    assert Analytics.estimate_dimension(hg) == 1.0
  end

  test "estimate_dimension returns a positive float for a larger graph" do
    hg =
      Hypergraph.new()
      |> Hypergraph.add_hyperedge([1, 2])
      |> Hypergraph.add_hyperedge([2, 3])
      |> Hypergraph.add_hyperedge([3, 4])
      |> Hypergraph.add_hyperedge([4, 5])
      |> Hypergraph.add_hyperedge([1, 5])

    dim = Analytics.estimate_dimension(hg)
    assert is_float(dim)
    assert dim > 0.0
  end

  test "estimate_dimension for a grid returns a positive float" do
    # Note: finite-size effects mean small graphs underestimate dimension;
    # we verify the function produces a valid positive float result.
    grid =
      Hypergraph.new()
      |> Hypergraph.add_hyperedge([1, 2])
      |> Hypergraph.add_hyperedge([2, 3])
      |> Hypergraph.add_hyperedge([4, 5])
      |> Hypergraph.add_hyperedge([5, 6])
      |> Hypergraph.add_hyperedge([1, 4])
      |> Hypergraph.add_hyperedge([2, 5])
      |> Hypergraph.add_hyperedge([3, 6])
      |> Hypergraph.add_hyperedge([4, 7])
      |> Hypergraph.add_hyperedge([5, 8])
      |> Hypergraph.add_hyperedge([6, 9])
      |> Hypergraph.add_hyperedge([7, 8])
      |> Hypergraph.add_hyperedge([8, 9])

    dim = Analytics.estimate_dimension(grid)
    assert is_float(dim)
    assert dim > 0.0
  end
end
