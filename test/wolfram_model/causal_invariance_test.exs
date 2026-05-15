defmodule WolframModel.CausalInvarianceTest do
  use ExUnit.Case, async: true
  alias WolframModel
  alias Hypergraph

  test "disjoint rule applications on separate edges commute" do
    # Two disconnected edges — both rules apply independently to different edges
    hg =
      Hypergraph.new()
      |> Hypergraph.add_hyperedge([1, 2])
      |> Hypergraph.add_hyperedge([3, 4])

    rule = %{
      pattern: [[:a, :b]],
      replacement: [[:a, :b]],
      name: "identity"
    }

    model = WolframModel.new(hg, [rule])
    assert WolframModel.causally_invariant?(model) == true
  end

  test "no matches means trivially invariant" do
    hg = Hypergraph.new() |> Hypergraph.add_hyperedge([1, 2])
    model = WolframModel.new(hg, [])
    assert WolframModel.causally_invariant?(model) == true
  end

  test "returns boolean" do
    hg = Hypergraph.new() |> Hypergraph.add_hyperedge([1, 2]) |> Hypergraph.add_hyperedge([2, 3])
    rule = %{pattern: [[:a, :b]], replacement: [[:a]], name: "shrink"}
    model = WolframModel.new(hg, [rule])
    result = WolframModel.causally_invariant?(model)
    assert is_boolean(result)
  end
end
