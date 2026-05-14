defmodule WolframModel.BranchialGraphTest do
  use ExUnit.Case, async: true
  alias WolframModel
  alias Hypergraph

  test "branchial_graph returns nodes and edges map" do
    hg = Hypergraph.new() |> Hypergraph.add_hyperedge([1, 2]) |> Hypergraph.add_hyperedge([2, 3])
    rule = %{pattern: [MapSet.new([:a, :b])], replacement: [MapSet.new([:a, :new])], name: "r"}
    model = WolframModel.new(hg, [rule])

    bg = WolframModel.branchial_graph(model)
    assert is_map(bg)
    assert Map.has_key?(bg, :nodes)
    assert Map.has_key?(bg, :edges)
  end

  test "branchial_graph nodes count matches number of matches" do
    hg = Hypergraph.new() |> Hypergraph.add_hyperedge([1, 2]) |> Hypergraph.add_hyperedge([2, 3])
    rule = %{pattern: [MapSet.new([:a, :b])], replacement: [MapSet.new([:a, :new])], name: "r"}
    model = WolframModel.new(hg, [rule])

    bg = WolframModel.branchial_graph(model)
    # Two hyperedges, one rule — two possible matches
    assert length(bg.nodes) == 2
  end

  test "branchial_graph connects overlapping matches" do
    # Both edges share vertex 2, so both matches are branchially adjacent
    hg = Hypergraph.new() |> Hypergraph.add_hyperedge([1, 2]) |> Hypergraph.add_hyperedge([2, 3])
    rule = %{pattern: [MapSet.new([:a, :b])], replacement: [MapSet.new([:a, :new])], name: "r"}
    model = WolframModel.new(hg, [rule])

    bg = WolframModel.branchial_graph(model)
    assert length(bg.edges) >= 1
  end

  test "branchial_graph edges reference valid node ids" do
    hg = Hypergraph.new() |> Hypergraph.add_hyperedge([1, 2]) |> Hypergraph.add_hyperedge([2, 3])
    rule = %{pattern: [MapSet.new([:a, :b])], replacement: [MapSet.new([:a, :new])], name: "r"}
    model = WolframModel.new(hg, [rule])

    bg = WolframModel.branchial_graph(model)
    node_ids = Enum.map(bg.nodes, & &1.id)

    Enum.each(bg.edges, fn e ->
      assert e.source in node_ids
      assert e.target in node_ids
    end)
  end

  test "branchial_graph is empty for model with no applicable rules" do
    hg = Hypergraph.new() |> Hypergraph.add_hyperedge([1, 2])
    model = WolframModel.new(hg, [])
    bg = WolframModel.branchial_graph(model)
    assert bg.nodes == []
    assert bg.edges == []
  end
end
