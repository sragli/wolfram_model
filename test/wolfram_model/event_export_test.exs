defmodule WolframModel.EventExportTest do
  use ExUnit.Case, async: true
  alias Hypergraph
  alias WolframModel

  test "export_event_graph returns nodes and parent-child edges" do
    hg = Hypergraph.new() |> Hypergraph.add_hyperedge([1, 2]) |> Hypergraph.add_hyperedge([2, 3])

    rule = %{
      pattern: [[:a, :b]],
      replacement: [[:a, :new]],
      name: "r"
    }

    model = WolframModel.new(hg, [rule])
    m1 = WolframModel.evolve_step(model)
    m2 = WolframModel.evolve_step(m1)

    graph = WolframModel.export_event_graph(m2)
    assert is_map(graph)
    assert length(graph.nodes) == 2
    assert length(graph.edges) >= 1

    # Edges refer to existing node ids
    node_ids = Enum.map(graph.nodes, & &1.id)

    Enum.each(graph.edges, fn e ->
      assert e.source in node_ids
      assert e.target in node_ids
    end)
  end
end
