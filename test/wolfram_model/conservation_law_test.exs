defmodule WolframModel.ConservationLawTest do
  use ExUnit.Case, async: true
  alias WolframModel
  alias WolframModel.Analytics
  alias Hypergraph

  test "detect_conserved_quantities/1 returns empty list for < 2 snapshots" do
    hg = Hypergraph.new() |> Hypergraph.add_hyperedge([1, 2])
    model = WolframModel.new(hg, [])
    result = Analytics.detect_conserved_quantities(model)
    assert result.conserved == []
  end

  test "detect_conserved_quantities/1 identifies edge_count_parity conservation" do
    hg =
      Hypergraph.new()
      |> Hypergraph.add_hyperedge([1, 2])
      |> Hypergraph.add_hyperedge([2, 3])

    # Rule: one edge becomes two edges — edge count grows but parity alternates
    rule = %{
      name: "split",
      pattern: [[1, 2]],
      replacement: [[1, :new], [:new, 2]]
    }

    model = WolframModel.new(hg, [rule]) |> WolframModel.evolve_steps(4)
    result = Analytics.detect_conserved_quantities(model)

    assert is_list(result.conserved)
    assert is_list(result.vertex_count_history)
    assert is_list(result.edge_count_history)
    assert length(result.vertex_count_history) >= 2
  end

  test "detect_conserved_quantities/1 detects vertex_count conservation for restructuring rule" do
    # Rule: rearranges edges without creating new vertices
    hg =
      Hypergraph.new()
      |> Hypergraph.add_hyperedge([1, 2])
      |> Hypergraph.add_hyperedge([2, 3])

    rule = %{
      name: "rearrange",
      pattern: [[1, 2], [2, 3]],
      replacement: [[1, 3], [1, 2]]
    }

    model = WolframModel.new(hg, [rule]) |> WolframModel.evolve_steps(3)
    result = Analytics.detect_conserved_quantities(model)

    assert :vertex_count in result.conserved
    assert :edge_count in result.conserved
  end
end
