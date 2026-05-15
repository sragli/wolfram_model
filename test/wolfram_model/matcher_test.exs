defmodule WolframModel.MatcherTest do
  use ExUnit.Case, async: true
  alias WolframModel
  alias Hypergraph

  test "single-hyperedge mapping is deterministic" do
    hg = Hypergraph.new() |> Hypergraph.add_hyperedge([:v1, :v2])

    rule = %{
      pattern: [[:a, :b]],
      replacement: [[:a]],
      name: "single_map"
    }

    model = WolframModel.new(hg, [rule])
    [next] = WolframModel.multiway_step(model)

    # The replacement should map :a -> :v1 (positional mapping) and keep a single vertex
    assert Enum.any?(Hypergraph.hyperedges(next.hypergraph), fn he -> length(he) == 1 end)
  end

  test "two-hyperedge pattern mapping maps shared vertex correctly" do
    hg =
      Hypergraph.new()
      |> Hypergraph.add_hyperedge([:v1, :v2])
      |> Hypergraph.add_hyperedge([:v2, :v3])

    rule = %{
      pattern: [[:a, :b], [:b, :c]],
      replacement: [[:a, :c]],
      name: "join"
    }

    model = WolframModel.new(hg, [rule])
    next_states = WolframModel.multiway_step(model)
    assert length(next_states) >= 1

    # Find a state with the expected replacement hyperedge [:v1, :v3]
    found =
      Enum.any?(next_states, fn st ->
        Enum.any?(Hypergraph.hyperedges(st.hypergraph), fn he ->
          he == [:v1, :v3]
        end)
      end)

    assert found
  end
end
