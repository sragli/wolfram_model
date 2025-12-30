defmodule WolframModel.CoreTest do
  use ExUnit.Case, async: true
  alias WolframModel
  alias WolframModel.Example
  alias Hypergraph

  test "new initializes evolution_history" do
    m = Example.simple_universe()
    assert length(m.evolution_history) == 1
  end

  test "evolve_step applies a rule and increments generation" do
    m = Example.simple_universe()
    m2 = WolframModel.evolve_step(m)

    assert m2.generation == 1
    assert Hypergraph.hyperedge_count(m2.hypergraph) == 3
    assert length(m2.evolution_history) == 2
    assert length(m2.causal_network) == 1

    event = hd(m2.causal_network)
    assert Map.get(event.rule, :name) == "binary_split"
  end

  test "multiway_step returns possible next states" do
    m = Example.simple_universe()
    next_states = WolframModel.multiway_step(m)
    assert is_list(next_states)
    assert length(next_states) >= 1
  end

  test "analyze_emergence returns numeric stats" do
    m = Example.simple_universe()
    em = WolframModel.Analytics.analyze_emergence(m)
    assert is_map(em)
    assert Map.has_key?(em, :vertex_count)
    assert is_number(Map.get(em, :vertex_count))
    assert Map.has_key?(em, :hyperedge_count)
    assert is_number(Map.get(em, :hyperedge_count))
  end
end
