defmodule WolframModel.WolframRulesTest do
  use ExUnit.Case, async: true
  alias WolframModel
  alias WolframModel.RuleSet
  alias Hypergraph

  test "rule_set(:wolfram, :rule_1) is a list with one rule" do
    rules = RuleSet.rule_set(:wolfram, :rule_1)
    assert is_list(rules)
    assert length(rules) == 1
    assert hd(rules).name == "wolfram_rule_1"
  end

  test "wolfram_rules/0 returns all 5 benchmark rules" do
    all = RuleSet.wolfram_rules()
    assert map_size(all) == 5
    assert Enum.all?([:rule_1, :rule_2, :rule_3, :rule_4, :rule_5], &Map.has_key?(all, &1))
  end

  test "wolfram rule_1 evolves a matching initial state" do
    rules = RuleSet.rule_set(:wolfram, :rule_1)

    hg =
      Hypergraph.new()
      |> Hypergraph.add_hyperedge([1, 2])
      |> Hypergraph.add_hyperedge([1, 3])

    model = WolframModel.new(hg, rules)
    evolved = WolframModel.evolve_step(model)
    assert evolved.generation == 1
  end

  test "wolfram rule_2 (triangle) evolves a triangle initial state" do
    rules = RuleSet.rule_set(:wolfram, :rule_2)

    hg =
      Hypergraph.new()
      |> Hypergraph.add_hyperedge([1, 2])
      |> Hypergraph.add_hyperedge([1, 3])
      |> Hypergraph.add_hyperedge([2, 3])

    model = WolframModel.new(hg, rules)
    evolved = WolframModel.evolve_step(model)
    assert evolved.generation == 1
    # Triangle rule adds a center vertex connected to all three corners
    edges = Hypergraph.hyperedges(evolved.hypergraph)
    # Should have [1,2,3] (3-hyperedge) plus 3 binary edges to the center
    assert Enum.any?(edges, fn he -> length(he) == 3 end)
  end

  test "wolfram_rules/0 raises on unknown key" do
    assert_raise KeyError, fn -> RuleSet.rule_set(:wolfram, :nonexistent) end
  end
end
