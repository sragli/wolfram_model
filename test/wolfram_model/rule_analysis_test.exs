defmodule WolframModel.RuleAnalysisTest do
  use ExUnit.Case, async: true
  alias WolframModel.RuleAnalysis
  alias WolframModel.RuleSet

  test "reversible? returns true when hyperedge sizes match" do
    rule = %{
      name: "reversible",
      pattern: [[1, 2], [2, 3]],
      replacement: [[1, 3], [3, 4]]
    }

    assert RuleAnalysis.reversible?(rule) == true
  end

  test "reversible? returns false when sizes differ" do
    rule = %{
      name: "non_reversible",
      pattern: [[1, 2]],
      replacement: [[1, 2], [2, 3]]
    }

    assert RuleAnalysis.reversible?(rule) == false
  end

  test "self_complementary? true for same count and sizes" do
    rule = %{
      name: "sc",
      pattern: [[1, 2]],
      replacement: [[1, 3]]
    }

    assert RuleAnalysis.self_complementary?(rule) == true
  end

  test "self_complementary? false when count differs" do
    rule = %{
      name: "not_sc",
      pattern: [[1, 2]],
      replacement: [[1, 2], [2, 3]]
    }

    assert RuleAnalysis.self_complementary?(rule) == false
  end

  test "introduces_new_vertices? true when replacement has unbound atom" do
    rule = %{
      name: "new_v",
      pattern: [[:a, :b]],
      replacement: [[:a, :new]]
    }

    assert RuleAnalysis.introduces_new_vertices?(rule) == true
  end

  test "introduces_new_vertices? false when replacement only uses pattern atoms" do
    rule = %{
      name: "no_new",
      pattern: [[:a, :b]],
      replacement: [[:a, :b]]
    }

    assert RuleAnalysis.introduces_new_vertices?(rule) == false
  end

  test "hyperedge_delta is positive for growth rules" do
    [rule | _] = RuleSet.rule_set(:growth)
    assert RuleAnalysis.hyperedge_delta(rule) > 0
  end

  test "hyperedge_delta is zero for same-count rules" do
    rule = %{
      name: "swap",
      pattern: [[1, 2]],
      replacement: [[1, 3]]
    }

    assert RuleAnalysis.hyperedge_delta(rule) == 0
  end

  test "arity returns sorted size tuples" do
    rule = %{
      name: "mixed",
      pattern: [[1, 2, 3], [1, 2]],
      replacement: [[1, 2]]
    }

    assert RuleAnalysis.arity(rule) == {[2, 3], [2]}
  end
end
