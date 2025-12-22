defmodule WolframModel.RuleSetTest do
  use ExUnit.Case, async: true
  alias WolframModel.RuleSet

  test "basic_rules contain binary_split and triangle_completion" do
    names = RuleSet.basic_rules() |> Enum.map(& &1.name)
    assert "binary_split" in names
    assert "triangle_completion" in names
  end

  test "rule_set(:growth) returns growth rules" do
    rules = RuleSet.rule_set(:growth)
    assert Enum.any?(rules, &(&1.name == "growth_split"))
  end
end
