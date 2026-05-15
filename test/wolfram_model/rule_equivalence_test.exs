defmodule WolframModel.RuleEquivalenceTest do
  use ExUnit.Case, async: true
  alias WolframModel.RuleAnalysis

  test "equivalent?/2 returns true for structurally identical rules" do
    r1 = %{name: "a", pattern: [[1, 2], [2, 3]], replacement: [[1, 3]]}
    r2 = %{name: "b", pattern: [[1, 2], [2, 3]], replacement: [[1, 3]]}
    assert RuleAnalysis.equivalent?(r1, r2)
  end

  test "equivalent?/2 returns true for rules differing only in variable names" do
    r1 = %{name: "a", pattern: [[1, 2], [2, 3]], replacement: [[1, 3]]}
    r2 = %{name: "b", pattern: [[10, 20], [20, 30]], replacement: [[10, 30]]}
    assert RuleAnalysis.equivalent?(r1, r2)
  end

  test "equivalent?/2 returns false for structurally different rules" do
    r1 = %{name: "a", pattern: [[1, 2]], replacement: [[1, 3], [3, 2]]}
    r2 = %{name: "b", pattern: [[1, 2]], replacement: [[1, 3]]}
    refute RuleAnalysis.equivalent?(r1, r2)
  end

  test "canonical_form/1 is stable across different variable numbering" do
    r1 = %{name: "x", pattern: [[1, 2], [2, 3]], replacement: [[1, 3]]}
    r2 = %{name: "y", pattern: [[5, 6], [6, 7]], replacement: [[5, 7]]}
    assert RuleAnalysis.canonical_form(r1) == RuleAnalysis.canonical_form(r2)
  end

  test "canonical_form/1 distinguishes rules with different structure" do
    # Pattern has shared variable at position 1 (same left vertex)
    r1 = %{name: "x", pattern: [[1, 2], [1, 3]], replacement: [[2, 3]]}
    # Pattern has shared variable at position 2/1 (shared middle)
    r2 = %{name: "y", pattern: [[1, 2], [2, 3]], replacement: [[1, 3]]}
    refute RuleAnalysis.canonical_form(r1) == RuleAnalysis.canonical_form(r2)
  end

  test "canonical_form/1 assigns new-vertex tags after shared variables" do
    r = %{name: "z", pattern: [[1, 2]], replacement: [[1, :new], [:new, 2]]}
    cf = RuleAnalysis.canonical_form(r)
    # Variables in first-appearance order: 1 -> 1, 2 -> 2, :new -> 3
    assert cf.pattern == [[1, 2]]
    assert cf.replacement == [[1, 3], [3, 2]]
  end
end
