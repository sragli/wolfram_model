defmodule WolframModel.RuleNotationTest do
  use ExUnit.Case, async: true
  alias WolframModel.Rule

  test "parse/1 handles integer vertices" do
    rule = Rule.parse("{{1,2},{1,3}} -> {{2,3},{1,4}}")
    assert rule.pattern == [[1, 2], [1, 3]]
    assert rule.replacement == [[2, 3], [1, 4]]
    assert rule.name == "parsed"
  end

  test "parse/1 handles symbolic atom vertices" do
    rule = Rule.parse("{{x,y},{y,z}} -> {{x,z}}")
    assert rule.pattern == [[:x, :y], [:y, :z]]
    assert rule.replacement == [[:x, :z]]
  end

  test "parse/1 handles single-element hyperedges" do
    rule = Rule.parse("{{1,2}} -> {{1},{2}}")
    assert rule.pattern == [[1, 2]]
    assert rule.replacement == [[1], [2]]
  end

  test "parse/1 accepts a custom rule name" do
    rule = Rule.parse("{{1,2}} -> {{1,2}}", "identity")
    assert rule.name == "identity"
  end

  test "to_string/1 formats integer rule correctly" do
    rule = %{pattern: [[1, 2], [2, 3]], replacement: [[1, 3]], name: "join"}
    assert Rule.to_string(rule) == "{{1,2},{2,3}} -> {{1,3}}"
  end

  test "to_string/1 formats atom rule correctly" do
    rule = %{pattern: [[:x, :y]], replacement: [[:x], [:y]], name: "split"}
    assert Rule.to_string(rule) == "{{x,y}} -> {{x},{y}}"
  end

  test "round-trip: parse -> to_string -> parse produces identical rule" do
    original = "{{1,2},{1,3}} -> {{2,3},{1,4}}"
    rule = Rule.parse(original)
    reprinted = Rule.to_string(rule)
    round_tripped = Rule.parse(reprinted)
    assert rule.pattern == round_tripped.pattern
    assert rule.replacement == round_tripped.replacement
  end

  test "parse/1 raises ArgumentError for invalid input" do
    assert_raise ArgumentError, fn -> Rule.parse("{{1,2}}") end
  end

  test "parsed rule can be used directly in WolframModel" do
    alias WolframModel
    alias Hypergraph

    rule = Rule.parse("{{1,2},{1,3}} -> {{2,3},{1,4}}")

    hg =
      Hypergraph.new()
      |> Hypergraph.add_hyperedge([1, 2])
      |> Hypergraph.add_hyperedge([1, 3])

    model = WolframModel.new(hg, [rule])
    evolved = WolframModel.evolve_step(model)
    assert evolved.generation == 1
  end
end
