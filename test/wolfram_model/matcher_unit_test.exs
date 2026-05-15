defmodule WolframModel.MatcherUnitTest do
  use ExUnit.Case, async: true
  alias WolframModel.Matcher

  test "match_two maps shared placeholder to shared vertex deterministically" do
    p1 = [:a, :b]
    p2 = [:b, :c]

    he1 = [:v1, :v2]
    he2 = [:v2, :v3]

    mapping = Matcher.build_mapping_for_two(p1, p2, he1, he2)

    assert Map.get(mapping, :b) == :v2
    assert Map.get(mapping, :a) == :v1
    assert Map.get(mapping, :c) == :v3
  end
end
