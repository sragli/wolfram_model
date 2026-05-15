defmodule WolframModel.Matcher do
  @moduledoc """
  Deterministic pattern matcher for WolframModel.

  Exposes functions to match single- and two-hyperedge patterns against a list
  of hyperedges and to build consistent mappings between pattern placeholders
  and actual vertices.

  Hyperedges are ordered lists. Patterns are matched positionally: the element
  at position i in the pattern binds to the element at position i in the
  hyperedge. Shared pattern variables across two patterns must map to the same
  vertex in both matched hyperedges.
  """

  alias Hypergraph

  @type match_result :: %{mapping: map(), matched_hyperedges: [Hypergraph.hyperedge()]}

  @spec match([Hypergraph.hyperedge()], [Hypergraph.hyperedge()]) :: [match_result()]
  def match(hyperedges, [single_pattern]) do
    pattern_size = length(single_pattern)

    hyperedges
    |> Enum.filter(fn he -> length(he) == pattern_size end)
    |> Enum.map(fn matched_he ->
      mapping = Enum.zip(single_pattern, matched_he) |> Map.new()
      %{mapping: mapping, matched_hyperedges: [matched_he]}
    end)
  end

  def match(hyperedges, [p1, p2]) do
    p1_size = length(p1)
    p2_size = length(p2)

    for he1 <- hyperedges,
        he2 <- hyperedges,
        he1 != he2,
        length(he1) == p1_size,
        length(he2) == p2_size,
        shares_vertex?(he1, he2) do
      map1 = Enum.zip(p1, he1) |> Map.new()
      map2 = Enum.zip(p2, he2) |> Map.new()

      if consistent_mappings?(map1, map2) do
        %{mapping: Map.merge(map1, map2), matched_hyperedges: [he1, he2]}
      end
    end
    |> Enum.reject(&is_nil/1)
  end

  # Fallback for unsupported patterns
  @spec match(term(), term()) :: []
  def match(_hyperedges, _pattern), do: []

  @spec build_mapping_for_two([term()], [term()], [term()], [term()]) :: map()
  def build_mapping_for_two(p1, p2, he1, he2) do
    map1 = Enum.zip(p1, he1) |> Map.new()
    map2 = Enum.zip(p2, he2) |> Map.new()
    Map.merge(map1, map2)
  end

  # Returns true if the two hyperedges share at least one vertex.
  defp shares_vertex?(he1, he2) do
    he1 -- he1 -- he2 != []
  end

  # Returns true if no key shared between map1 and map2 maps to different values.
  defp consistent_mappings?(map1, map2) do
    Enum.all?(map1, fn {k, v} -> not Map.has_key?(map2, k) or map2[k] == v end)
  end
end
