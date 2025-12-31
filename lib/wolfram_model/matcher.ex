defmodule WolframModel.Matcher do
  @moduledoc """
  Deterministic pattern matcher for WolframModel.

  Exposes functions to match single- and two-hyperedge patterns against a list
  of hyperedges and to build consistent mappings between pattern placeholders
  and actual vertices.
  """

  alias Hypergraph

  @type match_result :: %{mapping: map(), matched_hyperedges: [MapSet.t()]}

  @spec match([MapSet.t()] | MapSet.t(MapSet.t()), [MapSet.t()]) :: [match_result()]
  def match(hyperedges, [single_pattern]) do
    pattern_size = MapSet.size(single_pattern)

    hyperedges
    |> Enum.filter(fn he -> MapSet.size(he) == pattern_size end)
    |> Enum.map(fn matched_he ->
      pattern_list = Enum.sort(single_pattern)
      matched_list = Enum.sort(matched_he)
      mapping = Enum.zip(pattern_list, matched_list) |> Map.new()
      %{mapping: mapping, matched_hyperedges: [matched_he]}
    end)
  end

  def match(hyperedges, [p1, p2]) do
    p1_size = MapSet.size(p1)
    p2_size = MapSet.size(p2)

    for he1 <- hyperedges,
        he2 <- hyperedges,
        he1 != he2,
        MapSet.size(he1) == p1_size,
        MapSet.size(he2) == p2_size,
        !MapSet.disjoint?(he1, he2) do
      mapping = build_mapping_for_two(p1, p2, he1, he2)
      %{mapping: mapping, matched_hyperedges: [he1, he2]}
    end
  end

  # Fallback for unsupported patterns
  @spec match(term(), term()) :: []
  def match(_hyperedges, _pattern), do: []

  @spec build_mapping_for_two(MapSet.t(), MapSet.t(), MapSet.t(), MapSet.t()) :: map()
  def build_mapping_for_two(p1, p2, he1, he2) do
    p_shared = Enum.sort(MapSet.intersection(p1, p2))
    he_shared = Enum.sort(MapSet.intersection(he1, he2))

    # Start by mapping shared pattern placeholders to shared actual vertices in sorted order
    shared_mapping = Enum.zip(p_shared, he_shared) |> Map.new()

    # Map remaining pattern elements in p1 to remaining actual vertices in he1
    p1_remaining = Enum.sort(p1) -- p_shared
    he1_remaining = Enum.sort(he1) -- he_shared

    map_p1 = Enum.zip(p1_remaining, he1_remaining) |> Map.new()

    # Map remaining pattern elements in p2 to remaining actual vertices in he2
    p2_remaining = Enum.sort(p2) -- p_shared
    he2_remaining = Enum.sort(he2) -- he_shared

    map_p2 = Enum.zip(p2_remaining, he2_remaining) |> Map.new()

    Map.merge(shared_mapping, Map.merge(map_p1, map_p2))
  end
end
