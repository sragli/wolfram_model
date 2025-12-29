defmodule WolframModel.Matcher do
  @moduledoc """
  Deterministic pattern matcher for WolframModel.

  Exposes functions to match single- and two-hyperedge patterns against a list
  of hyperedges and to build consistent mappings between pattern placeholders
  and actual vertices.
  """

  alias Hypergraph

  @spec match([MapSet.t()], [MapSet.t()]) ::
          %{mapping: map(), matched_hyperedges: [MapSet.t()]} | nil
  def match(hyperedges, [single_pattern]) do
    match_single(hyperedges, single_pattern)
  end

  @spec match([MapSet.t()], [MapSet.t()]) ::
          %{mapping: map(), matched_hyperedges: [MapSet.t()]} | nil
  def match(hyperedges, [p1, p2]) do
    match_two(hyperedges, p1, p2)
  end

  @spec match_all([MapSet.t()], [MapSet.t()]) :: [map()]
  def match_all(hyperedges, [single_pattern]) do
    match_all_single(hyperedges, single_pattern)
  end

  @spec match_all([MapSet.t()], [MapSet.t()]) :: [map()]
  def match_all(hyperedges, [p1, p2]) do
    match_all_two(hyperedges, p1, p2)
  end

  @spec match_single([MapSet.t()], MapSet.t()) ::
          %{mapping: map(), matched_hyperedges: [MapSet.t()]} | nil
  def match_single(hyperedges, pattern) do
    pattern_size = MapSet.size(pattern)

    hyperedges
    |> Enum.find(fn he -> MapSet.size(he) == pattern_size end)
    |> case do
      nil ->
        nil

      matched_he ->
        pattern_list = Enum.sort(MapSet.to_list(pattern))
        matched_list = Enum.sort(MapSet.to_list(matched_he))
        mapping = Enum.zip(pattern_list, matched_list) |> Map.new()
        %{mapping: mapping, matched_hyperedges: [matched_he]}
    end
  end

  @spec match_all_single([MapSet.t()], MapSet.t()) :: [map()]
  def match_all_single(hyperedges, pattern) do
    pattern_size = MapSet.size(pattern)

    hyperedges
    |> Enum.filter(fn he -> MapSet.size(he) == pattern_size end)
    |> Enum.map(fn matched_he ->
      pattern_list = Enum.sort(MapSet.to_list(pattern))
      matched_list = Enum.sort(MapSet.to_list(matched_he))
      mapping = Enum.zip(pattern_list, matched_list) |> Map.new()
      %{mapping: mapping, matched_hyperedges: [matched_he]}
    end)
  end

  @spec match_two([MapSet.t()], MapSet.t(), MapSet.t()) ::
          %{mapping: map(), matched_hyperedges: [MapSet.t()]} | nil
  def match_two(hyperedges, p1, p2) do
    p1_size = MapSet.size(p1)
    p2_size = MapSet.size(p2)

    for he1 <- hyperedges,
        he2 <- hyperedges,
        he1 != he2,
        MapSet.size(he1) == p1_size,
        MapSet.size(he2) == p2_size,
        !MapSet.disjoint?(he1, he2) do
      mapping = build_mapping_for_two(p1, p2, he1, he2)

      if mapping != nil and map_size(mapping) > 0 do
        %{mapping: mapping, matched_hyperedges: [he1, he2]}
      end
    end
    |> List.first()
  end

  @spec match_all_two([MapSet.t()], MapSet.t(), MapSet.t()) :: [map()]
  def match_all_two(hyperedges, p1, p2) do
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

  @spec build_mapping_for_two(MapSet.t(), MapSet.t(), MapSet.t(), MapSet.t()) :: map() | nil
  def build_mapping_for_two(p1, p2, he1, he2) do
    p1_list = Enum.sort(MapSet.to_list(p1))
    p2_list = Enum.sort(MapSet.to_list(p2))
    he1_list = Enum.sort(MapSet.to_list(he1))
    he2_list = Enum.sort(MapSet.to_list(he2))

    p_shared = Enum.sort(MapSet.to_list(MapSet.intersection(p1, p2)))
    he_shared = Enum.sort(MapSet.to_list(MapSet.intersection(he1, he2)))

    # Start by mapping shared pattern placeholders to shared actual vertices in sorted order
    shared_mapping = Enum.zip(p_shared, he_shared) |> Map.new()

    # Map remaining pattern elements in p1 to remaining actual vertices in he1
    p1_remaining = p1_list -- p_shared
    he1_remaining = he1_list -- he_shared

    map_p1 = Enum.zip(p1_remaining, he1_remaining) |> Map.new()

    # Map remaining pattern elements in p2 to remaining actual vertices in he2
    p2_remaining = p2_list -- p_shared
    he2_remaining = he2_list -- he_shared

    map_p2 = Enum.zip(p2_remaining, he2_remaining) |> Map.new()

    Map.merge(shared_mapping, Map.merge(map_p1, map_p2))
  end
end
