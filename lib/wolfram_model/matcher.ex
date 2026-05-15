defmodule WolframModel.Matcher do
  @moduledoc """
  Deterministic pattern matcher for WolframModel.

  Exposes a general `match/2` that handles rules with any number of pattern
  hyperedges (1, 2, or N) via recursive backtracking. Patterns are matched
  positionally: the element at position i in the pattern binds to the element
  at position i in the matched hyperedge. Shared variables across patterns must
  map to the same vertex in every matched hyperedge.

  Hyperedges are ordered lists. Each pattern variable can be any term; atoms
  and integers are treated as variable names to bind.
  """

  alias Hypergraph

  @type match_result :: %{mapping: map(), matched_hyperedges: [Hypergraph.hyperedge()]}

  @doc """
  Finds all ways to match `patterns` against `hyperedges` with consistent
  variable bindings. Supports patterns of any length (1, 2, or N hyperedges).

  Returns a list of `%{mapping: map(), matched_hyperedges: [hyperedge()]}`.
  Each matched_hyperedges list has the same length as `patterns` and is in the
  same order. Each distinct hyperedge in the graph may only be used once per
  match (no self-overlap).
  """
  @spec match([Hypergraph.hyperedge()], [Hypergraph.hyperedge()]) :: [match_result()]
  def match(_hyperedges, []), do: []

  def match(hyperedges, patterns) when is_list(patterns) do
    do_match_patterns(hyperedges, patterns, %{}, [])
  end

  @doc """
  Builds a merged mapping from two (pattern, hyperedge) pairs.
  Kept for backward compatibility.
  """
  @spec build_mapping_for_two([term()], [term()], [term()], [term()]) :: map()
  def build_mapping_for_two(p1, p2, he1, he2) do
    map1 = Enum.zip(p1, he1) |> Map.new()
    map2 = Enum.zip(p2, he2) |> Map.new()
    Map.merge(map1, map2)
  end

  # Base case: all patterns consumed — emit the result.
  defp do_match_patterns(_hyperedges, [], mapping, matched) do
    [%{mapping: mapping, matched_hyperedges: Enum.reverse(matched)}]
  end

  # Recursive step: pick a hyperedge for the next pattern and recurse.
  defp do_match_patterns(hyperedges, [pattern | rest_patterns], mapping, matched) do
    pattern_size = length(pattern)

    hyperedges
    |> Enum.filter(fn he ->
      length(he) == pattern_size and he not in matched
    end)
    |> Enum.flat_map(fn candidate ->
      candidate_mapping = Enum.zip(pattern, candidate) |> Map.new()

      case merge_consistent(mapping, candidate_mapping) do
        {:ok, merged} ->
          do_match_patterns(hyperedges, rest_patterns, merged, [candidate | matched])

        :error ->
          []
      end
    end)
  end

  # Merges `additions` into `existing`, returning {:ok, merged} when all
  # shared keys agree on the same value, or :error on contradiction.
  defp merge_consistent(existing, additions) do
    Enum.reduce_while(additions, {:ok, existing}, fn {k, v}, {:ok, acc} ->
      case Map.fetch(acc, k) do
        {:ok, ^v} -> {:cont, {:ok, acc}}
        {:ok, _other} -> {:halt, :error}
        :error -> {:cont, {:ok, Map.put(acc, k, v)}}
      end
    end)
  end
end
