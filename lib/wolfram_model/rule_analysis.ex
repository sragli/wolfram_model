defmodule WolframModel.RuleAnalysis do
  @moduledoc """
  Rule property analysis for WolframModel rules.

  Provides predicates and metrics that characterise the structural properties of
  a rule without running a full evolution — useful for classifying rules before
  deciding how to use them.
  """

  @type rule :: WolframModel.rule()

  @doc """
  Returns `true` if the rule is structurally reversible: the multiset of
  hyperedge sizes in the pattern equals the multiset in the replacement.

  A reversible rule can (at least in principle) be run "backwards" by swapping
  pattern and replacement.
  """
  @spec reversible?(rule()) :: boolean()
  def reversible?(rule) do
    pattern_sizes(rule) == replacement_sizes(rule)
  end

  @doc """
  Returns `true` if the rule is self-complementary: it has the same number of
  hyperedges in the pattern and replacement, and the same multiset of hyperedge
  sizes (i.e., the rule maps one configuration to a structurally identical one).
  """
  @spec self_complementary?(rule()) :: boolean()
  def self_complementary?(rule) do
    length(rule.pattern) == length(rule.replacement) and reversible?(rule)
  end

  @doc """
  Returns `true` if the replacement contains at least one atom that does not
  appear in the pattern — indicating the rule introduces new vertices when
  applied.
  """
  @spec introduces_new_vertices?(rule()) :: boolean()
  def introduces_new_vertices?(rule) do
    pattern_atoms =
      rule.pattern
      |> Enum.flat_map(& &1)

    rule.replacement
    |> Enum.flat_map(& &1)
    |> Enum.any?(fn v -> is_atom(v) and v not in pattern_atoms end)
  end

  @doc """
  Returns the net hyperedge count change for a single rule application.

  A positive value means the rule grows the hypergraph, zero means it
  restructures without changing edge count, and negative means it shrinks it.
  """
  @spec hyperedge_delta(rule()) :: integer()
  def hyperedge_delta(rule) do
    length(rule.replacement) - length(rule.pattern)
  end

  @doc """
  Returns the arity of a rule as `{pattern_sizes, replacement_sizes}` where
  each element is a sorted list of hyperedge sizes.

  Useful for quickly comparing the "shape" of different rules.
  """
  @spec arity(rule()) :: {[non_neg_integer()], [non_neg_integer()]}
  def arity(rule) do
    {pattern_sizes(rule), replacement_sizes(rule)}
  end

  # --- private helpers ---

  defp pattern_sizes(rule),
    do: rule.pattern |> Enum.map(&length/1) |> Enum.sort()

  defp replacement_sizes(rule),
    do: rule.replacement |> Enum.map(&length/1) |> Enum.sort()
end
