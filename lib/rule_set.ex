defmodule RuleSet do
  @moduledoc """
  Rule sets responsible for the evolution of the Wolfram Model.
  """

  @type rule :: %{
          pattern: [MapSet.t()],
          replacement: [MapSet.t()],
          name: String.t()
        }

  @doc """
  Defines a set of basic evolution rules for experimentation.
  """
  @spec basic_rules() :: [rule()]
  def basic_rules do
    [
      # Rule 1: Binary split - edge becomes two edges with new vertex
      %{
        name: "binary_split",
        pattern: [MapSet.new([1, 2])],
        replacement: [MapSet.new([1, :new]), MapSet.new([:new, 2])]
      },

      # Rule 2: Triangle completion - two edges sharing vertex become triangle
      %{
        name: "triangle_completion",
        pattern: [MapSet.new([1, 2]), MapSet.new([2, 3])],
        replacement: [MapSet.new([1, 2, 3])]
      },

      # Rule 3: Triangle split - triangle becomes three edges with center
      %{
        name: "triangle_split",
        pattern: [MapSet.new([1, 2, 3])],
        replacement: [
          MapSet.new([1, :center]),
          MapSet.new([2, :center]),
          MapSet.new([3, :center])
        ]
      },

      # Rule 4: Edge duplication - single edge becomes parallel edges
      %{
        name: "edge_duplication",
        pattern: [MapSet.new([1, 2])],
        replacement: [MapSet.new([1, 2]), MapSet.new([1, 2, :parallel])]
      },

      # Rule 5: Four-cycle formation
      %{
        name: "four_cycle",
        pattern: [MapSet.new([1, 2]), MapSet.new([3, 4])],
        replacement: [MapSet.new([1, 2, 3, 4])]
      }
    ]
  end

  @doc """
  Creates some interesting specialized rule sets.
  """
  @spec rule_set(atom()) :: [rule()]
  def rule_set(:growth) do
    [
      %{
        name: "growth_split",
        pattern: [MapSet.new([1, 2])],
        replacement: [MapSet.new([1, :new]), MapSet.new([:new, 2]), MapSet.new([1, 2])]
      },
      %{
        name: "growth_expand",
        pattern: [MapSet.new([1, 2, 3])],
        replacement: [
          MapSet.new([1, 2, 3]),
          MapSet.new([1, :new1]),
          MapSet.new([2, :new2]),
          MapSet.new([3, :new3])
        ]
      }
    ]
  end

  def rule_set(:cellular_automaton) do
    [
      # Conway-like rules adapted for hypergraphs
      %{
        name: "survival",
        pattern: [MapSet.new([1, 2]), MapSet.new([1, 3])],
        replacement: [MapSet.new([1, 2, 3])]
      },
      %{
        name: "death",
        pattern: [MapSet.new([1, 2, 3, 4])],
        replacement: [MapSet.new([1, 2]), MapSet.new([3, 4])]
      }
    ]
  end

  def rule_set(:spacetime) do
    [
      # Rules that might generate spacetime-like structures
      %{
        name: "time_step",
        pattern: [MapSet.new([:t, :x])],
        replacement: [MapSet.new([:t, :x]), MapSet.new([{:t, 1}, :x])]
      },
      %{
        name: "space_connection",
        pattern: [MapSet.new([{:t, 0}, :x1]), MapSet.new([{:t, 0}, :x2])],
        replacement: [
          MapSet.new([{:t, 0}, :x1]),
          MapSet.new([{:t, 0}, :x2]),
          MapSet.new([{:t, 0}, :x1, :x2])
        ]
      }
    ]
  end
end
