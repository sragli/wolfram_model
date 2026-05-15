defmodule WolframModel.RuleSet do
  @moduledoc """
  Rule sets responsible for the evolution of the Wolfram Model.
  """

  @type rule :: %{
          pattern: [[term()]],
          replacement: [[term()]],
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
        pattern: [[1, 2]],
        replacement: [[1, :new], [:new, 2]]
      },

      # Rule 2: Triangle completion - two edges sharing vertex become triangle
      %{
        name: "triangle_completion",
        pattern: [[1, 2], [2, 3]],
        replacement: [[1, 2, 3]]
      },

      # Rule 3: Triangle split - triangle becomes three edges with center
      %{
        name: "triangle_split",
        pattern: [[1, 2, 3]],
        replacement: [
          [1, :center],
          [2, :center],
          [3, :center]
        ]
      },

      # Rule 4: Edge duplication - single edge becomes parallel edges
      %{
        name: "edge_duplication",
        pattern: [[1, 2]],
        replacement: [[1, 2], [1, 2, :parallel]]
      },

      # Rule 5: Four-cycle formation
      %{
        name: "four_cycle",
        pattern: [[1, 2], [2, 3]],
        replacement: [[1, 2, 2, 3]]
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
        pattern: [[1, 2]],
        replacement: [[1, :new], [:new, 2], [1, 2]]
      },
      %{
        name: "growth_expand",
        pattern: [[1, 2, 3]],
        replacement: [
          [1, 2, 3],
          [1, :new1],
          [2, :new2],
          [3, :new3]
        ]
      }
    ]
  end

  def rule_set(:cellular_automaton) do
    [
      # Conway-like rules adapted for hypergraphs
      %{
        name: "survival",
        pattern: [[1, 2], [1, 3]],
        replacement: [[1, 2, 3]]
      },
      %{
        name: "death",
        pattern: [[1, 2, 3, 4]],
        replacement: [[1, 2], [3, 4]]
      }
    ]
  end

  def rule_set(:spacetime) do
    [
      # Rules that might generate spacetime-like structures
      %{
        name: "time_step",
        pattern: [[:t, :x]],
        replacement: [[:t, :x], [{:t, 1}, :x]]
      },
      %{
        name: "space_connection",
        pattern: [[{:t, 0}, :x1], [{:t, 0}, :x2]],
        replacement: [
          [{:t, 0}, :x1],
          [{:t, 0}, :x2],
          [{:t, 0}, :x1, :x2]
        ]
      }
    ]
  end
end
