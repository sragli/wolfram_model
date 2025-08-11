defmodule WolframModel.Example do
  @moduledoc """
  Factory for sample models.
  """
  alias WolframModel.RuleSet

  @doc """
  Creates a simple initial universe for experimentation.
  """
  @spec simple_universe() :: WolframModel.t()
  def simple_universe do
    initial_hg =
      Hypergraph.new()
      |> Hypergraph.add_hyperedge([1, 2])
      |> Hypergraph.add_hyperedge([2, 3])

    WolframModel.new(initial_hg, RuleSet.basic_rules())
  end

  @doc """
  Creates a more complex initial universe.
  """
  @spec complex_universe() :: WolframModel.t()
  def complex_universe do
    initial_hg =
      Hypergraph.new()
      |> Hypergraph.add_hyperedge([1, 2, 3])
      |> Hypergraph.add_hyperedge([3, 4])
      |> Hypergraph.add_hyperedge([4, 5])
      |> Hypergraph.add_hyperedge([1, 5])

    WolframModel.new(initial_hg, RuleSet.basic_rules())
  end

end
