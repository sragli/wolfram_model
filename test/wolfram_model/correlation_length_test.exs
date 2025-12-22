defmodule WolframModel.CorrelationLengthTest do
  use ExUnit.Case, async: true
  alias WolframModel.CorrelationLength
  alias Hypergraph

  test "compute returns error tuple for tiny hypergraph" do
    hg = Hypergraph.new() |> Hypergraph.add_hyperedge([1, 2])
    assert match?({:error, _}, CorrelationLength.compute(hg, 2, 1, 5))
  end
end
