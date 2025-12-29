defmodule WolframModel.EvolveIdInjectionTest do
  use ExUnit.Case, async: true
  alias WolframModel
  alias Hypergraph

  test "evolve_steps accepts id_generator in opts and uses it across steps" do
    hg = Hypergraph.new() |> Hypergraph.add_hyperedge([1, 2])

    rule = %{
      pattern: [MapSet.new([:x, :y])],
      replacement: [MapSet.new([:x, :y]), MapSet.new([:new])],
      name: "add_new"
    }

    id_gen = fn -> 777 end

    model = WolframModel.new(hg, [rule])
    final = WolframModel.evolve_steps(model, 2, id_generator: id_gen)

    # After two steps we expect a generated vertex with generation 1 and id 777
    assert Enum.any?(Hypergraph.hyperedges(final.hypergraph), fn he ->
             MapSet.member?(he, {:new, 1, 777})
           end)
  end
end
