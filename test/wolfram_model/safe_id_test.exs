defmodule WolframModel.SafeIdTest do
  use ExUnit.Case, async: true
  alias WolframModel
  alias Hypergraph

  test "substitute :new yields tuple-based id" do
    hg = Hypergraph.new() |> Hypergraph.add_hyperedge([:a, :b])

    rule = %{
      pattern: [[:a, :b]],
      replacement: [[:new]],
      name: "new_vertex"
    }

    model = WolframModel.new(hg, [rule])
    m2 = WolframModel.evolve_step(model)

    # Find the replacement hyperedge with the new vertex
    new_vertex_found =
      Enum.any?(Hypergraph.hyperedges(m2.hypergraph), fn he ->
        Enum.any?(he, fn v ->
          case v do
            {tag, gen, id} when is_atom(tag) and is_integer(gen) and is_integer(id) -> tag == :new
            _ -> false
          end
        end)
      end)

    assert new_vertex_found
  end

  test "injectable id_generator yields deterministic ids" do
    hg = Hypergraph.new() |> Hypergraph.add_hyperedge([:a, :b])

    rule = %{
      pattern: [[:a, :b]],
      replacement: [[:new]],
      name: "new_vertex"
    }

    id_gen = fn -> 123 end

    model = WolframModel.new(hg, [rule], id_generator: id_gen)
    m2 = WolframModel.evolve_step(model)

    assert Enum.any?(Hypergraph.hyperedges(m2.hypergraph), fn he ->
             {:new, 0, 123} in he
           end)
  end
end
