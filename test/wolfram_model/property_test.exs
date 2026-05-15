defmodule WolframModel.PropertyTest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias Hypergraph
  alias WolframModel
  alias WolframModel.Analytics
  alias WolframModel.Matcher

  # Small hyperedge generator: non-empty list of small integers (ordered)
  defp hyperedge_gen() do
    StreamData.uniq_list_of(StreamData.integer(1..20), min_length: 2, max_length: 4)
  end

  defp hyperedges_list_gen(),
    do: StreamData.list_of(hyperedge_gen(), min_length: 1, max_length: 6)

  test "Matcher.match for single-pattern is correct" do
    check all(hyperedges <- hyperedges_list_gen(), max_runs: 50) do
      # pick a hyperedge that will be matched
      he = Enum.random(hyperedges)
      size = length(he)

      # build a pattern with placeholder atoms :a, :b, ...
      pattern = Enum.take([:a, :b, :c, :d, :e], size)

      # run match
      res = Matcher.match(hyperedges, [pattern])

      # If there's a match, verify mapping maps placeholders to some hyperedge in the input list
      Enum.each(res, fn %{mapping: mapping} ->
        mapped_values = Map.values(mapping) |> MapSet.new()
        assert Enum.any?(hyperedges, fn h -> MapSet.equal?(mapped_values, MapSet.new(h)) end)
      end)
    end
  end

  test "Analytics.build_adjacency_map is symmetric and covers vertices" do
    check all(hyperedges <- hyperedges_list_gen(), max_runs: 50) do
      hg =
        Enum.reduce(hyperedges, Hypergraph.new(), fn he, acc ->
          Hypergraph.add_hyperedge(acc, he)
        end)

      adj = Analytics.build_adjacency_map(hg)

      # Each vertex present and neighbor sets only include vertices from hg
      vertices = hg |> Hypergraph.hyperedges() |> Enum.flat_map(& &1) |> MapSet.new()

      Enum.each(vertices, fn v ->
        assert Map.has_key?(adj, v)

        Enum.each(MapSet.to_list(Map.get(adj, v)), fn nbr ->
          assert MapSet.member?(vertices, nbr)
          # symmetry
          assert MapSet.member?(Map.get(adj, nbr, MapSet.new()), v)
        end)
      end)
    end
  end

  test "Evolve step with :new in replacement produces tuple ids and preserves structure" do
    check all(hyperedges <- hyperedges_list_gen(), max_runs: 50) do
      # Build initial hypergraph with at least one hyperedge, pick one
      hg =
        Enum.reduce(hyperedges, Hypergraph.new(), fn he, acc ->
          Hypergraph.add_hyperedge(acc, he)
        end)

      he = Enum.random(hyperedges)

      # pattern placeholders sized like he
      size = length(he)
      placeholders = Enum.take([:a, :b, :c, :d, :e], size)
      pattern = placeholders

      rule = %{
        pattern: [pattern],
        replacement: [[List.first(placeholders), :new]],
        name: "add_new"
      }

      model = WolframModel.new(hg, [rule])
      new_model = WolframModel.evolve_step(model)

      # If evolution happened, the new hypergraph should contain a tuple vertex starting with :new
      if new_model.generation > 0 do
        found_new =
          Enum.any?(
            new_model.hypergraph |> Hypergraph.hyperedges() |> Enum.flat_map(& &1),
            fn v ->
              is_tuple(v) and elem(v, 0) in [:new, :center, :parallel, :new1, :new2, :new3]
            end
          )

        assert found_new
      end
    end
  end

  test "Matcher.match for two-hyperedge patterns produces consistent mappings" do
    check all(
            he1 <- hyperedge_gen(),
            he2 <- hyperedge_gen(),
            max_runs: 80
          ) do
      # Skip if hyperedges don't share a vertex (matcher requires connectivity)
      shared_vertices = he1 -- he1 -- he2

      if shared_vertices == [] do
        true
      else
        atoms = [:a, :b, :c, :d, :e, :f]
        p1 = Enum.take(atoms, length(he1))
        p2 = Enum.take(Enum.drop(atoms, length(he1)), length(he2))
        # Make p2 share the first variable with p1 via a shared placeholder
        p2 = [List.first(p1) | Enum.drop(p2, 1)]

        # he2 must start with the shared vertex for the mapping to be consistent
        shared_v = List.first(shared_vertices)
        he2_adj = [shared_v | Enum.reject(he2, &(&1 == shared_v))]

        hyperedges = [he1, he2_adj]

        res = Matcher.match(hyperedges, [p1, p2])

        Enum.each(res, fn %{mapping: mapping, matched_hyperedges: [m1, m2]} ->
          # matched hyperedges should be from the supplied list
          assert m1 in hyperedges
          assert m2 in hyperedges

          # positional mapping: p1[i] -> m1[i]
          Enum.each(Enum.zip(p1, m1), fn {k, v} ->
            assert Map.get(mapping, k) == v
          end)

          # positional mapping: p2[i] -> m2[i]
          Enum.each(Enum.zip(p2, m2), fn {k, v} ->
            assert Map.get(mapping, k) == v
          end)
        end)
      end
    end
  end

  test "Matcher.match_all is insensitive to input hyperedge ordering" do
    check all(hyperedges <- hyperedges_list_gen(), max_runs: 60) do
      he = Enum.random(hyperedges)
      size = length(he)
      pattern = Enum.take([:a, :b, :c, :d, :e], size)

      res1 = Matcher.match(hyperedges, [pattern])
      res2 = Matcher.match(Enum.shuffle(hyperedges), [pattern])

      normalize = fn res ->
        res
        |> Enum.map(fn %{mapping: m, matched_hyperedges: mh} ->
          mh_norm = mh |> Enum.map(&Enum.sort/1) |> Enum.sort()
          mapping_norm = m |> Map.to_list() |> Enum.sort()
          {mh_norm, mapping_norm}
        end)
        |> MapSet.new()
      end

      assert normalize.(res1) == normalize.(res2)
    end
  end
end
