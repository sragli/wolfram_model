defmodule WolframModel.PropertyTest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias Hypergraph
  alias WolframModel
  alias WolframModel.Analytics
  alias WolframModel.Matcher

  # Small hyperedge generator: non-empty set of small integers
  defp hyperedge_gen() do
    StreamData.uniq_list_of(StreamData.integer(1..20), min_length: 2, max_length: 4)
    |> StreamData.map(&MapSet.new/1)
  end

  defp hyperedges_list_gen(),
    do: StreamData.list_of(hyperedge_gen(), min_length: 1, max_length: 6)

  property "Matcher.match for single-pattern is deterministic and correct" do
    check all(hyperedges <- hyperedges_list_gen(), max_runs: 50) do
      # pick a hyperedge that will be matched
      he = Enum.random(hyperedges)
      size = MapSet.size(he)

      # build a pattern with placeholder atoms :a, :b, ...
      placeholders = Enum.take([:a, :b, :c, :d, :e], size)
      pattern = MapSet.new(placeholders)

      # run match
      res1 = Matcher.match(hyperedges, [pattern])
      res2 = Matcher.match(hyperedges, [pattern])

      # If there's a match, verify mapping maps placeholders to elements of the chosen he
      if res1 do
        mapped_values = Map.values(res1.mapping) |> MapSet.new()
        # mapping should correspond to some hyperedge in the input list
        assert Enum.any?(hyperedges, fn h -> MapSet.equal?(mapped_values, h) end)
        assert res1 == res2
      end
    end
  end

  property "Analytics.build_adjacency_map is symmetric and covers vertices" do
    check all(hyperedges <- hyperedges_list_gen(), max_runs: 50) do
      hg =
        Enum.reduce(hyperedges, Hypergraph.new(), fn he, acc ->
          Hypergraph.add_hyperedge(acc, he)
        end)

      adj = Analytics.build_adjacency_map(hg)

      # Each vertex present and neighbor sets only include vertices from hg
      Enum.each(Hypergraph.vertices(hg), fn v ->
        assert Map.has_key?(adj, v)

        Enum.each(MapSet.to_list(Map.get(adj, v)), fn nbr ->
          assert MapSet.member?(Hypergraph.vertices(hg), nbr)
          # symmetry
          assert MapSet.member?(Map.get(adj, nbr, MapSet.new()), v)
        end)
      end)
    end
  end

  property "Evolve step with :new in replacement produces tuple ids and preserves structure" do
    check all(hyperedges <- hyperedges_list_gen(), max_runs: 50) do
      # Build initial hypergraph with at least one hyperedge, pick one
      hg =
        Enum.reduce(hyperedges, Hypergraph.new(), fn he, acc ->
          Hypergraph.add_hyperedge(acc, he)
        end)

      he = Enum.random(hyperedges)

      # pattern placeholders sized like he
      size = MapSet.size(he)
      placeholders = Enum.take([:a, :b, :c, :d, :e], size)
      pattern = MapSet.new(placeholders)

      rule = %{
        pattern: [pattern],
        replacement: [MapSet.new([List.first(placeholders), :new])],
        name: "add_new"
      }

      model = WolframModel.new(hg, [rule])
      new_model = WolframModel.evolve_step(model)

      # If evolution happened, the new hypergraph should contain a tuple vertex starting with :new
      if new_model.generation > 0 do
        found_new =
          Enum.any?(Hypergraph.vertices(new_model.hypergraph), fn v ->
            is_tuple(v) and elem(v, 0) in [:new, :center, :parallel, :new1, :new2, :new3]
          end)

        assert found_new
      end
    end
  end

  property "Matcher.match for two-hyperedge patterns produces consistent mappings" do
    check all(
            a <- StreamData.uniq_list_of(StreamData.integer(1..30), min_length: 1, max_length: 3),
            b <- StreamData.uniq_list_of(StreamData.integer(1..30), min_length: 1, max_length: 3),
            max_runs: 80
          ) do
      he1 = MapSet.new(a)
      he2 = MapSet.new(b)
      shared = MapSet.intersection(he1, he2)

      # If there's no shared vertex, skip this run
      if MapSet.size(shared) == 0 do
        true
      else
        # Construct a small hyperedge list containing he1 and he2 and some extras
        extras =
          1..Enum.random(0..2)
          |> Enum.map(fn _ -> MapSet.new([Enum.random(1..30), Enum.random(1..30)]) end)

        hyperedges = [he1, he2 | extras]

        atoms = [:a, :b, :c, :d, :e, :f]
        s_count = MapSet.size(shared)
        shared_placeholders = Enum.take(atoms, s_count)

        p1_size = MapSet.size(he1)
        p2_size = MapSet.size(he2)

        p1_placeholders =
          shared_placeholders ++ Enum.take(Enum.drop(atoms, s_count), p1_size - s_count)

        p2_placeholders =
          shared_placeholders ++
            Enum.take(Enum.drop(atoms, s_count + (p1_size - s_count)), p2_size - s_count)

        p1 = MapSet.new(p1_placeholders)
        p2 = MapSet.new(p2_placeholders)

        res = Matcher.match_all(hyperedges, [p1, p2])

        Enum.each(res, fn %{mapping: mapping, matched_hyperedges: [m1, m2]} ->
          # matched hyperedges should be taken from the supplied hyperedges list
          assert Enum.any?(hyperedges, fn h -> MapSet.equal?(h, m1) end)
          assert Enum.any?(hyperedges, fn h -> MapSet.equal?(h, m2) end)

          # mapping values for p1 placeholders correspond exactly to m1
          keys1 = MapSet.to_list(p1)
          mapped1 = keys1 |> Enum.map(&Map.get(mapping, &1)) |> MapSet.new()
          assert MapSet.equal?(mapped1, m1)

          # mapping values for p2 placeholders correspond exactly to m2
          keys2 = MapSet.to_list(p2)
          mapped2 = keys2 |> Enum.map(&Map.get(mapping, &1)) |> MapSet.new()
          assert MapSet.equal?(mapped2, m2)

          # shared placeholders map to the intersection
          shared_keys = shared_placeholders
          shared_mapped = shared_keys |> Enum.map(&Map.get(mapping, &1)) |> MapSet.new()
          assert MapSet.equal?(shared_mapped, MapSet.intersection(m1, m2))
        end)
      end
    end
  end

  property "Matcher.match_all is insensitive to input hyperedge ordering" do
    check all(hyperedges <- hyperedges_list_gen(), max_runs: 60) do
      he = Enum.random(hyperedges)
      size = MapSet.size(he)
      placeholders = Enum.take([:a, :b, :c, :d, :e], size)
      pattern = MapSet.new(placeholders)

      res1 = Matcher.match_all(hyperedges, [pattern])
      res2 = Matcher.match_all(Enum.shuffle(hyperedges), [pattern])

      normalize = fn res ->
        res
        |> Enum.map(fn %{mapping: m, matched_hyperedges: mh} ->
          mh_norm = mh |> Enum.map(&Enum.sort(MapSet.to_list(&1))) |> Enum.sort()
          mapping_norm = m |> Map.to_list() |> Enum.sort()
          {mh_norm, mapping_norm}
        end)
        |> MapSet.new()
      end

      assert normalize.(res1) == normalize.(res2)
    end
  end
end
