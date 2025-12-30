defmodule WolframModel.Analytics do
  @moduledoc """
  Emergence and graph analytics helpers for the Wolfram Model.

  This module provides functions to build adjacency maps, compute
  clustering coefficients, estimate diameter, and other small analytics
  previously living directly on `WolframModel`.
  """

  alias Hypergraph

  @doc """
  Analyze emergent properties of the given `model`.

  Returns a map containing basic hypergraph stats plus derived metrics:
  - `:clustering_coefficient` (float)
  - `:estimated_diameter` (integer)
  - `:growth_rate` (float)
  - `:complexity_measure` (number)
  - `:evolution_generation` (integer)

  This function delegates to the helper functions provided by this module and
  returns a stable map suitable for reporting or tests.
  """
  @spec analyze_emergence(WolframModel.t()) :: map()
  def analyze_emergence(model) do
    hg = model.hypergraph
    stats = Hypergraph.stats(hg)

    adjacency_map = build_adjacency_map(hg)

    Map.merge(stats, %{
      clustering_coefficient: calculate_clustering_coefficient(adjacency_map),
      estimated_diameter: estimate_diameter(adjacency_map),
      growth_rate: calculate_growth_rate(model),
      complexity_measure: calculate_complexity(hg),
      evolution_generation: model.generation
    })
  end

  @doc """
  Build an adjacency map from `hg`.

  The returned map has the shape `%{vertex => MapSet.t(neighbors)}`. Isolated
  vertices are included with an empty `MapSet`.
  """
  @spec build_adjacency_map(Hypergraph.t()) :: map()
  def build_adjacency_map(hg) do
    adj =
      hg
      |> Hypergraph.hyperedges()
      |> Enum.reduce(%{}, fn he, acc ->
        vertices = MapSet.to_list(he)

        Enum.reduce(vertices, acc, fn v, acc2 ->
          neighbors = MapSet.delete(he, v)
          Map.update(acc2, v, neighbors, &MapSet.union(&1, neighbors))
        end)
      end)

    Enum.reduce(Hypergraph.vertices(hg), adj, fn v, acc ->
      Map.put_new(acc, v, MapSet.new())
    end)
  end

  @doc """
  Compute the average local clustering coefficient for the (projected) graph
  represented by `adjacency_map`.

  Uses the standard definition: average over vertices of (existing neighbor
  connections) / (possible neighbor connections).
  """
  @spec calculate_clustering_coefficient(map()) :: float()
  def calculate_clustering_coefficient(adjacency_map) do
    vertices = Map.keys(adjacency_map)

    if length(vertices) < 3 do
      0.0
    else
      local_coeffs =
        Enum.map(vertices, fn v ->
          neighbors = Map.get(adjacency_map, v, MapSet.new()) |> MapSet.to_list()
          n = length(neighbors)

          if n < 2 do
            0.0
          else
            pairs = for i <- neighbors, j <- neighbors, i < j, do: {i, j}

            existing =
              Enum.count(pairs, fn {i, j} ->
                MapSet.member?(Map.get(adjacency_map, i, MapSet.new()), j)
              end)

            possible = n * (n - 1) / 2
            existing / possible
          end
        end)

      Enum.sum(local_coeffs) / length(local_coeffs)
    end
  end

  @doc """
  Estimate the graph diameter (longest shortest path) from `adjacency_map`.

  For disconnected graphs, the maximum diameter among components is returned.
  For a graph with a single isolated vertex the function returns `1` to remain
  compatible with prior behavior in the project.
  """
  @spec estimate_diameter(map()) :: non_neg_integer()
  def estimate_diameter(adjacency_map) do
    vertices = Map.keys(adjacency_map)

    if vertices == [] do
      1
    else
      distances =
        Enum.map(vertices, fn v ->
          bfs_distances(adjacency_map, v)
          |> Map.values()
          |> Enum.filter(&is_integer/1)
          |> Enum.max(fn -> 0 end)
        end)

      max_distance = Enum.max(distances)
      max(1, max_distance)
    end
  end

  defp bfs_distances(adj_map, source) do
    bfs_distances(adj_map, [{source, 0}], Map.put(%{}, source, 0))
  end

  defp bfs_distances(_adj_map, [], visited), do: visited

  defp bfs_distances(adj_map, [{node, dist} | rest], visited) do
    neighbors = Map.get(adj_map, node, MapSet.new())

    {new_items, new_visited} =
      Enum.reduce(neighbors, {[], visited}, fn nbr, {items, vis} ->
        if Map.has_key?(vis, nbr) do
          {items, vis}
        else
          {[{nbr, dist + 1} | items], Map.put(vis, nbr, dist + 1)}
        end
      end)

    bfs_distances(adj_map, rest ++ Enum.reverse(new_items), new_visited)
  end

  @doc """
  Simple growth rate between the two most recent hypergraph snapshots in the
  model's `evolution_history` (recent - previous) / previous. Returns `0.0` if
  there are fewer than two snapshots or the previous count is zero.
  """
  @spec calculate_growth_rate(WolframModel.t()) :: float()
  def calculate_growth_rate(%WolframModel{evolution_history: history}) when length(history) < 2,
    do: 0.0

  def calculate_growth_rate(%WolframModel{evolution_history: history}) do
    recent = List.first(history) |> Hypergraph.vertex_count()
    previous = Enum.at(history, 1) |> Hypergraph.vertex_count()
    if previous == 0, do: 0.0, else: (recent - previous) / previous
  end

  @doc """
  Calculate a lightweight complexity measure for a hypergraph. Currently
  delegated to `Hypergraph.CorrelationLength.compute/1`. If that function is
  not available or returns an error, this returns `0`.
  """
  @spec calculate_complexity(Hypergraph.t()) :: number()
  def calculate_complexity(hg) do
    case Hypergraph.CorrelationLength.compute(hg) do
      {:ok, correlation_length} -> correlation_length
      _ -> 0
    end
  end
end
