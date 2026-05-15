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
  Analyzes the causal structure of evolution events using event indices for efficiency.
  Returns counts and density.
  """
  @spec analyze_causality(WolframModel.t()) :: map()
  def analyze_causality(model) do
    events = model.causal_network

    # Build unique parent->child edges using event indices to avoid O(n^2) scans
    edges =
      events
      |> Enum.flat_map(fn e ->
        e.parent_ids
        |> Enum.map(fn pid -> {pid, e.id} end)
      end)
      |> MapSet.new()

    %{
      event_count: length(events),
      causal_edges: MapSet.size(edges),
      generations: model.generation,
      causal_density: causal_density(edges, events)
    }
  end

  defp causal_density(edges, events) when length(events) > 1 do
    MapSet.size(edges) / (length(events) * (length(events) - 1))
  end

  defp causal_density(_edges, _events), do: 0.0

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
        Enum.reduce(he, acc, fn v, acc2 ->
          neighbors = he |> Enum.reject(&(&1 == v)) |> MapSet.new()
          Map.update(acc2, v, neighbors, &MapSet.union(&1, neighbors))
        end)
      end)

    adj
  end

  @doc """
  Compute the average local clustering coefficient for the (projected) graph
  represented by `adjacency_map`.

  Uses the standard definition: average over vertices of (existing neighbor
  connections) / (possible neighbor connections).
  """
  @spec calculate_clustering_coefficient(map()) :: float()
  def calculate_clustering_coefficient(adjacency_map) when map_size(adjacency_map) < 3, do: 0.0

  def calculate_clustering_coefficient(adjacency_map) do
    local_coeffs =
      adjacency_map
      |> Map.keys()
      |> Enum.map(fn v ->
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

  @doc """
  Estimate the graph diameter (longest shortest path) from `adjacency_map`.

  For disconnected graphs, the maximum diameter among components is returned.
  For a graph with a single isolated vertex the function returns `1` to remain
  compatible with prior behavior in the project.
  """
  @spec estimate_diameter(map()) :: non_neg_integer()
  def estimate_diameter(adjacency_map) when length(adjacency_map) == 0, do: 1

  def estimate_diameter(adjacency_map) do
    adjacency_map
    |> Map.keys()
    |> Enum.map(fn v ->
      bfs_distances(adjacency_map, v)
      |> Map.values()
      |> Enum.filter(&is_integer/1)
      |> Enum.max(fn -> 0 end)
    end)
    |> Enum.max()
    |> max(1)
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
    recent =
      List.first(history)
      |> Hypergraph.hyperedges()
      |> Enum.flat_map(& &1)
      |> MapSet.new()
      |> MapSet.size()

    previous =
      Enum.at(history, 1)
      |> Hypergraph.hyperedges()
      |> Enum.flat_map(& &1)
      |> MapSet.new()
      |> MapSet.size()

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

  @doc """
  Estimates the effective spatial dimension of the hypergraph using geodesic
  ball growth. Counts vertices within BFS distance r from several seed vertices
  and fits V(r) ~ r^d via log-log linear regression.

  Returns `1.0` for degenerate (fewer than 4 vertices) or disconnected graphs
  where a meaningful estimate cannot be produced.
  """
  @spec estimate_dimension(Hypergraph.t()) :: float()
  def estimate_dimension(hg) do
    vertices = hg |> Hypergraph.hyperedges() |> Enum.flat_map(& &1) |> Enum.uniq()
    n = length(vertices)
    if n < 4, do: 1.0, else: do_estimate_dimension(hg, vertices)
  end

  defp do_estimate_dimension(hg, vertices) do
    # Use more seeds for a better estimate (up to 10).
    seeds = Enum.take(vertices, min(10, length(vertices)))

    estimates =
      seeds
      |> Enum.map(&hypergraph_bfs_distances(hg, &1))
      |> Enum.map(&estimate_dim_from_distances/1)
      |> Enum.reject(&is_nil/1)

    if estimates == [], do: 1.0, else: Enum.sum(estimates) / length(estimates)
  end

  # Geodesic BFS on the *hypergraph* (not the projected binary graph).
  # At each step we move through entire hyperedges: from a vertex we reach all
  # other vertices in any hyperedge that contains it, in one hop.
  defp hypergraph_bfs_distances(hg, source) do
    hyperedges = Hypergraph.hyperedges(hg)
    do_hg_bfs(hyperedges, [{source, 0}], %{source => 0})
  end

  defp do_hg_bfs(_hyperedges, [], visited), do: visited

  defp do_hg_bfs(hyperedges, [{vertex, dist} | queue], visited) do
    new_neighbors =
      hyperedges
      |> Enum.filter(fn he -> vertex in he end)
      |> Enum.flat_map(& &1)
      |> Enum.reject(&Map.has_key?(visited, &1))
      |> Enum.uniq()

    new_visited =
      Enum.reduce(new_neighbors, visited, fn v, acc -> Map.put(acc, v, dist + 1) end)

    new_queue = queue ++ Enum.map(new_neighbors, &{&1, dist + 1})
    do_hg_bfs(hyperedges, new_queue, new_visited)
  end

  defp estimate_dim_from_distances(distances) do
    max_r = distances |> Map.values() |> Enum.max(fn -> 0 end)
    if max_r < 2, do: nil, else: log_log_slope(distances, max_r)
  end

  defp log_log_slope(distances, max_r) do
    points =
      2..max_r
      |> Enum.map(fn r ->
        count = Enum.count(distances, fn {_, d} -> d <= r end)
        if count > 1, do: {:math.log(r), :math.log(count)}, else: nil
      end)
      |> Enum.reject(&is_nil/1)

    if length(points) < 2, do: nil, else: linear_regression_slope(points)
  end

  defp linear_regression_slope(points) do
    {xs, ys} = Enum.unzip(points)
    n = length(xs)
    mean_x = Enum.sum(xs) / n
    mean_y = Enum.sum(ys) / n

    num =
      Enum.zip(xs, ys)
      |> Enum.reduce(0.0, fn {x, y}, acc -> acc + (x - mean_x) * (y - mean_y) end)

    den =
      Enum.reduce(xs, 0.0, fn x, acc -> acc + (x - mean_x) * (x - mean_x) end)

    if den == 0.0, do: nil, else: num / den
  end

  @doc """
  Detects conserved quantities by scanning the full evolution history.

  Checks the following candidates across all recorded hypergraph snapshots:
  - `:vertex_count` — total distinct vertex count is constant
  - `:edge_count` — total hyperedge count is constant
  - `:vertex_count_parity` — vertex count parity (mod 2) is constant
  - `:edge_count_parity` — edge count parity (mod 2) is constant
  - `:total_degree` — sum of all hyperedge sizes (total degree) is constant
  - `:total_degree_parity` — total degree parity is constant

  Returns:
  ```
  %{
    conserved: [:vertex_count, ...],
    vertex_count_history: [n, ...],
    edge_count_history: [n, ...],
    total_degree_history: [n, ...]
  }
  ```

  Requires at least 2 snapshots; returns `%{conserved: []}` otherwise.
  """
  @spec detect_conserved_quantities(WolframModel.t()) :: map()
  def detect_conserved_quantities(%WolframModel{evolution_history: history})
      when length(history) < 2 do
    %{conserved: [], vertex_count_history: [], edge_count_history: [], total_degree_history: []}
  end

  def detect_conserved_quantities(model) do
    snapshots = Enum.reverse(model.evolution_history)

    vertex_counts =
      Enum.map(snapshots, fn hg ->
        hg |> Hypergraph.hyperedges() |> Enum.flat_map(& &1) |> Enum.uniq() |> length()
      end)

    edge_counts =
      Enum.map(snapshots, fn hg -> hg |> Hypergraph.hyperedges() |> length() end)

    total_degrees =
      Enum.map(snapshots, fn hg ->
        hg |> Hypergraph.hyperedges() |> Enum.map(&length/1) |> Enum.sum()
      end)

    conserved =
      []
      |> maybe_conserved(:vertex_count, vertex_counts)
      |> maybe_conserved(:edge_count, edge_counts)
      |> maybe_conserved(:vertex_count_parity, Enum.map(vertex_counts, &rem(&1, 2)))
      |> maybe_conserved(:edge_count_parity, Enum.map(edge_counts, &rem(&1, 2)))
      |> maybe_conserved(:total_degree, total_degrees)
      |> maybe_conserved(:total_degree_parity, Enum.map(total_degrees, &rem(&1, 2)))

    %{
      conserved: conserved,
      vertex_count_history: vertex_counts,
      edge_count_history: edge_counts,
      total_degree_history: total_degrees
    }
  end

  defp maybe_conserved(acc, label, values) do
    if Enum.uniq(values) |> length() == 1, do: [label | acc], else: acc
  end
end
