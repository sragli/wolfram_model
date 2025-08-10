defmodule CorrelationLength do
  @moduledoc """
  Optimized correlation length calculator for Wolfram Models using mutual information approach.

  The correlation length measures how far structural information propagates
  in the hypergraph by calculating mutual information between distant regions.
  """
  alias Hypergraph

  @doc """
  Calculate correlation length for a hypergraph using mutual information decay.

  ## Parameters
  - hypergraph: Map with vertices and hyperedges
  - max_distance: Maximum distance to consider
  - region_size: Size of regions to compare
  - samples: Number of random region pairs to sample

  ## Returns
  - {:ok, correlation_length} | {:error, reason}
  """
  def compute(hypergraph, max_distance \\ 10, region_size \\ 5, samples \\ 100) do
    # Pre-compute expensive operations once
    vertex_set = MapSet.new(Hypergraph.vertices(hypergraph))
    adjacency_map = build_adjacency_map(hypergraph)

    distances = 1..max_distance |> Enum.to_list()

    mutual_info_by_distance =
      distances
      |> Task.async_stream(fn d ->
        {d, calculate_mutual_info_at_distance(vertex_set, adjacency_map, d, region_size, samples)}
      end, ordered: false, timeout: :infinity)
      |> Enum.map(fn {:ok, result} -> result end)
      |> Enum.filter(fn {_d, mi} -> mi > 0 end)  # Filter out zero MI values

    case fit_exponential_decay(mutual_info_by_distance) do
      {:ok, correlation_length} -> {:ok, correlation_length}
      {:error, reason} -> {:error, reason}
    end
  end

  @doc """
  Build adjacency map for efficient neighbor lookup (replaces distance matrix).
  """
  defp build_adjacency_map(hypergraph) do
    hyperedges = Hypergraph.hyperedges(hypergraph)

    # Build adjacency list more efficiently
    hyperedges
    |> Enum.reduce(%{}, fn hyperedge, acc ->
      # Add all pairwise connections within hyperedge
      hyperedge
      |> Enum.reduce(acc, fn vertex, acc2 ->
        neighbors = MapSet.delete(hyperedge, vertex)  # Remove self
        Map.update(acc2, vertex, MapSet.new(neighbors), fn existing ->
          MapSet.union(existing, MapSet.new(neighbors))
        end)
      end)
    end)
  end

  @doc """
  Calculate mutual information between regions at a specific distance using BFS.
  """
  defp calculate_mutual_info_at_distance(vertex_set, adjacency_map, distance, region_size, samples) do
    # Use streaming to avoid building large intermediate collections
    region_pairs = Stream.repeatedly(fn ->
      sample_region_pair_at_distance(vertex_set, adjacency_map, distance, region_size)
    end)
    |> Stream.take(samples)
    |> Stream.filter(fn {r1, r2} -> length(r1) > 0 and length(r2) > 0 end)
    |> Enum.take(samples)

    if length(region_pairs) < 10 do
      0.0  # Not enough samples for reliable MI calculation
    else
      # Use parallel processing for MI calculation
      region_pairs
      |> Task.async_stream(fn {region1, region2} ->
        calculate_mutual_information_optimized(region1, region2, adjacency_map)
      end, ordered: false)
      |> Enum.map(fn {:ok, mi} -> mi end)
      |> Enum.sum()
      |> Kernel./(length(region_pairs))
    end
  end

  @doc """
  Sample a region pair at target distance using BFS (much faster than Floyd-Warshall).
  """
  defp sample_region_pair_at_distance(vertex_set, adjacency_map, target_distance, region_size) do
    vertices = MapSet.to_list(vertex_set)
    region1 = sample_region(vertices, region_size)

    # Use BFS to find vertices at approximately target distance
    distant_vertices = find_vertices_at_distance_bfs(region1, adjacency_map, target_distance, vertex_set)
    region2 = sample_region(distant_vertices, region_size)

    {region1, region2}
  end

  @doc """
  BFS to find vertices at specific distance (replaces expensive all-pairs shortest path).
  """
  defp find_vertices_at_distance_bfs(source_region, adjacency_map, target_distance, all_vertices) do
    # Initialize BFS from all vertices in source region
    initial_queue = Enum.map(source_region, fn v -> {v, 0} end)
    initial_visited = MapSet.new(source_region)

    bfs_distance_search(initial_queue, initial_visited, adjacency_map, target_distance, all_vertices, [])
  end

  defp bfs_distance_search([], _visited, _adjacency_map, _target_distance, _all_vertices, result) do
    result
  end

  defp bfs_distance_search([{current, distance} | rest], visited, adjacency_map, target_distance, all_vertices, result) do
    cond do
      distance == target_distance ->
        # Found vertex at target distance
        bfs_distance_search(rest, visited, adjacency_map, target_distance, all_vertices, [current | result])

      distance < target_distance ->
        # Continue BFS
        neighbors = Map.get(adjacency_map, current, MapSet.new())
        new_neighbors = MapSet.difference(neighbors, visited)

        new_queue = Enum.map(MapSet.to_list(new_neighbors), fn n -> {n, distance + 1} end)
        new_visited = MapSet.union(visited, new_neighbors)

        bfs_distance_search(rest ++ new_queue, new_visited, adjacency_map, target_distance, all_vertices, result)

      true ->
        # Skip vertices beyond target distance
        bfs_distance_search(rest, visited, adjacency_map, target_distance, all_vertices, result)
    end
  end

  defp sample_region(vertices, size) when length(vertices) <= size, do: vertices
  defp sample_region(vertices, size) do
    vertices
    |> Enum.shuffle()
    |> Enum.take(size)
  end

  @doc """
  Optimized mutual information calculation using pre-computed features and efficient distributions.
  """
  defp calculate_mutual_information_optimized(region1, region2, adjacency_map) do
    # Pre-compute features once
    features1 = extract_region_features_optimized(region1, adjacency_map)
    features2 = extract_region_features_optimized(region2, adjacency_map)

    # Use more efficient distribution calculation
    calculate_mutual_info_from_features(features1, features2)
  end

  @doc """
  Optimized feature extraction using adjacency map instead of scanning all hyperedges.
  """
  defp extract_region_features_optimized(region, adjacency_map) do
    region
    |> Enum.map(fn vertex ->
      # Use adjacency map for O(1) degree lookup
      degree = case Map.get(adjacency_map, vertex) do
        nil -> 0
        neighbors -> MapSet.size(neighbors)
      end

      # More granular binning for better information content
      categorize_degree(degree)
    end)
  end

  defp categorize_degree(degree) do
    cond do
      degree == 0 -> :isolated
      degree == 1 -> :leaf
      degree <= 3 -> :low_degree
      degree <= 6 -> :medium_degree
      degree <= 10 -> :high_degree
      true -> :very_high_degree
    end
  end

  @doc """
  More efficient mutual information calculation using frequency maps.
  """
  defp calculate_mutual_info_from_features(features1, features2) do
    # Pre-compute frequencies
    freq1 = Enum.frequencies(features1)
    freq2 = Enum.frequencies(features2)

    n1 = length(features1)
    n2 = length(features2)
    total = n1 * n2

    # Build joint frequency map more efficiently
    joint_freq =
      for f1 <- Map.keys(freq1), f2 <- Map.keys(freq2), into: %{} do
        joint_count = freq1[f1] * freq2[f2]
        {{f1, f2}, joint_count}
      end

    # Calculate MI using pre-computed frequencies
    joint_freq
    |> Enum.reduce(0.0, fn {{f1, f2}, joint_count}, acc ->
      if joint_count > 0 do
        joint_prob = joint_count / total
        marginal_prob1 = freq1[f1] / n1
        marginal_prob2 = freq2[f2] / n2

        mi_contribution = joint_prob * :math.log(joint_prob / (marginal_prob1 * marginal_prob2))
        acc + mi_contribution
      else
        acc
      end
    end)
  end

  @doc """
  Optimized exponential decay fitting with better numerical stability.
  """
  defp fit_exponential_decay(data_points) do
    if length(data_points) < 3 do
      {:error, :insufficient_data}
    else
      # Filter and prepare data
      valid_points =
        data_points
        |> Enum.filter(fn {_d, mi} -> mi > 1.0e-10 end)  # More robust threshold
        |> Enum.sort_by(fn {d, _mi} -> d end)  # Sort by distance

      if length(valid_points) < 3 do
        {:error, :insufficient_positive_data}
      else
        case linear_regression_optimized(valid_points) do
          {:ok, correlation_length} -> {:ok, correlation_length}
          {:error, reason} -> {:error, reason}
        end
      end
    end
  end

  @doc """
  Optimized linear regression with better numerical stability.
  """
  defp linear_regression_optimized(data_points) do
    {distances, mutual_infos} = Enum.unzip(data_points)

    # Use log transformation with numerical stability
    log_mis =
      mutual_infos
      |> Enum.map(fn mi ->
        if mi > 0, do: :math.log(mi), else: :math.log(1.0e-10)
      end)

    n = length(distances)

    if n < 2 do
      {:error, :insufficient_points}
    else
      # Use more numerically stable formulation
      mean_x = Enum.sum(distances) / n
      mean_y = Enum.sum(log_mis) / n

      numerator =
        Enum.zip(distances, log_mis)
        |> Enum.map(fn {x, y} -> (x - mean_x) * (y - mean_y) end)
        |> Enum.sum()

      denominator =
        distances
        |> Enum.map(fn x -> (x - mean_x) * (x - mean_x) end)
        |> Enum.sum()

      if abs(denominator) < 1.0e-10 do
        {:error, :singular_matrix}
      else
        slope = numerator / denominator

        if slope >= -1.0e-10 do  # No significant decay
          {:error, :no_decay}
        else
          correlation_length = -1.0 / slope
          {:ok, correlation_length}
        end
      end
    end
  end
end
