defmodule CorrelationLength do
  @moduledoc """
  Calculates correlation length for Wolfram Model using Mutual Information.

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
    distances = Range.to_list(1..max_distance)

    mutual_info_by_distance =
      distances
      |> Enum.map(fn d ->
        {d, calculate_mutual_info_at_distance(hypergraph, d, region_size, samples)}
      end)
      # Filter out zero MI values
      |> Enum.filter(fn {_d, mi} -> mi > 0 end)

    case fit_exponential_decay(mutual_info_by_distance) do
      {:ok, correlation_length} -> {:ok, correlation_length}
      {:error, reason} -> {:error, reason}
    end
  end

  # Calculate Mutual Information between regions at a specific distance.
  defp calculate_mutual_info_at_distance(hypergraph, distance, region_size, samples) do
    vertex_distances = build_distance_matrix(hypergraph)

    region_pairs =
      sample_region_pairs_at_distance(
        hypergraph,
        vertex_distances,
        distance,
        region_size,
        samples
      )

    if length(region_pairs) < 10 do
      # Not enough samples for reliable MI calculation
      0.0
    else
      region_pairs
      |> Enum.map(fn {region1, region2} ->
        calculate_mutual_information(region1, region2, hypergraph)
      end)
      |> Enum.sum()
      |> Kernel./(length(region_pairs))
    end
  end

  # Build distance matrix using shortest path in hypergraph 2-skeleton.
  defp build_distance_matrix(hypergraph) do
    vertices = Hypergraph.vertices(hypergraph)
    edges = hypergraph_to_graph_edges(hypergraph)

    # Floyd-Warshall algorithm for all-pairs shortest paths
    initial_distances = initialize_distances(vertices, edges)
    floyd_warshall(initial_distances, vertices)
  end

  # Convert hypergraph to graph edges (2-skeleton).
  defp hypergraph_to_graph_edges(hypergraph) do
    hypergraph
    |> Hypergraph.hyperedges()
    |> Enum.flat_map(fn hyperedge ->
      # Create pairwise connections within each hyperedge
      vertices_in_edge = hyperedge
      for v1 <- vertices_in_edge, v2 <- vertices_in_edge, v1 != v2, do: {v1, v2}
    end)
    |> Enum.uniq()
  end

  defp initialize_distances(vertices, edges) do
    # Initialize with infinity, 0 for self, 1 for direct edges
    for v1 <- vertices, v2 <- vertices, into: %{} do
      cond do
        v1 == v2 -> {{v1, v2}, 0}
        {v1, v2} in edges -> {{v1, v2}, 1}
        true -> {{v1, v2}, :infinity}
      end
    end
  end

  defp floyd_warshall(distances, vertices) do
    Enum.reduce(vertices, distances, fn k, dist_acc ->
      Enum.reduce(vertices, dist_acc, fn i, dist_acc2 ->
        Enum.reduce(vertices, dist_acc2, fn j, dist_acc3 ->
          d_ik = Map.get(dist_acc3, {i, k}, :infinity)
          d_kj = Map.get(dist_acc3, {k, j}, :infinity)
          d_ij = Map.get(dist_acc3, {i, j}, :infinity)

          new_distance =
            case {d_ik, d_kj} do
              {:infinity, _} -> d_ij
              {_, :infinity} -> d_ij
              {a, b} -> min(d_ij, a + b)
            end

          Map.put(dist_acc3, {i, j}, new_distance)
        end)
      end)
    end)
  end

  # Sample pairs of regions that are approximately at the target distance.
  defp sample_region_pairs_at_distance(
         hypergraph,
         distances,
         target_distance,
         region_size,
         samples
       ) do
    vertices = Hypergraph.vertices(hypergraph)

    1..samples
    |> Enum.map(fn _ ->
      region1 = sample_region(vertices, region_size)
      region2 = find_distant_region(region1, vertices, distances, target_distance, region_size)
      {region1, region2}
    end)
    |> Enum.filter(fn {r1, r2} -> length(r1) > 0 and length(r2) > 0 end)
  end

  defp sample_region(vertices, size) do
    vertices
    |> Enum.shuffle()
    |> Enum.take(min(size, length(vertices)))
  end

  defp find_distant_region(
         reference_region,
         all_vertices,
         distances,
         target_distance,
         region_size
       ) do
    # Find vertices approximately at target distance from reference region
    distant_candidates =
      all_vertices
      |> Enum.filter(fn v ->
        avg_distance = calculate_avg_distance_to_region(v, reference_region, distances)
        # Allow some tolerance
        abs(avg_distance - target_distance) <= 1
      end)

    sample_region(distant_candidates, region_size)
  end

  defp calculate_avg_distance_to_region(vertex, region, distances) do
    region_distances =
      region
      |> Enum.map(fn r_vertex ->
        case Map.get(distances, {vertex, r_vertex}) do
          # Large finite number
          :infinity -> 999
          d -> d
        end
      end)

    if length(region_distances) > 0 do
      Enum.sum(region_distances) / length(region_distances)
    else
      999
    end
  end

  # Calculate Mutual Information between two regions based on structural features.
  defp calculate_mutual_information(region1, region2, hypergraph) do
    # Extract structural features for each region
    features1 = extract_region_features(region1, hypergraph)
    features2 = extract_region_features(region2, hypergraph)

    # Calculate joint and marginal distributions
    joint_dist = calculate_joint_distribution(features1, features2)
    marginal1 = calculate_marginal_distribution(features1)
    marginal2 = calculate_marginal_distribution(features2)

    # Compute Mutual Information: I(X;Y) = ΣΣ p(x,y) log(p(x,y) / (p(x)p(y)))
    joint_dist
    |> Enum.map(fn {{f1, f2}, joint_prob} ->
      marginal_prob1 = Map.get(marginal1, f1, 0)
      marginal_prob2 = Map.get(marginal2, f2, 0)

      if joint_prob > 0 and marginal_prob1 > 0 and marginal_prob2 > 0 do
        joint_prob * :math.log(joint_prob / (marginal_prob1 * marginal_prob2))
      else
        0
      end
    end)
    |> Enum.sum()
  end

  # Extract structural features from a region (e.g., degree distribution, hyperedge participation).
  defp extract_region_features(region, hypergraph) do
    hyperedges = Hypergraph.hyperedges(hypergraph)

    region
    |> Enum.map(fn vertex ->
      # Count hyperedges containing this vertex
      degree =
        hyperedges
        |> Enum.count(fn hyperedge -> vertex in hyperedge end)

      # Bin the degree into categories for discrete distribution
      cond do
        degree == 0 -> :isolated
        degree <= 2 -> :low_degree
        degree <= 5 -> :medium_degree
        true -> :high_degree
      end
    end)
  end

  defp calculate_joint_distribution(features1, features2) do
    # Create all combinations and count occurrences
    total = length(features1) * length(features2)

    combinations = for f1 <- features1, f2 <- features2, do: {f1, f2}

    combinations
    |> Enum.frequencies()
    |> Enum.map(fn {combo, count} -> {combo, count / total} end)
    |> Map.new()
  end

  defp calculate_marginal_distribution(features) do
    total = length(features)

    features
    |> Enum.frequencies()
    |> Enum.map(fn {feature, count} -> {feature, count / total} end)
    |> Map.new()
  end

  # Fit exponential decay to Mutual Information data and extract Correlation Length.
  # I(d) ≈ I(0) * exp(-d/ξ), so ξ is the Correlation Length.
  defp fit_exponential_decay(data_points) when length(data_points) < 3 do
    {:error, :insufficient_data}
  end

  defp fit_exponential_decay(data_points) do
    # Simple linear regression on log scale: log(I) = log(I0) - d/ξ
    {distances, mutual_infos} = Enum.unzip(data_points)

    # Filter out zero or negative MI values
    valid_points =
      Enum.zip(distances, mutual_infos)
      |> Enum.filter(fn {_d, mi} -> mi > 0 end)

    if length(valid_points) < 3 do
      {:error, :insufficient_positive_data}
    else
      {valid_distances, valid_mis} = Enum.unzip(valid_points)
      log_mis = Enum.map(valid_mis, &:math.log/1)

      # Linear regression: log(MI) = a + b*d, where b = -1/ξ
      correlation_length =
        case linear_regression(valid_distances, log_mis) do
          {:ok, {_intercept, slope}} when slope < 0 ->
            {:ok, -1.0 / slope}

          {:ok, {_intercept, slope}} when slope >= 0 ->
            {:error, :no_decay}

          {:error, reason} ->
            {:error, reason}
        end

      correlation_length
    end
  end

  # Simple linear regression: y = a + bx
  # Returns {:ok, {intercept, slope}} or {:error, reason}
  defp linear_regression(x_values, _y_values) when length(x_values) < 2 do
    {:error, :insufficient_points}
  end

  defp linear_regression(x_values, y_values) do
    n = length(x_values)

    sum_x = Enum.sum(x_values)
    sum_y = Enum.sum(y_values)
    sum_xy = Enum.zip(x_values, y_values) |> Enum.map(fn {x, y} -> x * y end) |> Enum.sum()
    sum_x2 = x_values |> Enum.map(fn x -> x * x end) |> Enum.sum()

    denominator = n * sum_x2 - sum_x * sum_x

    if abs(denominator) < 1.0e-10 do
      {:error, :singular_matrix}
    else
      slope = (n * sum_xy - sum_x * sum_y) / denominator
      intercept = (sum_y - slope * sum_x) / n
      {:ok, {intercept, slope}}
    end
  end
end
