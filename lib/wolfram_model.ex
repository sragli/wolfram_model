defmodule WolframModel do
  @moduledoc """
  A simplified implementation of the Wolfram Model using hypergraphs.
  """
  alias Hypergraph
  alias Hypergraph.CorrelationLength

  defstruct hypergraph: %Hypergraph{},
            generation: 0,
            causal_network: [],
            evolution_history: [],
            rules: [],
            id_generator: &WolframModel.default_id_gen/0

  @doc false
  @spec default_id_gen() :: integer()
  def default_id_gen do
    System.unique_integer([:positive])
  end

  @type rule :: %{
          pattern: [MapSet.t()],
          replacement: [MapSet.t()],
          name: String.t()
        }

  @type evolution_event :: %{
          generation: non_neg_integer(),
          rule: rule(),
          matched_hyperedges: [MapSet.t()],
          position: any()
        }

  @type t :: %__MODULE__{
          hypergraph: Hypergraph.t(),
          generation: non_neg_integer(),
          causal_network: [evolution_event()],
          evolution_history: [Hypergraph.t()],
          rules: [rule()],
          id_generator: (-> integer())
        }

  @doc """
  Creates a new Wolfram Model universe with initial hypergraph and rules.
  """
  @spec new(Hypergraph.t(), [rule()], keyword()) :: t()
  def new(initial_hypergraph, rules, opts \\ []) do
    id_gen = Keyword.get(opts, :id_generator, &WolframModel.default_id_gen/0)

    %__MODULE__{
      hypergraph: initial_hypergraph,
      rules: rules,
      evolution_history: [initial_hypergraph],
      id_generator: id_gen
    }
  end

  @doc """
  Evolves the universe by one step, applying the first applicable rule.

  Accepts `opts` where you can pass `:id_generator` to override the model's
  id generator for deterministic behavior during tests or specific runs.
  """
  @spec evolve_step(t(), keyword()) :: t()
  def evolve_step(model, opts \\ []) do
    id_gen = Keyword.get(opts, :id_generator)
    model = if id_gen, do: %{model | id_generator: id_gen}, else: model

    case find_first_match(model) do
      nil ->
        # No rules applicable
        model

      {rule, match_data} ->
        apply_rule(model, rule, match_data)
    end
  end

  @doc """
  Evolves the universe for a specified number of steps.
  Accepts `opts` forwarded to `evolve_step/2` (e.g., `:id_generator`).
  """
  @spec evolve_steps(t(), non_neg_integer(), keyword()) :: t()
  def evolve_steps(model, steps, opts \\ [])

  def evolve_steps(model, 0, _opts), do: model

  def evolve_steps(model, steps, opts) do
    model
    |> evolve_step(opts)
    |> evolve_steps(steps - 1, opts)
  end

  @doc """
  Finds all possible rule applications in the current state (multiway evolution).
  Returns a list of possible next states.
  """
  @spec multiway_step(t()) :: [t()]
  def multiway_step(model) do
    all_matches = find_all_matches(model)

    all_matches
    |> Enum.map(fn {rule, match_data} ->
      apply_rule(model, rule, match_data)
    end)
  end

  @doc """
  Explores the multiway graph up to a specified depth.
  Returns a tree of possible evolution paths.
  """
  @spec multiway_explore(t(), non_neg_integer()) :: %{model: t(), children: [map()]}
  def multiway_explore(model, 0) do
    %{model: model, children: []}
  end

  def multiway_explore(model, depth) do
    next_states = multiway_step(model)
    children = Enum.map(next_states, &multiway_explore(&1, depth - 1))

    %{model: model, children: children}
  end

  @doc """
  Analyzes the causal structure of evolution events.
  """
  @spec analyze_causality(t()) :: map()
  def analyze_causality(model) do
    events = model.causal_network

    dependencies =
      for e1 <- events, e2 <- events, causally_related?(e1, e2), do: {e1, e2}

    %{
      event_count: length(events),
      causal_edges: length(dependencies),
      generations: model.generation,
      causal_density:
        if(length(events) > 1,
          do: length(dependencies) / (length(events) * (length(events) - 1)),
          else: 0.0
        )
    }
  end

  @doc """
  Extracts emergent properties from the evolved hypergraph.

  Builds an adjacency map once and reuses it across analytics to avoid
  repeated scanning of hyperedges (performance improvement).
  """
  @spec analyze_emergence(t()) :: map()
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
  Creates a visualization-friendly representation of the causal network.
  """
  @spec causal_network_data(t()) :: %{nodes: [map()], edges: [map()]}
  def causal_network_data(model) do
    events = Enum.with_index(model.causal_network)

    nodes =
      events
      |> Enum.map(fn {event, idx} ->
        %{
          id: idx,
          generation: event.generation,
          rule_name: event.rule.name,
          type: "event"
        }
      end)

    edges =
      for {e1, idx1} <- events,
          {e2, idx2} <- events,
          idx1 < idx2,
          causally_related?(e1, e2) do
        %{
          source: idx1,
          target: idx2,
          type: "causal"
        }
      end

    %{nodes: nodes, edges: edges}
  end

  defp find_first_match(model) do
    model.rules
    |> Enum.find_value(fn rule ->
      case find_pattern_match(model.hypergraph, rule.pattern) do
        nil -> nil
        match_data -> {rule, match_data}
      end
    end)
  end

  defp find_all_matches(model) do
    model.rules
    |> Enum.flat_map(fn rule ->
      find_all_pattern_matches(model.hypergraph, rule.pattern)
      |> Enum.map(fn match_data -> {rule, match_data} end)
    end)
  end

  defp find_pattern_match(hypergraph, pattern) do
    hyperedges = Hypergraph.hyperedges(hypergraph)
    WolframModel.Matcher.match(hyperedges, pattern)
  end

  defp find_all_pattern_matches(hypergraph, pattern) do
    hyperedges = Hypergraph.hyperedges(hypergraph)
    WolframModel.Matcher.match_all(hyperedges, pattern)
  end

  defp apply_rule(model, rule, match_data) do
    # Remove matched hyperedges
    new_hg =
      match_data.matched_hyperedges
      |> Enum.reduce(model.hypergraph, fn he, hg ->
        Hypergraph.remove_hyperedge(hg, he)
      end)

    # Add replacement hyperedges with proper vertex substitution
    new_hg =
      rule.replacement
      |> Enum.reduce(new_hg, fn replacement_he, hg ->
        actual_vertices =
          substitute_vertices(
            replacement_he,
            match_data.mapping,
            model.generation,
            model.id_generator
          )

        Hypergraph.add_hyperedge(hg, MapSet.to_list(actual_vertices))
      end)

    # Create evolution event for causal tracking
    event = %{
      generation: model.generation + 1,
      rule: rule,
      matched_hyperedges: match_data.matched_hyperedges,
      # TODO Could be more sophisticated
      position: :global
    }

    %{
      model
      | hypergraph: new_hg,
        generation: model.generation + 1,
        causal_network: [event | model.causal_network],
        evolution_history: [new_hg | model.evolution_history]
    }
  end

  defp substitute_vertices(
         replacement_he,
         mapping,
         generation,
         id_generator
       ) do
    replacement_he
    |> MapSet.to_list()
    |> Enum.map(fn vertex ->
      case vertex do
        :new -> {:new, generation, id_generator.()}
        :center -> {:center, generation, id_generator.()}
        :parallel -> {:parallel, generation, id_generator.()}
        :new1 -> {:new1, generation, id_generator.()}
        :new2 -> {:new2, generation, id_generator.()}
        :new3 -> {:new3, generation, id_generator.()}
        # For spacetime coordinates
        {tag, offset} -> {tag, offset}
        _ -> Map.get(mapping, vertex, vertex)
      end
    end)
    |> MapSet.new()
  end

  # Two rewrite events are causally connected if they involve overlapping elements from the
  # hypergraph. For instance, if one rule application modifies a hyperedge, and a subsequent
  # rule application involves that same hyperedge (or elements connected to it), there's a
  # causal relationship between these events.
  defp causally_related?(event1, event2) do
    event1.generation < event2.generation and
      not Enum.empty?(
        for(
          e1 <- event1.matched_hyperedges,
          e2 <- event2.matched_hyperedges,
          do: MapSet.to_list(MapSet.intersection(e1, e2))
        )
        |> List.flatten()
      )
  end

  defp calculate_clustering_coefficient(adjacency_map) do
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
            # Count connected neighbor pairs using adjacency map
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

  defp calculate_complexity(hg) do
    case CorrelationLength.compute(hg) do
      {:ok, correlation_length} ->
        correlation_length

      _ ->
        # Not enough data to calculate Correlation Length
        0
    end
  end

  defp build_adjacency_map(hg) do
    # Build a map of vertex -> MapSet of neighbors by iterating hyperedges once.
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

    # Ensure isolated vertices are represented with empty neighbor sets
    Enum.reduce(MapSet.to_list(Hypergraph.vertices(hg)), adj, fn v, acc ->
      Map.put_new(acc, v, MapSet.new())
    end)
  end

  defp estimate_diameter(adjacency_map) do
    # Compute graph diameter (longest shortest path) using BFS over adjacency_map.
    # For disconnected graphs, compute diameter of each connected component and
    # return the maximum; for an isolated vertex we return 1 (consistent with prior behavior).

    vertices = Map.keys(adjacency_map)

    if vertices == [] do
      1
    else
      distances =
        Enum.map(vertices, fn v ->
          # BFS distances from v
          bfs_distances(adjacency_map, v)
          |> Map.values()
          |> Enum.filter(&is_integer/1)
          |> Enum.max(fn -> 0 end)
        end)

      max_distance = Enum.max(distances)
      max(1, max_distance)
    end
  end

  # BFS from source returning map vertex -> distance
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

  defp calculate_growth_rate(model) do
    history = model.evolution_history

    if length(history) < 2 do
      0.0
    else
      recent = List.first(history) |> Hypergraph.vertex_count()
      previous = Enum.at(history, 1) |> Hypergraph.vertex_count()
      if previous == 0, do: 0.0, else: (recent - previous) / previous
    end
  end

  @doc """
  Helper function to print evolution statistics.
  """
  @spec print_stats(t()) :: :ok
  def print_stats(model) do
    emergence = analyze_emergence(model)
    causality = analyze_causality(model)

    IO.puts("=== Wolfram Model Evolution Statistics ===")
    IO.puts("Generation: #{model.generation}")
    IO.puts("Vertices: #{emergence.vertex_count}")
    IO.puts("Hyperedges: #{emergence.hyperedge_count}")
    IO.puts("Growth rate: #{Float.round(emergence.growth_rate, 3)}")
    IO.puts("Complexity measure: #{emergence.complexity_measure}")
    IO.puts("Causal events: #{causality.event_count}")
    IO.puts("Causal density: #{Float.round(causality.causal_density, 3)}")
    IO.puts("==========================================")
  end
end
