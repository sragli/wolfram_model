defmodule WolframModel do
  @moduledoc """
  A simplified implementation of the Wolfram Model using hypergraphs.
  """
  alias Hypergraph
  alias WolframModel.CorrelationLength

  defstruct hypergraph: %Hypergraph{},
            generation: 0,
            causal_network: [],
            evolution_history: [],
            rules: []

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
          rules: [rule()]
        }

  @doc """
  Creates a new Wolfram Model universe with initial hypergraph and rules.
  """
  @spec new(Hypergraph.t(), [rule()]) :: t()
  def new(initial_hypergraph, rules) do
    %__MODULE__{
      hypergraph: initial_hypergraph,
      rules: rules,
      evolution_history: [initial_hypergraph]
    }
  end

  @doc """
  Evolves the universe by one step, applying the first applicable rule.
  """
  @spec evolve_step(t()) :: t()
  def evolve_step(model) do
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
  """
  @spec evolve_steps(t(), non_neg_integer()) :: t()
  def evolve_steps(model, 0), do: model

  def evolve_steps(model, steps) do
    model
    |> evolve_step()
    |> evolve_steps(steps - 1)
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
  """
  @spec analyze_emergence(t()) :: map()
  def analyze_emergence(model) do
    hg = model.hypergraph
    stats = Hypergraph.stats(hg)

    Map.merge(stats, %{
      clustering_coefficient: calculate_clustering_coefficient(hg),
      estimated_diameter: estimate_diameter(hg),
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

    # Simple pattern matching - find first occurrence
    case pattern do
      [single_pattern] ->
        # Single hyperedge pattern
        match_single_hyperedge(hyperedges, single_pattern)

      [p1, p2] ->
        # Two hyperedge pattern
        match_two_hyperedges(hyperedges, p1, p2)

      _ ->
        # TODO Handle more complex patterns
        nil
    end
  end

  defp find_all_pattern_matches(hypergraph, pattern) do
    hyperedges = Hypergraph.hyperedges(hypergraph)

    case pattern do
      [single_pattern] ->
        match_all_single_hyperedges(hyperedges, single_pattern)

      [p1, p2] ->
        match_all_two_hyperedges(hyperedges, p1, p2)

      _ ->
        []
    end
  end

  defp match_single_hyperedge(hyperedges, pattern) do
    pattern_size = MapSet.size(pattern)

    hyperedges
    |> Enum.find(fn he -> MapSet.size(he) == pattern_size end)
    |> case do
      nil ->
        nil

      matched_he ->
        # Create mapping from pattern to actual vertices
        pattern_list = MapSet.to_list(pattern)
        matched_list = MapSet.to_list(matched_he)
        mapping = Enum.zip(pattern_list, matched_list) |> Map.new()
        %{mapping: mapping, matched_hyperedges: [matched_he]}
    end
  end

  defp match_all_single_hyperedges(hyperedges, pattern) do
    pattern_size = MapSet.size(pattern)

    hyperedges
    |> Enum.filter(fn he -> MapSet.size(he) == pattern_size end)
    |> Enum.map(fn matched_he ->
      pattern_list = MapSet.to_list(pattern)
      matched_list = MapSet.to_list(matched_he)
      mapping = Enum.zip(pattern_list, matched_list) |> Map.new()
      %{mapping: mapping, matched_hyperedges: [matched_he]}
    end)
  end

  defp match_two_hyperedges(hyperedges, p1, p2) do
    p1_size = MapSet.size(p1)
    p2_size = MapSet.size(p2)

    # Find pairs of hyperedges with right sizes that share vertices
    for he1 <- hyperedges,
        he2 <- hyperedges,
        he1 != he2,
        MapSet.size(he1) == p1_size,
        MapSet.size(he2) == p2_size,
        !MapSet.disjoint?(he1, he2) do
      # Try to create consistent mapping
      shared = MapSet.intersection(he1, he2)

      if MapSet.size(shared) > 0 do
        %{mapping: %{}, matched_hyperedges: [he1, he2]}
      end
    end
    |> List.first()
  end

  defp match_all_two_hyperedges(hyperedges, p1, p2) do
    p1_size = MapSet.size(p1)
    p2_size = MapSet.size(p2)

    for he1 <- hyperedges,
        he2 <- hyperedges,
        he1 != he2,
        MapSet.size(he1) == p1_size,
        MapSet.size(he2) == p2_size,
        !MapSet.disjoint?(he1, he2) do
      %{mapping: %{}, matched_hyperedges: [he1, he2]}
    end
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
          substitute_vertices(replacement_he, match_data.mapping, model.generation)

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

  defp substitute_vertices(replacement_he, mapping, generation) do
    replacement_he
    |> MapSet.to_list()
    |> Enum.map(fn vertex ->
      case vertex do
        :new -> :"new_#{generation}_#{:rand.uniform(1_000_000)}"
        :center -> :"center_#{generation}_#{:rand.uniform(1_000_000)}"
        :parallel -> :"parallel_#{generation}_#{:rand.uniform(1_000_000)}"
        :new1 -> :"new1_#{generation}_#{:rand.uniform(1_000_000)}"
        :new2 -> :"new2_#{generation}_#{:rand.uniform(1_000_000)}"
        :new3 -> :"new3_#{generation}_#{:rand.uniform(1_000_000)}"
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
        (for e1 <- event1.matched_hyperedges,
            e2 <- event2.matched_hyperedges,
            do:
              MapSet.to_list(MapSet.intersection(e1, e2)))
        |> List.flatten()
      )
  end

  defp calculate_clustering_coefficient(hg) do
    # Simplified clustering coefficient for hypergraphs
    vertices = Hypergraph.vertices(hg)
    # Placeholder
    if length(vertices) < 3, do: 0.0, else: 0.5
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

  defp estimate_diameter(hg) do
    # Simplified diameter estimation
    vertex_count = Hypergraph.vertex_count(hg)
    max(1, trunc(:math.log2(vertex_count)))
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
