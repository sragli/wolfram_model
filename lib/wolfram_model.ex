defmodule WolframModel do
  @moduledoc """
  A simplified implementation of the Wolfram Model using hypergraphs.
  """
  alias Hypergraph

  defstruct hypergraph: %Hypergraph{},
            generation: 0,
            causal_network: [],
            evolution_history: [],
            rules: [],
            id_generator: &WolframModel.default_id_gen/0,
            adjacency_map: %{},
            # Event bookkeeping
            next_event_id: 1,
            event_index: %{},
            event_map: %{}

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

  defmodule Event do
    @moduledoc """
    Represents a single rewrite event in the Wolfram Model evolution.

    Fields:
    - `id` - unique event id
    - `generation` - generation when the event occurred
    - `rule` - the rule applied
    - `matched_hyperedges` - the hyperedges matched (as MapSet)
    - `removed` - hyperedges removed (list of MapSet)
    - `added` - hyperedges added (list of MapSet)
    - `affected_vertices` - MapSet of vertices affected
    - `position` - richer position information (map)
    - `parent_ids` - list of parent event ids
    - `metadata` - optional map for extra info
    """

    @enforce_keys [:id, :generation, :rule]
    defstruct id: nil,
              generation: 0,
              rule: nil,
              matched_hyperedges: [],
              removed: [],
              added: [],
              affected_vertices: MapSet.new(),
              position: :global,
              parent_ids: [],
              metadata: %{}

    @type t :: %__MODULE__{
            id: integer(),
            generation: non_neg_integer(),
            rule: WolframModel.rule(),
            matched_hyperedges: [MapSet.t()],
            removed: [MapSet.t()],
            added: [MapSet.t()],
            affected_vertices: MapSet.t(),
            position: any(),
            parent_ids: [integer()],
            metadata: map()
          }
  end

  @type t :: %__MODULE__{
          hypergraph: Hypergraph.t(),
          generation: non_neg_integer(),
          causal_network: [Event.t()],
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
    adj = WolframModel.Analytics.build_adjacency_map(initial_hypergraph)

    %__MODULE__{
      hypergraph: initial_hypergraph,
      rules: rules,
      evolution_history: [initial_hypergraph],
      id_generator: id_gen,
      adjacency_map: adj
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
  Returns a deduplicated list of possible next states (unique hypergraphs).
  """
  @spec multiway_step(t()) :: [t()]
  def multiway_step(model) do
    all_matches = find_all_matches(model)

    next_models =
      all_matches
      |> Enum.map(fn {rule, match_data} ->
        apply_rule(model, rule, match_data)
      end)

    # Deduplicate by canonical hypergraph representation to avoid redundant
    # identical states differing only by which match produced them.
    next_models
    |> Enum.reduce(%{}, fn m, acc ->
      key = canonical_hypergraph(m.hypergraph)
      Map.put_new(acc, key, m)
    end)
    |> Map.values()
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
  Analyzes the causal structure of evolution events using event indices for efficiency.
  Returns counts and density.
  """
  @spec analyze_causality(t()) :: map()
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
    hypergraph
    |> Hypergraph.hyperedges()
    |> WolframModel.Matcher.match(pattern)
  end

  defp find_all_pattern_matches(hypergraph, pattern) do
    hypergraph
    |> Hypergraph.hyperedges()
    |> WolframModel.Matcher.match_all(pattern)
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

        # Hypergraph.add_hyperedge accepts MapSet directly
        Hypergraph.add_hyperedge(hg, actual_vertices)
      end)

    # Update adjacency_map cache for the modified hypergraph
    new_adj = WolframModel.Analytics.build_adjacency_map(new_hg)

    # Determine removed and added hyperedges
    removed_hyperedges = match_data.matched_hyperedges |> Enum.map(& &1)

    # collect added hyperedges (as MapSet) from the replacements
    added_hyperedges =
      rule.replacement
      |> Enum.map(fn replacement_he ->
        substitute_vertices(
          replacement_he,
          match_data.mapping,
          model.generation,
          model.id_generator
        )
      end)

    # affected vertices
    affected_vertices =
      (removed_hyperedges ++ added_hyperedges)
      |> Enum.flat_map(&MapSet.to_list/1)
      |> MapSet.new()

    # parent ids: events that previously touched any affected vertex
    parent_ids =
      affected_vertices
      |> Enum.flat_map(fn v -> Map.get(model.event_index, v, MapSet.new()) |> MapSet.to_list() end)
      |> Enum.uniq()

    # Create evolution event for causal tracking with richer info
    id = model.next_event_id

    event = %Event{
      id: id,
      generation: model.generation + 1,
      rule: rule,
      matched_hyperedges: removed_hyperedges,
      removed: removed_hyperedges,
      added: added_hyperedges,
      affected_vertices: affected_vertices,
      position: %{
        type: :local,
        hyperedge_keys: Enum.map(removed_hyperedges, &canonical_hyperedge/1),
        vertices: MapSet.to_list(affected_vertices)
      },
      parent_ids: parent_ids,
      metadata: %{}
    }

    # Update event indices for quick causality lookups
    new_event_index =
      Enum.reduce(affected_vertices, model.event_index, fn v, acc ->
        Map.update(acc, v, MapSet.new([id]), &MapSet.put(&1, id))
      end)

    # Persist event in event map and prepend to causal_network
    new_event_map = Map.put(model.event_map, id, event)

    %{
      model
      | hypergraph: new_hg,
        generation: model.generation + 1,
        causal_network: [event | model.causal_network],
        evolution_history: [new_hg | model.evolution_history],
        adjacency_map: new_adj,
        next_event_id: id + 1,
        event_index: new_event_index,
        event_map: new_event_map
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
      Enum.any?(event1.matched_hyperedges, fn e1 ->
        Enum.any?(event2.matched_hyperedges, fn e2 ->
          MapSet.size(MapSet.intersection(e1, e2)) > 0
        end)
      end)
  end

  @doc """
  Export the causal event graph as nodes and edges suitable for visualization.

  Nodes use `event.id`; edges are `{source: parent_id, target: child_id}`.
  """
  @spec export_event_graph(t()) :: %{nodes: [map()], edges: [map()]}
  def export_event_graph(model) do
    events = model.causal_network

    nodes =
      events
      |> Enum.map(fn e ->
        %{
          id: e.id,
          generation: e.generation,
          rule_name: e.rule.name,
          affected_vertex_count: MapSet.size(e.affected_vertices)
        }
      end)

    edges =
      events
      |> Enum.flat_map(fn e ->
        Enum.map(e.parent_ids, fn pid -> %{source: pid, target: e.id} end)
      end)

    %{nodes: nodes, edges: edges}
  end

  # Canonical representation of hypergraph for quick deduplication.
  # Returns a sorted list of sorted vertex lists which can be used as a map key.
  defp canonical_hypergraph(hg) do
    hg
    |> Hypergraph.hyperedges()
    |> Enum.map(&canonical_hyperedge/1)
    |> Enum.sort()
  end

  # Canonical key for a single hyperedge (for event position indexing)
  defp canonical_hyperedge(he) do
    Enum.sort(he)
  end

  @doc """
  Helper function to print evolution statistics.
  """
  @spec print_stats(t()) :: :ok
  def print_stats(model) do
    emergence = WolframModel.Analytics.analyze_emergence(model)
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
