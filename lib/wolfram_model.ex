defmodule WolframModel do
  @moduledoc """
  A simplified implementation of the Wolfram Model using hypergraphs,capturing
  key structural ideas of the Wolfram Model (hypergraph rewriting, event
  causality, multiway branching).
  """
  alias Hypergraph

  defstruct hypergraph: %Hypergraph{},
            generation: 0,
            causal_network: [],
            evolution_history: [],
            rules: [],
            id_generator: &WolframModel.default_id_gen/0,
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
          pattern: [[Hypergraph.vertex()]],
          replacement: [[Hypergraph.vertex()]],
          name: String.t()
        }

  defmodule Event do
    @moduledoc """
    Represents a single rewrite event in the Wolfram Model evolution.

    Fields:
    - `id` - unique event id
    - `generation` - generation when the event occurred
    - `rule` - the rule applied
    - `removed` - hyperedges removed (list of vertex lists)
    - `added` - hyperedges added (list of vertex lists)
    - `affected_vertices` - list of vertices affected
    - `parent_ids` - list of parent event ids
    - `metadata` - optional map for extra info
    """

    @enforce_keys [:id, :generation, :rule]
    defstruct id: nil,
              generation: 0,
              rule: nil,
              removed: [],
              added: [],
              affected_vertices: [],
              parent_ids: [],
              metadata: %{}

    @type t :: %__MODULE__{
            id: integer(),
            generation: non_neg_integer(),
            rule: WolframModel.rule(),
            removed: [[Hypergraph.vertex()]],
            added: [[Hypergraph.vertex()]],
            affected_vertices: [Hypergraph.vertex()],
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

    case find_all_matches(model) |> List.first() do
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
    next_models =
      model
      |> find_all_matches()
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
    children =
      model
      |> multiway_step()
      |> Enum.map(&multiway_explore(&1, depth - 1))

    %{model: model, children: children}
  end

  defp find_all_matches(model) do
    model.rules
    |> Enum.flat_map(fn rule ->
      model.hypergraph
      |> Hypergraph.hyperedges()
      |> WolframModel.Matcher.match(rule.pattern)
      |> Enum.map(fn match_data -> {rule, match_data} end)
    end)
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

        # Hypergraph.add_hyperedge accepts a vertex list
        Hypergraph.add_hyperedge(hg, actual_vertices)
      end)

    # Determine removed and added hyperedges
    removed_hyperedges = match_data.matched_hyperedges |> Enum.map(& &1)

    # collect added hyperedges from the replacements
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

    # affected vertices (deduplicated)
    affected_vertices =
      (removed_hyperedges ++ added_hyperedges)
      |> Enum.flat_map(& &1)
      |> Enum.uniq()

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
      removed: removed_hyperedges,
      added: added_hyperedges,
      affected_vertices: affected_vertices,
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
        next_event_id: id + 1,
        event_index: new_event_index,
        event_map: new_event_map
    }
  end

  # Substitutes vertices in a replacement hyperedge using the pattern mapping.
  # Any atom present in the mapping is replaced with its mapped value.
  # Any atom NOT in the mapping is treated as a fresh-vertex tag and produces a
  # new unique vertex tuple {atom, generation, id}. This handles any rule
  # generically without a hardcoded list of "new vertex" atoms.
  # Two-element {atom, integer} tuples are treated as spacetime coordinates and
  # kept verbatim (they appear as literals in spacetime rules).
  defp substitute_vertices(replacement_he, mapping, generation, id_generator) do
    replacement_he
    |> Enum.map(fn vertex ->
      case Map.fetch(mapping, vertex) do
        {:ok, mapped} ->
          mapped

        :error ->
          case vertex do
            {tag, offset} when is_atom(tag) and is_integer(offset) ->
              # Spacetime coordinate literal — keep as-is
              {tag, offset}

            atom when is_atom(atom) ->
              # Unbound atom in replacement = new vertex generator
              {atom, generation, id_generator.()}

            other ->
              other
          end
      end
    end)
  end

  @doc """
  Creates a visualization-friendly representation of the causal network.
  Each event (i.e., a specific hypergraph update) becomes a vertex in the causal graph.
  Implements the partial order of event dependencies (a DAG): if the application of
  event B depends on the output of event A (e.g., B’s matching pattern overlaps with
  hyperedges created by A), then the causal graph includes a directed edge A → B.
  """
  @spec causal_network_data(t()) :: %{nodes: [map()], edges: [map()]}
  def causal_network_data(model) do
    events = Enum.with_index(model.causal_network)

    # Map event id → index in the causal_network list for edge construction
    id_to_idx =
      events
      |> Enum.map(fn {event, idx} -> {event.id, idx} end)
      |> Map.new()

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
      events
      |> Enum.flat_map(fn {event, idx} ->
        event.parent_ids
        |> Enum.flat_map(fn pid ->
          case Map.fetch(id_to_idx, pid) do
            {:ok, parent_idx} -> [%{source: parent_idx, target: idx, type: "causal"}]
            :error -> []
          end
        end)
      end)
      |> Enum.uniq()

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
  # Lists are ordered; return as-is since order is semantically meaningful
  defp canonical_hyperedge(he) do
    he
  end

  @doc """
  Computes foliations of the causal network: layers of events where each layer
  contains only events whose parents all belong to earlier layers (spacelike
  slices of the causal partial order).

  Returns a list of lists of `Event.t()`, ordered from earliest to latest layer.
  Layer 0 contains root events (no causal parents). Layer N contains events
  whose deepest ancestor is in layer N-1.
  """
  @spec foliations(t()) :: [[Event.t()]]
  def foliations(%__MODULE__{event_map: []}), do: []

  def foliations(model) do
    events =
      model.event_map
      |> Enum.sort_by(fn {id, _} -> id end)
      |> Enum.map(fn {_, event} -> event end)

    {layer_groups, _id_to_layer} =
      Enum.reduce(events, {%{}, %{}}, fn event, {layer_groups, id_to_layer} ->
        layer =
          if event.parent_ids == [] do
            0
          else
            event.parent_ids
            |> Enum.map(fn pid -> Map.get(id_to_layer, pid, 0) end)
            |> Enum.max()
            |> Kernel.+(1)
          end

        updated_groups = Map.update(layer_groups, layer, [event], &(&1 ++ [event]))
        {updated_groups, Map.put(id_to_layer, event.id, layer)}
      end)

    layer_groups
    |> Enum.sort_by(fn {layer, _} -> layer end)
    |> Enum.map(fn {_, evts} -> evts end)
  end

  @doc """
  Builds the branchial graph for one step of multiway evolution.

  Nodes are the possible rule matches at the current state. Two matches are
  branchially connected when they overlap (touch at least one common hyperedge),
  meaning they represent conflicting/branching rewrite choices.

  Returns `%{nodes: [map()], edges: [map()]}`.
  """
  @spec branchial_graph(t()) :: %{nodes: [map()], edges: [map()]}
  def branchial_graph(model) do
    indexed =
      find_all_matches(model)
      |> Enum.with_index()

    nodes =
      Enum.map(indexed, fn {{rule, match_data}, idx} ->
        %{
          id: idx,
          rule_name: rule.name,
          matched_hyperedges: match_data.matched_hyperedges
        }
      end)

    edges =
      for {{_, m1}, i} <- indexed,
          {{_, m2}, j} <- indexed,
          i < j,
          Enum.any?(m1.matched_hyperedges, fn he1 ->
            Enum.any?(m2.matched_hyperedges, fn he2 ->
              he1 -- he1 -- he2 != []
            end)
          end) do
        %{source: i, target: j}
      end

    %{nodes: nodes, edges: edges}
  end

  @doc """
  Checks approximate causal invariance (confluence) for the current model state.

  Tests all pairs of non-overlapping rule matches: applies them in both orders
  and verifies that the resulting hypergraphs are structurally equivalent (same
  normalized canonical form after replacing fresh-vertex IDs with stable tokens).

  Returns `true` if all tested pairs commute, `false` otherwise.
  An empty match set (no applicable rules) is trivially invariant.
  """
  @spec causally_invariant?(t()) :: boolean()
  def causally_invariant?(model) do
    indexed = find_all_matches(model) |> Enum.with_index()

    pairs =
      for {{r1, m1}, i} <- indexed,
          {{r2, m2}, j} <- indexed,
          i < j,
          m1.matched_hyperedges -- m1.matched_hyperedges -- m2.matched_hyperedges == [] do
        {r1, m1, r2, m2}
      end

    Enum.all?(pairs, fn {r1, m1, r2, m2} ->
      s12 = model |> apply_rule(r1, m1) |> apply_rule(r2, m2)
      s21 = model |> apply_rule(r2, m2) |> apply_rule(r1, m1)
      normalize_for_confluence(s12.hypergraph) == normalize_for_confluence(s21.hypergraph)
    end)
  end

  # Normalize a hypergraph for confluence comparison by replacing all fresh
  # vertex tuples {atom, int, int} with stable positional tokens so that two
  # structurally identical hypergraphs compare equal even with different IDs.
  defp normalize_for_confluence(hg) do
    edges = Hypergraph.hyperedges(hg)

    fresh_map =
      edges
      |> Enum.flat_map(& &1)
      |> Enum.filter(&fresh_vertex?/1)
      |> Enum.uniq()
      |> Enum.sort()
      |> Enum.with_index()
      |> Map.new(fn {v, i} -> {v, {:_fresh, i}} end)

    edges
    |> Enum.map(fn he ->
      he |> Enum.map(&Map.get(fresh_map, &1, &1)) |> Enum.sort()
    end)
    |> Enum.sort()
  end

  defp fresh_vertex?({a, g, id}) when is_atom(a) and is_integer(g) and is_integer(id), do: true
  defp fresh_vertex?(_), do: false

  @doc """
  Exports the causal event graph as a map of nodes and edges.

  Nodes correspond to events in chronological order. Edges link parent events to
  child events using the `parent_ids` recorded during evolution.
  """
  @spec export_event_graph(t()) :: %{nodes: [map()], edges: [map()]}
  def export_event_graph(model) do
    events = Enum.reverse(model.causal_network)

    nodes =
      Enum.map(events, fn event ->
        %{id: event.id, generation: event.generation, rule_name: event.rule.name}
      end)

    edges =
      events
      |> Enum.flat_map(fn event ->
        Enum.map(event.parent_ids, fn pid ->
          %{source: pid, target: event.id}
        end)
      end)
      |> Enum.uniq()

    %{nodes: nodes, edges: edges}
  end

  @doc """
  Helper function to print evolution statistics.
  """
  @spec print_stats(t()) :: :ok
  def print_stats(model) do
    emergence = WolframModel.Analytics.analyze_emergence(model)
    causality = WolframModel.Analytics.analyze_causality(model)

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
