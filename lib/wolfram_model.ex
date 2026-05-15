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
  Evolves the universe by one step, applying one rule match selected by
  `opts[:ordering]`.

  Supported orderings:
  - `:first` (default) — first match in rule/hyperedge order
  - `:leftmost` — match whose hyperedges have the smallest vertex sort key
  - `:random` — uniformly random match

  Accepts `opts[:id_generator]` to override the model's id generator for
  deterministic behaviour during tests.

  Returns the unchanged model when no rules are applicable (fixpoint).
  """
  @spec evolve_step(t(), keyword()) :: t()
  def evolve_step(model, opts \\ []) do
    id_gen = Keyword.get(opts, :id_generator)
    ordering = Keyword.get(opts, :ordering, :first)
    model = if id_gen, do: %{model | id_generator: id_gen}, else: model

    matches = find_all_matches(model)

    selected =
      case ordering do
        :first -> List.first(matches)
        :leftmost -> select_leftmost(matches)
        :random -> if matches == [], do: nil, else: Enum.random(matches)
      end

    case selected do
      nil -> model
      {rule, match_data} -> apply_rule(model, rule, match_data)
    end
  end

  @doc """
  Returns `true` when no rule can be applied to the current hypergraph
  (the system has reached a fixpoint).
  """
  @spec fixpoint?(t()) :: boolean()
  def fixpoint?(model), do: find_all_matches(model) == []

  @doc """
  Evolves the universe by repeatedly calling `evolve_step/2` until either
  no rule is applicable or `max_steps` is reached.

  Returns the final model. You can check `fixpoint?/1` on the result to
  distinguish a natural fixpoint from a step-limit halt.
  """
  @spec evolve_until_fixpoint(t(), non_neg_integer(), keyword()) :: t()
  def evolve_until_fixpoint(model, max_steps \\ 1_000, opts \\ [])
  def evolve_until_fixpoint(model, 0, _opts), do: model

  def evolve_until_fixpoint(model, remaining, opts) do
    if fixpoint?(model) do
      model
    else
      model |> evolve_step(opts) |> evolve_until_fixpoint(remaining - 1, opts)
    end
  end

  @doc """
  Applies all non-conflicting rule matches in parallel during a single step.

  Matches are processed greedily in rule/hyperedge order: once a hyperedge is
  consumed by a chosen match it is unavailable to subsequent matches in the
  same step. Returns the unchanged model when no matches exist.
  """
  @spec evolve_parallel(t()) :: t()
  def evolve_parallel(model) do
    selected = find_all_matches(model) |> select_non_conflicting()

    Enum.reduce(selected, model, fn {rule, match_data}, m ->
      apply_rule(m, rule, match_data)
    end)
  end

  defp select_leftmost([]), do: nil

  defp select_leftmost(matches) do
    Enum.min_by(matches, fn {_rule, match_data} ->
      match_data.matched_hyperedges
      |> List.flatten()
      |> Enum.map(&vertex_sort_key/1)
      |> Enum.sort()
    end)
  end

  defp vertex_sort_key(v) when is_integer(v), do: {0, v, 0, 0}
  defp vertex_sort_key(v) when is_atom(v), do: {1, Atom.to_string(v), 0, 0}
  defp vertex_sort_key({a, g, id}) when is_atom(a), do: {2, Atom.to_string(a), g, id}
  defp vertex_sort_key(_), do: {3, "", 0, 0}

  defp select_non_conflicting(matches) do
    {selected, _used} =
      Enum.reduce(matches, {[], MapSet.new()}, fn {rule, match_data}, {sel, used} ->
        edges = MapSet.new(match_data.matched_hyperedges)

        if MapSet.disjoint?(edges, used) do
          {[{rule, match_data} | sel], MapSet.union(edges, used)}
        else
          {sel, used}
        end
      end)

    Enum.reverse(selected)
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

  @doc """
  Explores the multiway system as a directed acyclic graph (DAG) up to
  `depth` steps. Unlike `multiway_explore/2`, states that are reachable via
  multiple paths are represented once and converging branches share nodes.

  Returns:
  ```
  %{
    root: canonical_key,
    nodes: %{canonical_key => %WolframModel{}},
    edges: MapSet.t({from_key, to_key})
  }
  ```
  where `canonical_key` is a sorted list-of-lists encoding of the hypergraph.
  """
  @spec multiway_explore_dag(t(), non_neg_integer()) ::
          %{root: term(), nodes: map(), edges: MapSet.t()}
  def multiway_explore_dag(model, depth) do
    root_key = canonical_hypergraph(model.hypergraph)
    acc = %{root: root_key, nodes: %{root_key => model}, edges: MapSet.new()}
    do_explore_dag(model, root_key, depth, acc)
  end

  defp do_explore_dag(_model, _key, 0, acc), do: acc

  defp do_explore_dag(model, parent_key, depth, acc) do
    model
    |> multiway_step()
    |> Enum.reduce(acc, fn child, a ->
      child_key = canonical_hypergraph(child.hypergraph)
      new_nodes = Map.put_new(a.nodes, child_key, child)
      new_edges = MapSet.put(a.edges, {parent_key, child_key})
      new_acc = %{a | nodes: new_nodes, edges: new_edges}

      # Only recurse into nodes we haven't visited yet to avoid cycles.
      if Map.has_key?(a.nodes, child_key) do
        new_acc
      else
        do_explore_dag(child, child_key, depth - 1, new_acc)
      end
    end)
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

    # Compute all added hyperedges with a single substitution pass so that
    # each fresh-vertex tag produces exactly one new ID shared between the
    # hypergraph update and the event record.
    # We accumulate a per-tag memoization map so that the same unbound atom
    # appearing in multiple replacement hyperedges always generates the same
    # new vertex within a single rule application.
    removed_hyperedges = match_data.matched_hyperedges

    {added_hyperedges, _memo} =
      rule.replacement
      |> Enum.map_reduce(%{}, fn replacement_he, memo ->
        substitute_vertices_memo(
          replacement_he,
          match_data.mapping,
          model.generation,
          model.id_generator,
          memo
        )
      end)

    # Apply removals then additions to the hypergraph.
    new_hg =
      added_hyperedges
      |> Enum.reduce(new_hg, fn actual_vertices, hg ->
        Hypergraph.add_hyperedge(hg, actual_vertices)
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
  # Atoms present in the mapping are replaced with their bound vertices.
  # Atoms NOT in the mapping are treated as new-vertex tags and produce a unique
  # vertex tuple {atom, generation, id}. Two-element {atom, integer} tuples are
  # kept verbatim as spacetime coordinate literals.
  #
  # `memo` maps unbound atom tags to the fresh vertex already created for them
  # within the current rule application, so each distinct tag gets one stable ID.
  defp substitute_vertices_memo(replacement_he, mapping, generation, id_gen, memo) do
    {vertices, new_memo} =
      Enum.map_reduce(replacement_he, memo, fn vertex, m ->
        case Map.fetch(mapping, vertex) do
          {:ok, mapped} ->
            {mapped, m}

          :error ->
            case vertex do
              {tag, offset} when is_atom(tag) and is_integer(offset) ->
                {{tag, offset}, m}

              atom when is_atom(atom) ->
                case Map.fetch(m, atom) do
                  {:ok, existing} ->
                    {existing, m}

                  :error ->
                    fresh = {atom, generation, id_gen.()}
                    {fresh, Map.put(m, atom, fresh)}
                end

              other ->
                {other, m}
            end
        end
      end)

    {vertices, new_memo}
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
  Checks causal invariance (confluence) for the current model state.

  Tests all pairs of rule matches — both overlapping and non-overlapping:

  - *Non-overlapping pairs*: applied in both orders; the results must be
    identical (immediate commutativity).
  - *Overlapping pairs*: each match is applied independently; then evolution
    continues for up to `depth` additional steps (default 2) on each branch.
    Invariance holds if the two branches reach a common state within that
    depth (local Church-Rosser property).

  Returns `true` if all tested pairs satisfy the check, `false` otherwise.
  An empty match set is trivially invariant.
  """
  @spec causally_invariant?(t(), non_neg_integer()) :: boolean()
  def causally_invariant?(model, depth \\ 2) do
    indexed = find_all_matches(model) |> Enum.with_index()

    pairs =
      for {{r1, m1}, i} <- indexed,
          {{r2, m2}, j} <- indexed,
          i < j do
        {r1, m1, r2, m2}
      end

    Enum.all?(pairs, fn {r1, m1, r2, m2} ->
      overlapping? =
        m1.matched_hyperedges -- m1.matched_hyperedges -- m2.matched_hyperedges != []

      if overlapping? do
        # Overlapping: check local Church-Rosser — after applying one, can we
        # still reach the same final state as after applying the other?
        branch1 = apply_rule(model, r1, m1) |> evolve_steps(depth)
        branch2 = apply_rule(model, r2, m2) |> evolve_steps(depth)

        reachable_states(branch1, depth)
        |> Enum.any?(fn s1 ->
          reachable_states(branch2, depth)
          |> Enum.any?(fn s2 ->
            normalize_for_confluence(s1.hypergraph) ==
              normalize_for_confluence(s2.hypergraph)
          end)
        end)
      else
        # Non-overlapping: must commute immediately.
        s12 = model |> apply_rule(r1, m1) |> apply_rule(r2, m2)
        s21 = model |> apply_rule(r2, m2) |> apply_rule(r1, m1)
        normalize_for_confluence(s12.hypergraph) == normalize_for_confluence(s21.hypergraph)
      end
    end)
  end

  # Collect all states reachable from `model` in up to `steps` single-step
  # multiway expansions (breadth-first), including `model` itself.
  defp reachable_states(model, 0), do: [model]

  defp reachable_states(model, steps) do
    nexts = multiway_step(model)

    [model | nexts |> Enum.flat_map(&reachable_states(&1, steps - 1))]
    |> Enum.uniq_by(fn m -> canonical_hypergraph(m.hypergraph) end)
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
