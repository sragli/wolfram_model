# WolframModel

The Wolfram Model is a rule-based computational framework in which a system is represented by a hypergraph that evolves through simple, local rewriting rules. Even with simple rules, the model can generate rich, emergent behavior and is used to explore complex systems.

This repository contains a full-featured Elixir implementation of the Wolfram Model, providing:
* N-pattern hypergraph rewriting rules (any number of input hyperedges)
* Configurable update orderings and parallel evolution
* Causal networks tracking rule applications
* Multiway DAG evolution exploring all possible paths
* Emergent structure analysis including dimension estimation and conservation laws
* SVG visualizations for hypergraphs, multiway DAGs, branchial graphs, and geodesic plots

## Installation

```elixir
def deps do
  [
    {:wolfram_model, "~> 1.3.0"}
  ]
end
```

## Key Features

### Evolution Rules

* **N-pattern matching** — rules with any number of input hyperedges via recursive backtracking
* Configurable update orderings: `:first`, `:leftmost`, `:random`
* **Parallel evolution** via `evolve_parallel/1` — applies all non-conflicting matches in one step
* **Fixpoint detection** via `fixpoint?/1` and `evolve_until_fixpoint/3`
* Multiple built-in rule sets: `basic_rules/0`, `:growth`, `:cellular_automaton`, `:spacetime`
* Classic Wolfram Physics Project benchmark rules via `RuleSet.rule_set(:wolfram, key)`
* Standard rule notation parser/printer via `WolframModel.Rule.parse/2` and `WolframModel.Rule.to_string/1`

### Causal Networks

* Tracks every rule application as an event with `parent_ids` for O(1) causal lookup
* Builds causal relationships between events
* Exports causal graph via `export_event_graph/1` and `causal_network_data/1`
* Computes spacelike foliations (layers of causally independent events) via `foliations/1`
* Checks causal invariance (confluence) via `causally_invariant?/2` — tests both overlapping pairs (Church-Rosser) and non-overlapping pairs (commutativity)

### Multiway Evolution

* Explores all possible rule applications
* `multiway_explore/2` — branching evolution tree
* `multiway_explore_dag/2` — proper DAG where converging branches share nodes
* Builds the branchial graph of conflicting branches via `branchial_graph/1`

### Emergent Structure Analysis

* Measures complexity, growth rates, clustering
* Estimates effective spatial dimension via **hypergraph geodesic** ball growth (`Analytics.estimate_dimension/1`)
* Estimates **Ricci scalar curvature** from the next-order ball-growth correction (`Analytics.estimate_ricci_scalar/1`) — positive for sphere-like, negative for hyperbolic-like geometries
* Detects conserved quantities (vertex count, edge count, total degree and their parities) via `Analytics.detect_conserved_quantities/1`
* Uses an information-theoretic approach to measure spatial coherence (Correlation Length)
* Analyzes diameter and connectivity patterns

### Visualization

* `HypergraphSVG.to_svg/2` — force-directed layout of a hypergraph; binary edges as directed arrows, N-ary edges as translucent polygons
* `HypergraphSVG.evolution_to_svg/2` — horizontal strip of panels showing every generation
* `MultiwayGraphSVG.to_svg/2` — hierarchical DAG layout of multiway evolution; nodes labelled with vertex/edge counts
* `BranchialGraphSVG.to_svg/2` — circular layout of conflicting rule matches with rule-name legend
* `GeodesicPlotSVG.to_svg/2` — dual-panel chart: linear `V(r)` vs `r` and log-log with best-fit dimension slope
* `CausalGraphSVG.to_svg/1` — generation-layered causal event graph

### Rule Analysis

* `RuleAnalysis.reversible?/1` — checks structural reversibility
* `RuleAnalysis.self_complementary?/1` — checks pattern/replacement symmetry
* `RuleAnalysis.introduces_new_vertices?/1` — detects vertex-generating rules
* `RuleAnalysis.hyperedge_delta/1` — net hyperedge count change per application
* `RuleAnalysis.arity/1` — hyperedge size signature of a rule
* `RuleAnalysis.canonical_form/1` — normalises variable names in first-appearance order
* `RuleAnalysis.equivalent?/2` — checks if two rules are isomorphic up to variable renaming

## Example Usage

```elixir
# Create a simple universe
universe = WolframModel.Example.simple_universe()

# Evolve it for 10 steps (default :first ordering)
evolved = WolframModel.evolve_steps(universe, 10)

# Use leftmost or random ordering
WolframModel.evolve_step(universe, ordering: :leftmost)
WolframModel.evolve_step(universe, ordering: :random)

# Apply all non-conflicting matches in one parallel step
WolframModel.evolve_parallel(universe)

# Evolve until no rules apply (fixpoint)
final = WolframModel.evolve_until_fixpoint(universe)
WolframModel.fixpoint?(final)  # => true

# Analyze what emerged
WolframModel.print_stats(evolved)

# Explore multiway evolution as a tree...
multiway_tree = WolframModel.multiway_explore(universe, 3)

# ...or as a proper DAG where converging branches share nodes
dag = WolframModel.multiway_explore_dag(universe, 3)
# dag.nodes :: %{canonical_key => %WolframModel{}}
# dag.edges :: MapSet of {from_key, to_key}

# Analyze causality
causality = WolframModel.Analytics.analyze_causality(evolved)

# Compute spacelike foliations
layers = WolframModel.foliations(evolved)

# Explore the branchial graph of conflicting branches
bg = WolframModel.branchial_graph(universe)

# Check causal invariance (tests both overlapping and non-overlapping pairs)
WolframModel.causally_invariant?(universe)
WolframModel.causally_invariant?(universe, 3)  # depth-3 Church-Rosser check

# Estimate the emergent spatial dimension (uses hypergraph geodesics)
dim = WolframModel.Analytics.estimate_dimension(evolved.hypergraph)

# Estimate Ricci scalar curvature (positive → sphere-like, negative → hyperbolic)
r_scalar = WolframModel.Analytics.estimate_ricci_scalar(evolved.hypergraph)

# Detect conserved quantities across evolution history
conserved = WolframModel.Analytics.detect_conserved_quantities(evolved)
# => %{conserved: [:edge_count_parity, ...], vertex_count_history: [...], ...}

# Use classic Wolfram Physics Project benchmark rules
rules = WolframModel.RuleSet.rule_set(:wolfram, :rule_1)

# Parse rules from standard Wolfram notation
rule = WolframModel.Rule.parse("{{1,2},{1,3}} -> {{2,3},{1,4}}")
WolframModel.Rule.to_string(rule)
# => "{{1,2},{1,3}} -> {{2,3},{1,4}}"

# Inspect rule properties
alias WolframModel.RuleAnalysis
RuleAnalysis.reversible?(rule)
RuleAnalysis.introduces_new_vertices?(rule)
RuleAnalysis.hyperedge_delta(rule)
RuleAnalysis.canonical_form(rule)
RuleAnalysis.equivalent?(rule_a, rule_b)

# SVG visualizations
alias WolframModel.{HypergraphSVG, MultiwayGraphSVG, BranchialGraphSVG, GeodesicPlotSVG, CausalGraphSVG}

# Render the current hypergraph
evolved.hypergraph
|> HypergraphSVG.to_svg(title: "Step #{evolved.generation}")
|> then(&File.write!("hypergraph.svg", &1))

# Render the full evolution as a strip of panels
evolved
|> HypergraphSVG.evolution_to_svg(max_snapshots: 8, panel_size: 200)
|> then(&File.write!("evolution.svg", &1))

# Render the multiway DAG
dag = WolframModel.multiway_explore_dag(universe, 3)
dag
|> MultiwayGraphSVG.to_svg()
|> then(&File.write!("multiway.svg", &1))

# Render the branchial graph
WolframModel.branchial_graph(universe)
|> BranchialGraphSVG.to_svg(title: "Branchial graph")
|> then(&File.write!("branchial.svg", &1))

# Render the causal graph
evolved
|> WolframModel.causal_network_data()
|> CausalGraphSVG.to_svg()
|> then(&File.write!("causal.svg", &1))

# Render the geodesic ball growth plot with dimension estimate
evolved.hypergraph
|> GeodesicPlotSVG.to_svg(seeds: 5, title: "Geodesic dimension")
|> then(&File.write!("geodesic.svg", &1))
```

## Interactive Livebook

For a step-by-step guided tour — including theory, worked examples, visualisations, and curvature analysis — open [`wolfram_model.livemd`](wolfram_model.livemd) in [Livebook](https://livebook.dev/):

```bash
livebook server wolfram_model.livemd
```

The notebook covers:
1. Wolfram Physics background and core concepts
2. Building and evolving universes
3. Update orderings and parallel evolution
4. Causal networks, foliations, and causal invariance
5. Multiway evolution and branchial graphs
6. Emergent spatial dimension and Ricci scalar curvature
7. Classic Wolfram benchmark rules
8. Rule analysis and conservation law detection

## References

* [The Wolfram Physics Project](https://www.wolframphysics.org/)
* [Technical Introduction](https://www.wolframphysics.org/technical-introduction/)
* [arXiv: 2004.08210](https://arxiv.org/abs/2004.08210)