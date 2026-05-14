# WolframModel

The Wolfram Model is a rule-based computational framework in which a system is represented by a hypergraph that evolves through simple, local rewriting rules. Even with simple rules, the model can generate rich, emergent behavior and is used to explore complex systems.

This repository contains a compact Elixir implementation of the Wolfram Model, providing:
* Evolution rules that transform hypergraph patterns
* Causal networks tracking rule applications
* Multiway evolution exploring all possible paths
* Emergent structure analysis

## Installation

```elixir
def deps do
  [
    {:wolfram_model, "~> 0.3.0"}
  ]
end
```

## Key Features

### Evolution Rules

* Pattern matching on hypergraph structures
* Local rewriting rules that transform small patterns
* Multiple rule sets for different behaviors (growth, cellular automaton, spacetime)

### Causal Networks

* Tracks every rule application as an event with `parent_ids` for O(1) causal lookup
* Builds causal relationships between events
* Exports causal graph via `export_event_graph/1` and `causal_network_data/1`
* Computes spacelike foliations (layers of causally independent events) via `foliations/1`
* Checks causal invariance (confluence) of rule applications via `causally_invariant?/1`

### Multiway Evolution

* Explores all possible rule applications
* Creates branching evolution trees
* Implements the "multiway graph" concept from Wolfram's theory
* Builds the branchial graph of conflicting branches via `branchial_graph/1`

### Emergent Structure Analysis

* Measures complexity, growth rates, clustering
* Estimates effective spatial dimension via geodesic ball growth (`Analytics.estimate_dimension/1`)
* Uses an information-theoretic approach to measure spatial coherence (Correlation Length)
* Tracks how simple rules lead to complex structures
* Analyzes diameter and connectivity patterns

### Rule Analysis

* `RuleAnalysis.reversible?/1` — checks structural reversibility
* `RuleAnalysis.self_complementary?/1` — checks pattern/replacement symmetry
* `RuleAnalysis.introduces_new_vertices?/1` — detects vertex-generating rules
* `RuleAnalysis.hyperedge_delta/1` — net hyperedge count change per application
* `RuleAnalysis.arity/1` — hyperedge size signature of a rule

## Example Usage

```elixir
# Create a simple universe
universe = WolframModel.Example.simple_universe()

# Evolve it for 10 steps
evolved = WolframModel.evolve_steps(universe, 10)

# Analyze what emerged
WolframModel.print_stats(evolved)

# Explore multiway evolution
multiway_tree = WolframModel.multiway_explore(universe, 3)

# Analyze causality
causality = WolframModel.Analytics.analyze_causality(evolved)

# Compute spacelike foliations
layers = WolframModel.foliations(evolved)

# Explore the branchial graph of conflicting branches
bg = WolframModel.branchial_graph(universe)

# Check if rule applications commute
WolframModel.causally_invariant?(universe)

# Estimate the emergent spatial dimension
dim = WolframModel.Analytics.estimate_dimension(evolved.hypergraph)

# Inspect rule properties
alias WolframModel.RuleAnalysis
RuleAnalysis.reversible?(rule)
RuleAnalysis.introduces_new_vertices?(rule)
RuleAnalysis.hyperedge_delta(rule)
```

## References

* [The Wolfram Physics Project](https://www.wolframphysics.org/)