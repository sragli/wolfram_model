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
    {:wolfram_model, "~> 0.2.0"}
  ]
end
```

## Release notes

See `CHANGELOG.md` for full release notes.

## Key Features

### Evolution Rules

* Pattern matching on hypergraph structures
* Local rewriting rules that transform small patterns
* Multiple rule sets for different behaviors (growth, cellular automaton, spacetime)

### Causal Networks

* Tracks every rule application as an event
* Builds causal relationships between events
* Analyzes causal structure and density

### Multiway Evolution

* Explores all possible rule applications
* Creates branching evolution trees
* Implements the "multiway graph" concept from Wolfram's theory

### Emergent Analysis

* Measures complexity, growth rates, clustering
* Uses an information-theoretic approach to measure spatial coherence (Correlation Length, the distance between regions where Mutual Information drops to 1/e of its maximum)
* Tracks how simple rules lead to complex structures
* Analyzes diameter and connectivity patterns

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
causality = WolframModel.analyze_causality(evolved)
```