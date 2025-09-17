# WolframModel

Elixir module that implements a simplified version of the Wolfram Model, including evolution rules, causal networks, and multiway evolution.

The Wolfram Model provides a computational framework for problems that can be modeled by a hypergraph that evolves according to simple local rewriting rules. This module provides:
* Evolution rules that transform hypergraph patterns
* Causal networks tracking rule applications
* Multiway evolution exploring all possible paths
* Emergent structure analysis

## Installation

If [available in Hex](https://hex.pm/docs/publish), the package can be installed by adding `wolfram_model` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:wolfram_model, "~> 0.1.0"}
  ]
end
```

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
universe = WolframModel.simple_universe()

# Evolve it for 10 steps
evolved = WolframModel.evolve_steps(universe, 10)

# Analyze what emerged
WolframModel.print_stats(evolved)

# Explore multiway evolution
multiway_tree = WolframModel.multiway_explore(universe, 3)

# Analyze causality
causality = WolframModel.analyze_causality(evolved)
```