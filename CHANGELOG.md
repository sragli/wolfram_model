# Changelog

All notable changes to this project are documented in this file.

## v1.0.0 (2026-05-15)

### Breaking changes
- `causally_invariant?/1` now accepts an optional `depth` argument
  (`causally_invariant?/2`); callers that relied on the 1-arity form are
  unaffected (default depth is `2`).
- Fresh-vertex substitution is now memoised per rule application: a single
  unbound atom tag produces one stable vertex ID across all replacement
  hyperedges in the same application. Code that relied on each hyperedge
  receiving a distinct fresh ID for the same tag will observe different IDs.

### New features
- **N-pattern rule matching** — `WolframModel.Matcher.match/2` now supports
  rules with any number of input hyperedges via recursive backtracking.
  Previously only 1- and 2-pattern rules were supported.
- **Update orderings** — `evolve_step/2` accepts `ordering: :first | :leftmost |
  :random` to select which rule match is applied.
- **Parallel evolution** — `evolve_parallel/1` greedily applies all
  non-conflicting matches in a single model step.
- **Fixpoint detection** — `fixpoint?/1` and `evolve_until_fixpoint/3`.
- **Multiway DAG** — `multiway_explore_dag/2` returns a proper DAG
  (`%{root:, nodes:, edges:}`) where converging branches share nodes.
- **Improved causal invariance** — `causally_invariant?/2` now tests both
  non-overlapping pairs (commutativity) and overlapping pairs (local
  Church–Rosser property up to configurable depth).
- **Hypergraph-native dimension estimation** — `Analytics.estimate_dimension/1`
  now uses hypergraph BFS (traversal through whole hyperedges) and up to 10
  seed vertices.
- **Conservation law detection** — `Analytics.detect_conserved_quantities/1`
  scans the evolution history for conserved vertex count, edge count, total
  degree, and their parities.
- **Rule notation parser/printer** — new `WolframModel.Rule` module with
  `parse/2` and `to_string/1` for the standard Wolfram notation
  `{{1,2},{1,3}} -> {{2,3},{1,4}}`.
- **Rule equivalence** — `RuleAnalysis.canonical_form/1` and
  `RuleAnalysis.equivalent?/2` check structural isomorphism up to variable
  renaming.
- **Wolfram benchmark rules** — `RuleSet.rule_set(:wolfram, key)` and
  `RuleSet.wolfram_rules/0` expose five canonical rules from the Wolfram
  Physics Project (including the 3-pattern triangle rule).

## v0.3.0 (2026-05-15)

- Generalised fresh vertex substitution: any unbound atom in a rule replacement
  is now treated as a new-vertex tag, removing the hardcoded `:new`/`:center`/
  `:parallel`/`:new1`/`:new2`/`:new3` special-cases
- Fixed `causal_network_data/1` to derive edges from pre-computed `parent_ids`
  (O(n) instead of O(n²)) and removed dead `causally_related?/2`
- Fixed `calculate_clustering_coefficient/1` guard to use `map_size/1` instead
  of `length/1` on a map (guard now fires correctly for small adjacency maps)
- Added `WolframModel.foliations/1` — spacelike event layers (foliations of the
  causal partial order)
- Added `WolframModel.branchial_graph/1` — branchial connections between
  conflicting/overlapping rule matches at the current state
- Added `WolframModel.causally_invariant?/1` — confluence check: verifies that
  all pairs of non-overlapping rule applications commute
- Added `WolframModel.export_event_graph/1` — exports causal events as a
  `%{nodes, edges}` graph using stored `parent_ids`
- Added `WolframModel.Analytics.estimate_dimension/1` — estimates effective
  spatial dimension via geodesic ball growth and log-log regression
- Added `WolframModel.RuleAnalysis` module with `reversible?/1`,
  `self_complementary?/1`, `introduces_new_vertices?/1`, `hyperedge_delta/1`,
  and `arity/1`

## v0.2.0 (2025-12-29)

- Added property tests for two-hyperedge patterns
- Improved deterministic matching
- Enhanced event/cause indexing and analytics
- Fixed various invariants and Dialyzer warnings
