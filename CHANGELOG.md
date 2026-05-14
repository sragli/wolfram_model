# Changelog

All notable changes to this project are documented in this file.

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
