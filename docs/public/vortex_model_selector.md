# Vortex v1 Model Selector Reference

`vortex_model_selector` searches VortexHash training configurations for a vector
dataset. It optimizes deployment locality quality, recall, model size, training
cost, and hash latency under explicit candidate and wall-time budgets.

## Basic Usage

```bash
build/hash_functions_core/Vortex/vortex_model_selector \
  --dataset vector_datasets/siftsmall/siftsmall_base.fvecs \
  --nsw vortex_v1_output/siftsmall_nsw.csr \
  --query vector_datasets/siftsmall/siftsmall_query.fvecs \
  --output vortex_v1_output/siftsmall_selector.json
```

When an HNSW index is available, the selector can search skeleton size and model
parameters together. If no target skeleton values are supplied, it uses the
scale-aware auto policy described below:

```bash
build/hash_functions_core/Vortex/vortex_model_selector \
  --dataset vector_datasets/siftsmall/siftsmall_base.fvecs \
  --index vortex_v1_output/siftsmall_hnsw.index \
  --query vector_datasets/siftsmall/siftsmall_query.fvecs \
  --output vortex_v1_output/siftsmall_selector_auto.json
```

The selector writes JSON to `--output` and generates an HTML report beside it by
default. Use `--disable-html-report` to skip HTML or `--html-report <path>` to
choose a path.

The HTML report is optimized as a compact summary rather than a raw telemetry dump. It
opens with the selected deployment-locality winner, a short metric guide, search
health and budget state, recommendation roles, and phase-2 figures. Expensive
diagnostics such as per-context beam telemetry and base-rank cache attribution
are still present, but collapsed by default so the main report stays focused.

## Search Strategy

The default selector is scale-aware and overlay-match first:

- `--index` with no explicit skeleton targets auto-selects a bounded
  target-skeleton ladder from dataset size; full SIFT reaches the full base
  layer while very large future datasets are capped unless recall mode is
  requested;
- `--target-skeleton-values` and `--target-skeleton-percentages` are exact
  overrides for controlled sweeps;
- dataset and skeleton size shape the effective `K` ladder;
- `centroid_knn`, CDF branch factor, CDF model family, and 2-opt effort are
  coupled with `K` instead of searched as an independent flat grid;
- centroid ordering keeps the Euclidean centroid kNN/MST/2-opt order as the
  phase-1 baseline, then compares graph-aware order variants during refinement.
  The graph-aware variant builds a sparse centroid-transition graph from NSW
  skeleton edges and point-to-centroid labels, using it as a bounded tie-breaker
  for local centroid neighborhoods. The report includes
  `selection.graph_centroid_order_evidence`, so runs show whether graph-aware or
  Euclidean ordering measured better on that dataset. Use
  `--disable-graph-centroid-order` to reproduce Euclidean-only ordering;
- centroid construction uses the selected NSW skeleton, while range mass and
  CDF-distance samples use exact full-dataset assignments at current SIFT scale.
  Above that scale, the selector auto-caps assignment samples to keep training
  bounded. Use `--assignment-sample-limit 0` to force exact assignment,
  `--assignment-sample-limit <n>` to set an explicit cap, or
  `--disable-full-dataset-assignments` only to reproduce legacy skeleton-only
  assignment behavior;
- phase-1 probes prune weak families before expensive full evaluation;
- progressive high-`K` guards use fixed-window recall only in
  `--optimize-for-recall` mode. In the default deployment mode they use
  `locality_quality_score`, so larger-`K` candidates are not discarded just
  because recall saturates before overlay match does;
- beam/hill refinement spends budget near promising configurations and stops
  when the best locality-quality score no longer improves by
  `--min-quality-improvement`;
- when graph-aware and Euclidean centroid-order variants are both enabled, the
  selector shares one graph-capable training base and CDF fit across them while
  keeping the final centroid-order/model caches policy-specific. If graph-aware
  ordering is disabled, training keeps the cheaper graphless base;
- large sampled beam and refinement evaluations reuse exact nearest-centroid
  caches across CDF/branch/order variants that share centroids;
- final calibration promotes bounded candidates against the full base set and
  prewarms exact shared nearest-centroid caches for ready models before parallel
  base-rank evaluation;
- nearest-cache-backed 64-bit base-rank evaluation uses exact typed linear/cubic
  hash-fill kernels with parity sampling against the runtime model;
- remaining-time and model-train guards stop before budget overruns;
- 10M-scale recall searches receive larger default time and model-train budgets
  than 1M-scale runs, but context-level plateau stopping remains available via
  `--max-stagnant-contexts` so longer runs must keep improving quality.

Use `--max-search-seconds`, `--max-model-trains`, `--selector-parallelism 0`,
and `--selector-memory-budget-bytes` to bound runtime and resource use.

The default assignment cap is exact through 2M vectors. Above that, balanced
runs cap assignment samples at 2M, 3M, or 5M as dataset scale grows; recall
mode raises those caps to 3M, 5M, or 10M. The sample is deterministic and evenly
spaced over the base vector file, so repeated runs with the same input produce
the same model.

For distributed Vortex deployments, use `overlay_match_score` as the primary
placement signal. It asks whether exact true nearest neighbors land close in
the simulated overlay ring for the configured 32-1024 node range. Top true
neighbors receive more weight, and match quality decays continuously with node
distance, so this score is more sensitive than a single fixed hash window or a
same-node hit rate. `node_locality_score` remains reported as the compatibility
bucket metric for same-node, +/-1-node, +/-2-node, and +/-4-node hit rates.
Recall is still reported and should be used for ANN-quality studies, but
fixed-window recall alone is less sensitive to peer fanout.

The selector ranks candidates with `locality_quality_score`, a bounded composite
score that gives most weight to overlay match and keeps node-locality,
recall-curve, query-tail, and rank-distance signals as guardrails. Use
`--min-quality-improvement` to control quality-plateau stopping.
`--min-recall-improvement` remains available for recall-only refinement and as
a compatibility alias for older scripts.

For large datasets, use selector metrics as search signals and generated-code
`vortex_eval` as the final acceptance signal. Sampled or screened selector
candidates can reorder under full materialization, so the benchmark pack reports
both selector-vs-`vortex_eval` recall delta and generated-code
`eval_overlay_match_score`/`eval_node_locality_score` beside every materialized
run.

## Default Scale Policy

When using `--index` without explicit skeleton targets, the selector searches
larger skeletons as dataset size grows:

| Base vectors | Auto `target_skeleton` anchors |
| --- | --- |
| `<=20K` | `20%`, `50%`, `100%` |
| `<=100K` | `10K`, `30K`, `60%`, `100%` |
| `<=500K` | `10K`, `30K`, `100K`, `300K`, `100%` |
| `<=2M` | `30K`, `100K`, `300K`, `1M`, `100%` |
| `<=10M` | `100K`, `300K`, `1M`, `3M`, plus `5M` in recall mode |
| `<=50M` | `300K`, `1M`, `3M`, plus `5M` or `10M` in recall mode |
| `>50M` | `1M`, `3M`, `10M`, plus `30M` in recall mode |

The default `K` ladder is generated from dataset scale and then filtered by
the extracted skeleton capacity and a stable k-means training-density guard.
Recall mode searches more aggressive caps: full SIFT can reach about `K=24K`
with a full 1M-node skeleton, 5M-scale datasets can reach `K=65536`, and
100M+ datasets can reach `K=262144` when the skeleton is large enough.
When multiple skeleton contexts are compared, capacity scaling keeps low,
middle, and high K probes for smaller contexts instead of retaining only the
largest K values. Explicit `--K-values` always override the auto ladder.

## Recommendation Roles

The JSON and HTML report expose several roles:

- `peak_recall`: highest calibrated recall found.
- `knee`: balanced recall/cost tradeoff.
- `fast`: fastest useful candidate.
- `small`: smallest useful model.
- `best`: weighted deployment objective, overlay-match first.

For benchmark packs, `--materialize-role auto` materializes `best`. Use
`--materialize-role peak_recall` only when the experiment specifically asks for
maximum fixed-window recall.

On SIFT-scale runs, prefer `best` for Vortex deployment placement studies and
verify it with the benchmark pack's generated-code `vortex_eval` stage. Prefer
`peak_recall` for recall-only studies and report that it optimizes a different
role.

## Metrics

Candidate metrics include:

- `overlay_match_score`, the primary deployment-match score used by the default
  selector objective;
- `overlay_match_curve`, the match score for each evaluated overlay size;
- `query_overlay_match_p05/p50/p95`, tail/median/head query match scores;
- `node_locality_score`, the compatibility deployment score;
- `node_locality_curve`, with same-node, +/-1-node, +/-2-node, and +/-4-node
  true-neighbor hit rates for each evaluated overlay size;
- `recall_curve` and `recall_auc_log_window` over configured hash windows;
- `recall_at_k_in_window` for the primary window;
- query-tail recall percentiles;
- rank-distance percentiles;
- `model_size_bytes` and `model_memory_bytes`;
- `config.graph_centroid_order`, whether the candidate used the graph-aware
  centroid-ordering variant;
- hash/evaluation latency substages;
- candidate training, cache-hit, and final-calibration nearest-cache prewarm
  telemetry.

Treat phase-1 and phase-2 recall/match values as screening evidence. They are
useful for search guidance and skeleton/K comparison, but they may use smaller
base/query limits than the final artifact check. For a selected model, report
the `phase=final_calibration` role metrics and the benchmark pack's
generated-code `vortex_eval` metrics as the correctness reference.

The default node-count ladder is `32,64,128,256,512,1024`, filtered by the
evaluation base sample size. Override it with `--node-count-values <csv>` when
studying a different deployment size.

Selector-level metrics are available in two forms:

- flat `selection.*` fields for compatibility with existing scripts;
- grouped `selection.diagnostics` for new consumers.

`selection.diagnostics_schema_version=1` currently groups telemetry into
`search`, `training`, `scheduler`, `evaluation`, `strategy`,
`final_calibration`, `memory`, and `errors`.

JSON output defaults to `--selector-json-mode compat`, which emits
`selector_json_schema_version=1` and keeps the historical flat telemetry fields.
Use `--selector-json-mode compact` for new consumers that only need the stable
top-level report, candidate rows, recommendation roles, and grouped diagnostics.
Compact output emits `selector_json_schema_version=2` and sets
`selection.flat_compatibility_fields=false`. The default remains compatibility
mode until downstream report consumers have either migrated to grouped
diagnostics or explicitly pinned the compatibility schema.

## Help Modes

```bash
build/hash_functions_core/Vortex/vortex_model_selector --help
build/hash_functions_core/Vortex/vortex_model_selector --help-advanced
```

Normal help shows the public workflow. Advanced help keeps diagnostics,
compatibility flags, manual objective weights, and low-level calibration/search
controls discoverable without crowding the default help output.
