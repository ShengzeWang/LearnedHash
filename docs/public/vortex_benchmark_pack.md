# Vortex SIFT Benchmark Pack

`tools/vortex_sift_benchmark_pack.py` is the reproducible public benchmark entry
point for VortexHash selector and generated-code runs on SIFT-style datasets.
Run it before changing selector algorithms or hot-path inference code.

Dataset roles:

- `siftsmall` is a test/smoke dataset for fast correctness, report, and
  pipeline checks, not a reported benchmark dataset.
- `sift` is SIFT1M and is one of the reported benchmark datasets. Use it for
  reported VortexHash quality/runtime numbers.
- `gist1m`, `sift10m`, `deep10m_l2`, and `spacev10m` are reported benchmark
  extension profiles. They are accepted only when their validated local
  manifests exist and match the configured base/query/ground-truth paths.

Dataset provenance and future reported benchmark workloads are tracked in
`dataset/SOURCES.md`. The planned public benchmark set is SIFT1M, GIST1M,
SIFT10M, DEEP10M-L2, and SPACEV10M. The 10M profiles must be deterministic
first-10M slices from their official 1B/larger sources, with checksums and exact
ground-truth provenance recorded before reporting benchmark numbers. Use
`tools/vortex_dataset_prepare.py materialize` to crop, patch, validate, and
manifest the canonical 10M triple, then use
`tools/vortex_dataset_prepare.py validate-materialized --spot-check-queries 32`
to prove that a 10M base, query set, and scale-specific ground truth belong to
the same canonical prefix slice. Run `tools/vortex_dataset_prepare.py convert`
after validation to produce Vortex `.fvecs`/`.ivecs` files and record converted
output checksums before running the 10M benchmark-pack profiles.

All public Vortex vector workloads are L2 k-NN workloads. Before using a dataset
in benchmark reports, run `tools/vortex_dataset_prepare.py contract-check` to
verify that the local specs still encode L2 distance, canonical base/query
splits, and exact ground truth over the evaluated base or prefix slice.

Use `tools/vortex_dataset_prepare.py profiles --json` when wiring these
datasets into an external benchmark repo. It emits the stable profile names,
local `vector_datasets/<profile>/` filenames, split contracts, and manifest
requirements without requiring the external repo to import LearnedHash implementation modules.

For TexMex archive datasets, use `tools/vortex_dataset_prepare.py
texmex-materialize --dataset <siftsmall|sift|gist1m>` to download from the
official archive, verify the published MD5 and exact archive size, extract the
expected `.fvecs`/`.ivecs` files, validate dimensions and ground-truth ID
bounds, and write a local manifest. GIST1M runs are manifest-gated like the 10M
profiles, and generated reports include that manifest reference.

Use the profile preflight before expensive selector runs:

```bash
tools/vortex_sift_benchmark_pack.py \
  --profile sift10m \
  --preset smoke \
  --check-profile-only
```

The preflight checks the configured dataset paths, validates the required
manifest, and prints the base/query/profile provenance JSON without building,
training, or evaluating. Full benchmark-pack reports include the manifest path,
manifest SHA-256, source family, split contract, validation status, and
base/query/ground-truth file checksums.

## Selector Scale Validation

Use the selector scale-validation harness when changing selector scale policy
or bringing up new reported benchmark profiles:

```bash
tools/vortex_selector_scale_validation.py \
  --profiles sift,sift10m \
  --presets smoke,baseline,deep \
  --skip-build
```

The harness runs the benchmark pack with explicit per-preset selector budgets,
applies dataset-size budget floors for 1M and 10M profiles, records the exact
child commands, and writes a compact JSON/CSV/HTML summary under
`vortex_v1_output/selector_scale_validation/<run>/`. It is strict:
missing required manifests, missing HNSW indices, absent recall/overlay-match
metrics, or selector/materialized recall mismatches are reported as blocked or
unstable rather than silently passing.

For 10M-scale profiles, the default validation floors are intentionally longer:
`smoke=600s`, `baseline=1200s`, and `deep=3600s`. Baseline and deep runs also
raise the selector candidate/model-train budget and enable a small
context-plateau guard so the extra time is spent on new promising candidates
rather than repeating a saturated skeleton context.

For manifest-gated profiles with official exact ground truth, the benchmark
pack passes the ground-truth file to `vortex_eval`. This keeps 10M validation
on the canonical evaluated slice and avoids repeated brute-force exact search
for every scale-policy report. The generated summary still records the
truth-cache mode and ground-truth load time so reports distinguish cached,
ground-truth-backed, and brute-force truth paths.

For 10M profiles, the benchmark pack builds the missing benchmark HNSW index
automatically before the selector stage. The selector scale-validation harness
is stricter: pass `--build-missing-hnsw` only to
`tools/vortex_selector_scale_validation.py` when you intentionally allow a
validation run to spend time constructing a missing index. Without that flag,
the validation report states the missing index as a blocker.

## Benchmark Artifact Bundle

Use the artifact bundle helper before moving benchmark-pack numbers into a
report or figure:

```bash
tools/vortex_paper_artifact_bundle.py \
  --scale-validation-dir vortex_v1_output/selector_scale_validation/<run> \
  --bundle-name <benchmark-run-name> \
  --copy-reports
```

The helper discovers benchmark-pack run directories from the scale-validation
`summary.json`, verifies that each reported benchmark run has a dataset source
manifest, checks the recorded manifest SHA-256 against the local file, and
writes:

- `artifact_manifest.json`: source manifests, validation report paths, command
  ledger, environment, git commit and dirty state, report paths, checksums, and
  final metric rows;
- `metrics_summary.csv`: compact table for reports;
- `commands.txt`: exact command ledger from benchmark and scale-validation
  manifests;
- `file_checksums.sha256`: checksums for recorded source files and copied small
  reports.

Raw vector payloads, generated models, and build caches are intentionally not
copied. For committed release candidates, rerun with `--require-clean-git`:

```bash
tools/vortex_paper_artifact_bundle.py \
  --scale-validation-dir vortex_v1_output/selector_scale_validation/<run> \
  --bundle-name <benchmark-run-name> \
  --copy-reports \
  --require-clean-git
```

Verify an existing bundle with:

```bash
tools/vortex_paper_artifact_bundle.py \
  --verify-bundle vortex_v1_output/paper_artifacts/<benchmark-run-name>
```

## Fast Smoke

```bash
tools/vortex_sift_benchmark_pack.py \
  --profile siftsmall \
  --preset smoke \
  --build-dir build \
  --skip-build
```

## Full SIFT Baseline

```bash
tools/vortex_sift_benchmark_pack.py \
  --profile sift \
  --preset baseline \
  --build-dir build \
  --budget-seconds 3600
```

Use `--preset deep` for broader recall-oriented search when runtime budget is
available. On 10M-scale profiles, deep is the hour-scale policy intended for
peak-recall exploration.

For full SIFT baseline/deep runs, the pack lets `vortex_model_selector` choose
the default scale-aware target-skeleton, `K`, centroid-neighbor, and CDF-branch
ladders. This now reaches larger skeleton targets automatically and preserves
low/mid K and low-neighbor probes that are important for recall. The selector
also caps default K values to keep k-means training density stable; pass
explicit `--K-values` to run controlled stress sweeps above that default. Use
`--target-skeleton-percentages <csv>` only when you need an explicit controlled
skeleton sweep.

Controlled expansion runs can also override the coupled selector ladders with
`--knn-values`, `--cdf-branches`, `--cdf-models`, and `--two-opt-iters`. Use
these only when intentionally exploring a new parameter band; otherwise prefer
the profile/preset defaults so benchmark runs stay comparable.

By default, selector refinement may compare graph-aware centroid-order
variants against the Euclidean centroid-order baseline. Pass
`--disable-graph-centroid-order` when reproducing legacy Euclidean-only runs.
Selector JSON records `selection.graph_centroid_order_evidence` to show the
best measured quality, overlay-match, and recall for graph-aware and Euclidean
ordering candidates.

Training uses exact full-dataset centroid assignments for range mass and CDF
fitting at SIFT scale. Larger selector runs auto-cap assignment samples to bound
training cost. Pass `--assignment-sample-limit 0` to force exact assignment,
`--assignment-sample-limit <n>` to set an explicit cap, or
`--disable-full-dataset-assignments` only when reproducing legacy skeleton-only
assignment runs.

## What It Measures

The pack records:

- selector wall time and candidate counts;
- selector overlay-match score and final materialized `vortex_eval`
  overlay-match score;
- compatibility selector/node-locality scores;
- selector recall and final materialized `vortex_eval` recall;
- selector/materialized recall validation status and delta;
- selector training/cache/scheduler telemetry;
- generated-code train, codegen, build, cold-load, and warm-load time;
- generated model/package size;
- hash latency average/p95/p99;
- optional hash hot-path breakdowns;
- sampled peak memory;
- dominant-cost recommendations.

Outputs are written under `vortex_v1_output/benchmarks/<run>/` as JSON, CSV,
Markdown, HTML, and raw command logs.

`--materialize-role auto` materializes the selector `best` role, whose objective
is overlay-match first. Use `--materialize-role peak_recall` for recall-only
experiments.

Final generated-code evaluation passes the benchmark pack `--jobs` value to
`vortex_eval --threads`. This parallelizes full-base hash-index cache
construction for 10M-scale runs while preserving deterministic hash outputs and
sorted-cache contents.

## Caches

By default the pack reuses deterministic evaluation artifacts under
`vortex_v1_output/benchmarks/cache/`:

- exact truth labels;
- sorted 64-bit base hash-index caches;
- generated-code shared-library build cache.

Disable caches for cold measurements:

```bash
tools/vortex_sift_benchmark_pack.py \
  --profile siftsmall --preset smoke --build-dir build \
  --eval-cache-root none \
  --codegen-build-cache-root none
```

## Summary Schema

Summary schema version `2` separates actual selector construction work from
repeated candidate-phase accounting:

- `selector_train_*_sum`: actual construction totals copied from `selection.*`.
- `selector_candidate_*_sum`: repeated per-candidate phase totals useful for
  heat maps.
- `materialized_recall_at_k_in_window`: recall measured by the selector for the
  candidate selected for train/codegen.
- `eval_recall_at_k_in_window`: recall measured by the generated-code
  `vortex_eval` run.
- `materialized_overlay_match_score`: overlay match measured by the selector
  for the candidate selected for train/codegen.
- `eval_overlay_match_score`: overlay match measured by generated-code
  `vortex_eval` over the same node-count ladder.
- `materialized_node_locality_score`: node-locality score measured by the
  selector for the candidate selected for train/codegen.
- `eval_node_locality_score`: node-locality score measured by generated-code
  `vortex_eval` over the same node-count ladder.
- `materialized_config_graph_centroid_order`: whether the materialized
  candidate used the graph-aware centroid-ordering variant.
- `selector_centroid_order_policy`: whether graph-aware centroid-order variants
  were included in selector refinement for the run.
- `selector_eval_nearest_cache_inflight_bypasses`: sampled selector evaluations
  that skipped an in-flight nearest-cache wait and hashed directly. The bypass is
  a scheduling optimization only; final calibration still uses the shared cache.
- `selector_eval_nearest_cache_prewarm_requests`,
  `selector_eval_nearest_cache_prewarm_ms`, and
  `selector_eval_nearest_cache_prewarm_threads`: final-calibration shared
  nearest-cache prewarm work for ready models before parallel base-rank
  evaluation. This is an exact cache-construction optimization, not an
  approximate search path.

- `selector_eval_nearest_cache_sampled_wait_ms` and
  `selector_eval_nearest_cache_calibration_wait_ms`: split aggregate
  nearest-cache waits so reports can distinguish sampled-search stalls from
  final-calibration cache reuse.
- `materialized_recall_validation_status`: `matched`, `mismatch`, or
  `unavailable` after comparing selector materialized recall with `vortex_eval`.
- `materialized_selector_vs_eval_recall_delta`: `vortex_eval` recall minus
  selector materialized recall. Non-zero values mean the selector and final
  evaluation were not measuring the same thing closely enough for that run.

Legacy candidate-phase aliases named `selector_<field>_sum` are omitted by
default. Use `--legacy-metric-aliases` only for old spreadsheets/scripts that
have not moved to `selector_candidate_<field>_sum`.

The benchmark pack intentionally invokes `vortex_model_selector` in its default
`--selector-json-mode compat` mode because summary CSV generation reads
historical flat `selection.*` timing fields. Standalone selector consumers that
do not need those flat fields can use compact selector JSON instead. The
benchmark-pack schema check guards this assumption so a future selector JSON
default change does not silently break reported benchmark summaries.

## Locality And Recall Gates

For Vortex deployment work, treat generated-code `eval_overlay_match_score` as
the primary acceptance metric because it measures how tightly exact true
neighbors remain in low-fanout peer neighborhoods for 32-1024 node overlays.
Top true neighbors receive more weight, and the score decays continuously with
node distance, making it more sensitive than a single recall window. Also report
generated-code `eval_node_locality_score` for compatibility with older reports.
For recall-only studies, also report generated-code `eval_recall_at_k_in_window`.

Selector metrics can be sampled or calibrated with fewer queries during search;
they are useful for ranking candidates, but they are not a substitute for full
materialization when deciding whether a policy improved the final artifact. The
benchmark pack surfaces recall consistency through
`materialized_recall_validation_status` and the recall-delta fields above, and
surfaces final deployment locality through `eval_overlay_match_score`.

## Interpreting Dominant Cost

The dominant-cost report uses measured exclusive substages where available. For
base-rank cache work, it separates nearest-cache, hash, sort, rank-index, and
residual overhead instead of treating inclusive build time as one opaque cost.
Use this report to pick the next optimization branch from evidence rather than
speculative tuning.
