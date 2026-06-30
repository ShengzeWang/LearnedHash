# Tool Reference

This page summarizes the command-line tools that are useful to most users.
Compatibility wrappers remain available for older command names; new scripts
should prefer the primary commands shown here.

## Public Runtime And Training CLIs

These binaries are built by the default full CMake build. Library-only builds
skip all CLI tools. Lean runtime builds also skip FAISS-backed Vortex training,
selector, and evaluation tools.

Use `LEARNEDHASH_LOG=trace|debug|info|warn|error|off` for shared training and
selector diagnostics. `RUST_LOG` is retained as an older-script fallback, and
`LEARNEDHASH_LOG` has precedence when both are set.

| Tool | Build path | Audience | Purpose |
| --- | --- | --- | --- |
| `rm_model_learner` | `build/learned_structures/rm_model/rm_model_learner` | public | Train recursive learned models and emit rm_model artifacts. |
| `rm_model_inferencer` | `build/learned_structures/rm_model/rm_model_inferencer` | public | Load trained rm_model artifacts and run scalar inference. |
| `lead_hashgen` | `build/hash_functions_core/LEAD/lead_hashgen` | public | Train/codegen scalar LEAD hash functions. |
| `lead_hasher` | `build/hash_functions_core/LEAD/lead_hasher` | public | Load LEAD artifacts and hash scalar keys. |
| `vortex_v1_cli` | `build/hash_functions_core/Vortex/vortex_v1_cli` | public | Train VortexHash v1 models, extract NSW skeletons, and emit generated-code packages. |
| `vortex_model_selector` | `build/hash_functions_core/Vortex/vortex_model_selector` | public | Search dataset-aware VortexHash training configurations. |
| `vortex_eval` | `build/hash_functions_core/Vortex/vortex_eval` | public | Evaluate VortexHash generated-code recall, overlay locality, and hash latency. |
| `hnsw_eval` | `build/hash_functions_core/Vortex/hnsw_eval` | diagnostic | Build or inspect FAISS HNSW indices used by VortexHash training and benchmarks. |

## Public Benchmark And Dataset Tools

| Tool | Tier | Purpose |
| --- | --- | --- |
| `tools/vortex_sift_benchmark_pack.py` | compatibility wrapper | Reproducible Vortex benchmark-pack entrypoint for SIFT-style profiles. It delegates to `tools/vortex_benchmark_pack/cli.py`. |
| `tools/vortex_dataset_prepare.py` | public | Validated dataset materialization, validation, manifest writing, and `.fvecs`/`.ivecs` conversion. |
| `tools/vortex_selector_scale_validation.py` | maintenance | Bounded selector scale-policy validation across profiles and presets. |
| `tools/vortex_paper_artifact_bundle.py` | maintenance | Bundle reported benchmark manifests, commands, checksums, and report summaries without copying raw datasets. |

## Release And Maintenance Tools

| Tool | Tier | Purpose |
| --- | --- | --- |
| `tools/check_release_ready.py` | validation | Full readiness check using fresh build directories, schema checks, CTest, and install/export smoke. |
| `tools/check_source_package.py` | validation | Check package boundaries, required files, executable Git modes, ignored generated roots, and source documentation. |
| `tools/check_git_hygiene.py` | maintenance | Apply/check local Git config, path names, active refs, and sync-conflict Git metadata. |
| `tools/check_install_export.py` | package audit | Validate installed `find_package(LearnedHash CONFIG)` consumers. |
| `tools/audit_faiss_vendor.py` | maintenance | Audit the trimmed bundled CPU FAISS closure and attribution surface. |
| `tools/cleanup_local_artifacts.py` | maintenance | Dry-run-first local cleanup for ignored build/output/cache artifacts. |
| `tools/check_cleanup_local_artifacts.py` | maintenance regression | Regression tests for cleanup safety policy. |
| `tools/check_vortex_benchmark_pack_schema.py` | maintenance regression | Fixture-based benchmark-pack schema and compatibility check. |
| `tools/check_vortex_dataset_prepare.py` | maintenance regression | Dataset-preparation schema and safety checks. |
| `tools/check_vortex_selector_scale_validation.py` | maintenance regression | Scale-validation harness schema checks. |
| `tools/check_vortex_paper_artifact_bundle.py` | maintenance regression | Artifact-bundle schema checks. |

## Python Implementation Packages

`tools/vortex_benchmark_pack/` contains reusable implementation modules for the
benchmark pack. New first-party imports should use the focused modules directly,
for example `dominant_cost`, `metric_flattening`, and `selector_summary`.
`tools/vortex_benchmark_pack/metrics.py` remains as a compatibility re-export
for older external scripts, but new first-party code should not import symbols from
that wrapper.

`tools/vortex_dataset_prepare_lib/` contains the implementation modules behind
`tools/vortex_dataset_prepare.py`, split into dataset specs, path helpers,
binary validation, downloads, conversion, manifest writing, materialization,
and CLI routing. Keep `tools/vortex_dataset_prepare.py` as the package command
and import-compatible wrapper.

`tools/path_safety.py` checks path components used by generated-output tools and
the Git hygiene audit. Benchmark run names, scale-validation child run names,
and artifact-bundle names should pass through this helper before creating
directories.

## Additional Diagnostics

Experimental diagnostics may be useful while developing VortexHash, but they are
not part of the standard quickstart workflow.

- `hnsw_eval` is diagnostic support for HNSW index construction/inspection.
- Selector `--help-advanced` flags are diagnostic and compatibility-oriented;
  normal users should start with `--help` and the benchmark-pack presets.
