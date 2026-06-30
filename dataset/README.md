# Dataset Metadata

This directory contains dataset provenance and layout metadata for LearnedHash
workflows. Raw vector payloads, generated models, benchmark logs, and temporary
downloads are intentionally not tracked in git.

## Files

- `SOURCES.md`: upstream dataset sources, expected local layouts, and
  reproducible materialization notes.

## Local Data Roots

Dataset payloads are expected under ignored local roots such as
`vector_datasets/<profile>/`. Use the dataset preparation tool to inspect and
materialize supported profiles:

```bash
tools/vortex_dataset_prepare.py profiles --json
```

Supported VortexHash profiles include SIFT-Small for fast smoke tests and larger
SIFT/GIST/DEEP/SPACEV profiles for benchmark workflows when the corresponding
local payloads are available.

## Reporting Artifacts

Raw datasets and generated model/cache payloads should stay outside source
packages. When benchmark evidence needs to be shared, use the artifact bundler
on a validated scale-validation directory:

```bash
tools/vortex_paper_artifact_bundle.py \
  --scale-validation-dir vortex_v1_output/selector_scale_validation/<run> \
  --bundle-name <benchmark-run-name> \
  --copy-reports
```
