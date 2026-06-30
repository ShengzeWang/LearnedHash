# Vortex Dataset Source List

This file lists official sources and materialization workflows for Vortex
benchmark datasets. It does not include raw vectors, generated models, or local
benchmark logs. The benchmark pack exposes `siftsmall`, `sift`, and
manifest-gated benchmark profiles for GIST1M and the 10M workloads listed below.

## Source Policy

- All current Vortex vector workloads are L2 k-NN workloads. TexMex
  ground-truth files store squared Euclidean distances, which preserve the same
  ordering as L2 distance; Vortex reports the workload metric as `L2`.
- Use canonical base/query splits when the source provides them. For TexMex,
  that means the official `*_base.fvecs`, `*_query.fvecs`, `*_learn.fvecs`, and
  `*_groundtruth.ivecs` files from the source archive. For BigANN-style 10M
  profiles, that means the official public query set paired with the canonical
  prefix of the public 1B-scale base file.
- Use official original sources whenever available. Do not use Kaggle,
  HuggingFace mirrors, blog mirrors, or third-party repackaging for benchmark
  results unless the official source is unavailable and the deviation is recorded
  in the local manifest.
- SIFT-Small is a test/smoke dataset only, not a reported benchmark dataset.
- Treat SIFT1M as the standard SIFT reported benchmark baseline profile.
- For 10M-scale benchmark datasets derived from 1B/larger corpora, use canonical
  prefix slices: the first 10,000,000 base vectors in original storage order.
  This matches the BigANN framework's cropped-file workflow and preserves
  compatibility with scale-specific official 10M ground truth.
- Never random-sample a 10M reported dataset when using official 10M ground truth.
  Random sampling changes nearest-neighbor identities and requires a separate
  exact ground-truth computation.
- Record checksums, byte sizes, vector count, dimension, dtype, query count,
  ground-truth source, conversion command, and git commit in each local run
  manifest before reporting numbers.
- Use official scale-specific 10M ground truth for canonical prefix slices.
  Compute exact L2 ground truth only for non-canonical samples or if an official
  10M file becomes unavailable.
- For slice-based workloads, ground truth must be exact over the evaluated
  slice. A full-1B ground-truth file, random-sample ground truth, or
  source-family-mismatched ground truth is invalid for a 10M prefix run.

## Evaluation Dataset Roles

| Dataset | Role | Size | Dim. | Queries | Metric | Benchmark source | Practical source |
|---|---:|---:|---:|---:|---|---|---|
| SIFT-Small | test/smoke only | 10K base | 128 | 100 | L2 | TexMex | TexMex `siftsmall.tar.gz` |
| SIFT1M | benchmark baseline | 1M base | 128 | 10K | L2 | TexMex / ANN-Benchmarks naming | TexMex `sift.tar.gz` |
| GIST1M | high-dimensional benchmark baseline | 1M base | 960 | 1K | L2 | TexMex / ANN-Benchmarks naming | TexMex `gist.tar.gz` |
| SIFT10M | 10M descriptor stress test | first 10M from SIFT1B/BIGANN | 128 | 10K | L2 | TexMex SIFT1B and BigANN | BigANN `base.1B.u8bin` byte-range prefix + `GT_10M/bigann-10M` |
| DEEP10M-L2 | neural-embedding stress test | first 10M from DEEP1B | 96 | 10K | L2 | Yandex DEEP1B and BigANN | Yandex `base.1B.fbin` or `base.10M.fbin` + `GT_10M/deep-10M` |
| SPACEV10M | production-style web-search embeddings | first 10M from SPACEV1B | 100 | 29,316 | L2 | Microsoft SPACEV1B and BigANN | SPACEV `spacev1b_base.i8bin` byte-range prefix + `msspacev-gt-10M` |

## Official Source URLs

### TexMex / INRIA-IRISA

TexMex documents the ANN datasets, file formats, dimensions, base/query/learn
counts, and ground-truth convention for SIFT-Small, SIFT1M, GIST1M, and SIFT1B.
The same page states that SIFT1B subset ground truth is defined over the first
N vectors of `bigann_base.bvecs`.

- Overview: http://corpus-texmex.irisa.fr/
- MD5 sums: ftp://ftp.irisa.fr/local/texmex/corpus/MD5SUM
- SIFT-Small: ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz
- SIFT1M: ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
- GIST1M: ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz
- SIFT1B base: ftp://ftp.irisa.fr/local/texmex/corpus/bigann_base.bvecs.gz
- SIFT1B query: ftp://ftp.irisa.fr/local/texmex/corpus/bigann_query.bvecs.gz
- SIFT1B ground truth: ftp://ftp.irisa.fr/local/texmex/corpus/bigann_gnd.tar.gz
- Official MD5 values used by this workflow:
  - `siftsmall.tar.gz`: `0b8324a7a82d7f2663d7dcbd57642df7`
  - `sift.tar.gz`: `b23d1b3b2ee8469d819b61ca900ef0ed`
  - `gist.tar.gz`: `31185e0f00854f74d27e8ad8d52628a9`
  - `bigann_base.bvecs.gz`: `4346fa68813d71e7f3d75d2faba669ec`
  - `bigann_query.bvecs.gz`: `b5f61050d3c435911d54db122b5469b5`
  - `bigann_gnd.tar.gz`: `f40646ba10e152229818e82cd5bd84b2`
- ANN-Benchmarks convenience HDF5 files may be used for local debugging only
  when the manifest records them as a packaging convenience, not as benchmark
  source metadata:
  - https://ann-benchmarks.com/sift-128-euclidean.hdf5
  - https://ann-benchmarks.com/gist-960-euclidean.hdf5

### BigANN Benchmark Format and 10M Slices

The BigANN benchmark page documents the common `.fbin`, `.u8bin`, and `.i8bin`
binary format, public query files, official 10M/100M/1B ground-truth bundles,
and recommended download tools for the large corpora.

- Dataset page: https://big-ann-benchmarks.com/neurips21.html
- 10M ground-truth bundle: https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/GT_10M_v2.tgz
- 100M ground-truth bundle: https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/GT_100M_v2.tgz
- 1B ground-truth bundle: https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/GT_1B_v2.tgz
- BIGANN/SIFT1B base: https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/base.1B.u8bin
- BIGANN/SIFT public queries: https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/query.public.10K.u8bin
- BIGANN/SIFT10M official GT: https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/GT_10M/bigann-10M
- DEEP1B base: https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/base.1B.fbin
- DEEP10M convenience base: https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/base.10M.fbin
- DEEP public queries: https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/query.public.10K.fbin
- DEEP10M official BigANN GT: https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/GT_10M/deep-10M
- DEEP1B full-base public GT: https://storage.yandexcloud.net/yandex-research/ann-datasets/deep_new_groundtruth.public.10K.bin
- SPACEV1B base: https://comp21storage.z5.web.core.windows.net/comp21/spacev1b/spacev1b_base.i8bin
- SPACEV100M sample: https://comp21storage.z5.web.core.windows.net/comp21/spacev1b/spacev100m_base.i8bin
- SPACEV public queries: https://comp21storage.z5.web.core.windows.net/comp21/spacev1b/query.i8bin
- SPACEV10M official GT: https://comp21storage.z5.web.core.windows.net/comp21/spacev1b/msspacev-gt-10M
- SPACEV public full-scale GT: https://comp21storage.z5.web.core.windows.net/comp21/spacev1b/public_query_gt100.bin

### Microsoft SPACEV1B

Microsoft's SPTAG repository documents SPACEV1B as a Bing web vector-search
dataset with 1,402,020,720 100-dimensional int8 document descriptors, 29,316
100-dimensional int8 query descriptors, and top-100 L2 ground truth.

- Dataset README: https://github.com/microsoft/SPTAG/tree/main/datasets/SPACEV1B
- License: https://github.com/microsoft/SPTAG/blob/main/datasets/SPACEV1B/LICENSE

## Base/Query/Ground-Truth Match Contract

The 10M pipeline is source-family locked. A benchmark profile is valid only when
the base, query, and ground-truth files all come from the same dataset spec in
`tools/vortex_dataset_prepare.py`.

- `sift10m`: use BigANN common-format `base.1B.u8bin`,
  `query.public.10K.u8bin`, and `GT_10M/bigann-10M`.
- `deep10m_l2`: use Yandex/BigANN `base.1B.fbin` or the byte-identical
  official `base.10M.fbin` prefix, `query.public.10K.fbin`, and
  `GT_10M/deep-10M`.
- `spacev10m`: use the BigANN competition `spacev1b_base.i8bin` 1B file,
  `query.i8bin`, and `msspacev-gt-10M`. Do not pair the 10M BigANN ground truth
  with the original SPTAG `vectors.bin` shards unless a separate validator
  proves identical ordering and ID space.

This rule is stricter than merely matching dimensions. It prevents mistakes
such as using full-1B ground truth with a 10M base, using a debug 10M file with
a different vector order, or mixing SIFT/DEEP/SPACEV query sets that happen to
share a query count.

## Canonical 10M Prefix Slices

For SIFT10M, DEEP10M-L2, and SPACEV10M, the canonical 10M database is:

```text
D_10M = x_0, x_1, ..., x_9,999,999
```

where `x_i` is the vector at position `i` in the public 1B-scale benchmark base
file. The query set is not sliced. Use the official scale-specific 10M ground
truth listed above.

BigANN common binary files start with an 8-byte little-endian header:

```text
uint32 num_vectors
uint32 dim
raw vector payload
```

The 10M prefix byte ranges are therefore:

| Dataset | dtype | dim | Expected prefix file size | HTTP byte range |
|---|---:|---:|---:|---:|
| SIFT10M | uint8 | 128 | 1,280,000,008 bytes | `0-1280000007` |
| DEEP10M-L2 | float32 | 96 | 3,840,000,008 bytes | `0-3840000007` |
| SPACEV10M | int8 | 100 | 1,000,000,008 bytes | `0-1000000007` |

After a byte-range download, patch only the first header integer to
`10,000,000`; the dimension must already match. This mirrors the BigANN
framework behavior for cropped datasets: it downloads `8 + d * n * sizeof(dtype)`
bytes and then rewrites `num_vectors` in the header.

Example range-download commands:

```bash
tools/vortex_dataset_prepare.py range-commands --dataset sift10m
tools/vortex_dataset_prepare.py range-commands --dataset deep10m_l2
tools/vortex_dataset_prepare.py range-commands --dataset spacev10m
```

## Canonical Local Layout

Raw payloads stay out of git. Use this layout for local experiments:

```text
vector_datasets/
  _sources/
    texmex/
    bigann/
    deep1b/
    spacev1b/
  siftsmall/
    siftsmall_base.fvecs
    siftsmall_query.fvecs
    siftsmall_learn.fvecs
    siftsmall_groundtruth.ivecs
    manifest.json
  sift/
    sift_base.fvecs
    sift_query.fvecs
    sift_learn.fvecs
    sift_groundtruth.ivecs
    manifest.json
  gist1m/
    gist_base.fvecs
    gist_query.fvecs
    gist_learn.fvecs
    gist_groundtruth.ivecs
    manifest.json
  sift10m/
    base.10M.u8bin
    query.public.10K.u8bin
    bigann-10M
    sift10m_base.fvecs
    sift10m_query.fvecs
    sift10m_groundtruth.ivecs
    manifest.json
  deep10m_l2/
    base.10M.fbin
    query.public.10K.fbin
    deep-10M
    deep10m_l2_base.fvecs
    deep10m_l2_query.fvecs
    deep10m_l2_groundtruth.ivecs
    manifest.json
  spacev10m/
    base.10M.i8bin
    query.i8bin
    msspacev-gt-10M
    spacev10m_base.fvecs
    spacev10m_query.fvecs
    spacev10m_groundtruth.ivecs
    manifest.json
```

The 10M materializer produces validated raw `.u8bin`, `.i8bin`, and `.fbin`
working files first. The conversion step then stream-converts those validated files to Vortex
`.fvecs`/`.ivecs`. `uint8` and `int8` components are cast to `float32` without
normalization, and `float32` components are preserved. L2 comparisons must use
the same casted values used by Vortex training/evaluation.

## TexMex Materialization

The TexMex materializer handles SIFT-Small, SIFT1M, and GIST1M directly from the
official TexMex archives. It downloads with resume support, verifies the
published MD5 and exact archive byte size, selectively extracts only the
expected files, validates `.fvecs`/`.ivecs` dimensions and ground-truth ID
bounds, and writes a local `manifest.json`.

```bash
# List supported TexMex archive profiles.
tools/vortex_dataset_prepare.py texmex-list

# Download/extract/validate and write a manifest with output checksums.
tools/vortex_dataset_prepare.py texmex-materialize \
  --dataset sift \
  --include-file-sha256

# Re-run validation without downloading or extracting.
tools/vortex_dataset_prepare.py texmex-validate --dataset gist1m
```

Expected local manifests produced by the TexMex workflow:

- `vector_datasets/siftsmall/manifest.json`
- `vector_datasets/sift/manifest.json`
- `vector_datasets/gist1m/manifest.json`

## 10M Prefix Materialization

The 10M materializer downloads only the first-10M byte range from the official
1B-scale base file, stores the unmodified prefix under
`vector_datasets/_sources/<family>/`, copies it into the workload directory,
patches only the materialized copy's header to `10,000,000`, downloads the
official public query set and scale-specific 10M ground truth, validates the
triple, and writes `manifest.json`.

```bash
# Download/crop/patch/validate SIFT10M and record checksums.
tools/vortex_dataset_prepare.py materialize \
  --dataset sift10m \
  --include-file-sha256

# Run a validated exact L2 spot check over fixed-seed queries.
tools/vortex_dataset_prepare.py validate-materialized \
  --dataset sift10m \
  --spot-check-queries 32 \
  --spot-check-seed 20260101
```

Use `--force-download` only to replace a corrupt or source-mismatched file. The
default path is resumable and refuses to append a response if the server ignores
the requested byte range.

## Vortex Format Conversion

The converter runs only after the raw materialized triple passes validation.
It streams base/query vectors from `.u8bin`, `.i8bin`, or `.fbin` into Vortex
`.fvecs`, extracts ground-truth IDs into `.ivecs`, validates output row headers
and ID bounds, and updates `manifest.json` with dtype casts, output byte sizes,
and optional SHA-256 checksums.

```bash
# Convert a validated materialized 10M workload into Vortex-readable files.
tools/vortex_dataset_prepare.py convert \
  --dataset sift10m \
  --include-file-sha256

# Re-run conversion-output validation without touching raw files.
tools/vortex_dataset_prepare.py validate-converted \
  --dataset sift10m
```

Use `--force-convert` only to replace a corrupt converted file. The converter
refuses to overwrite an existing output with an unexpected byte size unless that
flag is present.

## Reproducible Materialization Workflow

1. Download only from official URLs above. Use resumable tools (`axel` for
   BigANN/Yandex and `azcopy` for Microsoft blobs when available).
2. Store raw archives under `vector_datasets/_sources/<family>/`.
3. Verify official MD5/SHA checksums when published. If no official checksum is
   published, compute and record local SHA-256 after first successful download.
4. Parse headers before conversion and fail if vector count, dimension, dtype,
   or file size disagrees with this source list.
5. For each 10M dataset, stream the first 10,000,000 base vectors in original
   file order; do not shuffle, stratify, or sample randomly.
6. Patch cropped xbin headers to the selected prefix count and verify exact
   byte size before conversion.
7. Run `tools/vortex_dataset_prepare.py convert --dataset <profile>` to produce
   Vortex `.fvecs`/`.ivecs` outputs and update the manifest with source URL,
   source checksum, output checksum, count, dimension, dtype conversion, and
   command line.
8. Use official first-10M ground truth when it exactly matches the selected
   prefix slice. Otherwise compute exact top-k L2 in chunks and record the exact
   command, chunk size, thread count, and output checksum.
9. Run a lightweight validator that checks vector counts, dimensions, query
   count, ground-truth ID bounds, monotonic distance order when distances are
   available, and a deterministic exact-search spot check over 32-128 fixed-seed
   queries. Use tie-aware comparison.
10. Register the dataset profile in the benchmark pack only after the validator
    passes.
11. Include source and manifest checksums in benchmark artifact metadata and final
    benchmark reports.

## Validation Helper

Use `tools/vortex_dataset_prepare.py` before registering any 10M profile in the
benchmark pack. The helper encodes the canonical first-10M prefix specs and
reports blockers when the base, query, and ground-truth files do not match.

```bash
# Verify the repository-level workload contract: L2 metric, canonical splits,
# and exact ground truth over the evaluated base/slice.
tools/vortex_dataset_prepare.py contract-check

# Show the official URLs, byte range, expected dimensions, and query/GT counts.
tools/vortex_dataset_prepare.py list

# Print deterministic range-download commands for a 10M workload.
tools/vortex_dataset_prepare.py range-commands --dataset sift10m

# Materialize the canonical first-10M workload triple and write manifest.json.
tools/vortex_dataset_prepare.py materialize \
  --dataset sift10m \
  --include-file-sha256

# Patch the cropped xbin header from the original 1B count to 10,000,000.
tools/vortex_dataset_prepare.py patch-xbin-header \
  --dataset sift10m \
  --base vector_datasets/_sources/bigann/base.10M.u8bin

# Validate that base, query, and GT belong to the same benchmark slice.
tools/vortex_dataset_prepare.py validate \
  --dataset sift10m \
  --base vector_datasets/sift10m/base.10M.u8bin \
  --query vector_datasets/sift10m/query.public.10K.u8bin \
  --groundtruth vector_datasets/sift10m/bigann-10M

# Validate the materialized local layout and run tie-aware exact L2 spot checks.
tools/vortex_dataset_prepare.py validate-materialized \
  --dataset sift10m \
  --spot-check-queries 32

# Convert validated raw files into Vortex .fvecs/.ivecs and record checksums.
tools/vortex_dataset_prepare.py convert \
  --dataset sift10m \
  --include-file-sha256

# Validate converted row headers, sizes, and ground-truth ID bounds.
tools/vortex_dataset_prepare.py validate-converted \
  --dataset sift10m
```

The validation checks are intentionally stricter than a simple file-exists
check: exact base/query headers, exact byte sizes, query count and dimension,
ground-truth query count and top-k, ground-truth ID bounds inside the 10M base,
and nondecreasing ground-truth distances. This prevents accidental mixing such
as using a SIFT query file with DEEP vectors, using full-1B ground truth with a
10M prefix, or evaluating a random 10M sample against prefix-slice truth.


## Minimum Manifest Fields

Every reported benchmark dataset should have a local manifest next to the materialized
files. At minimum:

```yaml
dataset: SIFT10M
role: reported_benchmark
source_family: bigann
base_url: https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/base.1B.u8bin
query_url: https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/query.public.10K.u8bin
groundtruth_url: https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/GT_10M/bigann-10M
slice_rule: first_10000000_vectors
byte_range: 0-1280000007
dtype: uint8
dim: 128
metric: L2
query_count: 10000
groundtruth_topk: 100
base_sha256: "<computed>"
query_sha256: "<computed>"
groundtruth_sha256: "<computed>"
materializer_git_commit: "<git rev-parse HEAD>"
download_date_utc: "<YYYY-MM-DD>"
conversion: uint8_to_float32_fvecs_no_normalization
validator: header_size_id_bounds_distance_order_exact_spot_check
```

## Dataset Readiness Workflow

A dataset profile is ready for reporting only after all of these conditions are true:

1. The official source URLs, counts, dimensions, dtype, metric, query count, and
   ground-truth top-k are listed in this file.
2. The local materializer has written a manifest next to the materialized files.
3. Raw base/query/ground-truth files pass header, byte-size, ID-bound, and
   distance-order validation.
4. For canonical 10M prefix slices, fixed-seed exact L2 spot checks pass with
   tie-aware comparison against the official scale-specific 10M ground truth.
5. Vortex `.fvecs`/`.ivecs` conversion has completed from the validated raw
   triple and output validation passes.
6. The benchmark-pack profile preflight passes and records the manifest path,
   manifest checksum, source family, split contract, and source-file checksums.
7. Selector scale-validation has been run for the target benchmark profile and the
   generated-code/materialized evaluation agrees with selector-reported recall
   and overlay-locality metrics.
8. A benchmark artifact bundle has been created from the final scale-validation run
   without copying raw datasets or generated model/cache payloads.

Use these commands as the normal execution order:

```bash
# TexMex profiles: SIFT-Small, SIFT1M, GIST1M.
tools/vortex_dataset_prepare.py texmex-materialize --dataset sift --include-file-sha256
tools/vortex_dataset_prepare.py texmex-validate --dataset sift

# Canonical 10M prefix profiles: SIFT10M, DEEP10M-L2, SPACEV10M.
tools/vortex_dataset_prepare.py materialize --dataset sift10m --include-file-sha256
tools/vortex_dataset_prepare.py validate-materialized --dataset sift10m --spot-check-queries 32
tools/vortex_dataset_prepare.py convert --dataset sift10m --include-file-sha256
tools/vortex_dataset_prepare.py validate-converted --dataset sift10m

# Benchmark and benchmark-artifact gates.
tools/vortex_sift_benchmark_pack.py --profile sift10m --preset smoke --check-profile-only
tools/vortex_selector_scale_validation.py --profiles sift,sift10m --presets smoke,baseline,deep --skip-build
tools/vortex_paper_artifact_bundle.py \
  --scale-validation-dir vortex_v1_output/selector_scale_validation/<run> \
  --bundle-name <benchmark-run-name> \
  --copy-reports
```

For committed release candidates, rerun the artifact bundle with
`--require-clean-git`.
