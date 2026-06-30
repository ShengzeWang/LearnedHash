# LearnedHash Quickstart

This guide is the shortest public path from a fresh checkout to a verified build
and basic CLI usage.

## Requirements

- CMake 3.16+
- C++17 compiler
- POSIX threads
- OpenMP recommended for FAISS-backed Vortex training/evaluation
- A trimmed CPU FAISS snapshot bundled by default under `include/faiss`, or an
  installed FAISS CMake package when using
  `-DLEARNEDHASH_FAISS_PROVIDER=system`

## Build And Test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build --output-on-failure -j8
```

With CMake 3.21 or newer, the same Release workflow is available through
project presets:

```bash
cmake --preset release-bundled
cmake --build --preset release-bundled
ctest --preset release-bundled
```

For library-only builds:

```bash
cmake -S . -B build_lib -DCMAKE_BUILD_TYPE=Release \
  -DLEARNEDHASH_BUILD_TOOLS=OFF
cmake --build build_lib --config Release
```

Preset equivalent:

```bash
cmake --preset release-libonly
cmake --build --preset release-libonly
ctest --preset release-libonly
```

For a lean runtime/codegen package that does not configure or export FAISS:

```bash
cmake -S . -B build_lean -DCMAKE_BUILD_TYPE=Release \
  -DLEARNEDHASH_BUILD_TOOLS=OFF \
  -DLEARNEDHASH_BUILD_VORTEX_FAISS_COMPONENTS=OFF
cmake --build build_lean --config Release
ctest --test-dir build_lean --output-on-failure
```

Preset equivalent:

```bash
cmake --preset release-lean-runtime
cmake --build --preset release-lean-runtime
ctest --preset release-lean-runtime
```

Install the built package and command-line tools:

```bash
cmake --install build --prefix /tmp/learnedhash-install
```

Downstream CMake projects can consume the installed package with:

```cmake
find_package(LearnedHash CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE LearnedHash::vortex_v1_core)
```

Lean runtime builds export `LearnedHash::vortex_v1_runtime` instead of
`LearnedHash::vortex_v1_core`.

To verify the installed package from an existing build:

```bash
tools/check_install_export.py --build-dir build
```

If Python was found during CMake configure, the same check is also available as
a build target:

```bash
cmake --build build --target learnedhash_check_install_export
```

Or through the preset build directory:

```bash
cmake --build --preset release-bundled-install-export
```

For a clean readiness pass using fresh build directories:

```bash
tools/check_release_ready.py --jobs 8
```

To keep a machine-readable evidence file:

```bash
tools/check_release_ready.py --jobs 8 --report-json /tmp/learnedhash-readiness.json
```

After committing a release candidate, add `--strict-source-archive` to fail if
`git archive HEAD` would miss a required source file:

```bash
tools/check_release_ready.py --jobs 8 --strict-source-archive
```

For focused checks, use CTest labels:

```bash
ctest --test-dir build -L rm_model --output-on-failure
ctest --test-dir build -L lead --output-on-failure
ctest --test-dir build -L vortex --output-on-failure
ctest --test-dir build -L codegen --output-on-failure
```

## Runtime Logging

Training, selector, and generated-model loading paths share the rm_model logging
layer. Use `LEARNEDHASH_LOG=trace|debug|info|warn|error|off` for diagnostics.
`RUST_LOG` remains a compatibility fallback for older scripts, but
`LEARNEDHASH_LOG` wins when both variables are present.

## Primary Tools

- `build/learned_structures/rm_model/rm_model_learner`: train scalar learned models.
- `build/learned_structures/rm_model/rm_model_inferencer`: run scalar model inference.
- `build/hash_functions_core/LEAD/lead_hashgen`: train and emit LEAD hash artifacts.
- `build/hash_functions_core/LEAD/lead_hasher`: hash scalar keys or datasets.
- `build/hash_functions_core/Vortex/vortex_v1_cli`: VortexHash v1 build/train/hash/codegen pipeline.
- `build/hash_functions_core/Vortex/vortex_model_selector`: dataset-aware VortexHash config search.
- `build/hash_functions_core/Vortex/vortex_eval`: hash locality and latency evaluation.
- `build/hash_functions_core/Vortex/hnsw_eval`: pure HNSW baseline evaluation.

## Minimal LEAD Smoke

```bash
python3 - <<'PY'
import struct
vals = [i * 3 for i in range(1024)]
with open('/tmp/lh_small_uint64', 'wb') as f:
    f.write(struct.pack('<Q', len(vals)))
    for v in vals:
        f.write(struct.pack('<Q', v))
PY

build/hash_functions_core/LEAD/lead_hashgen \
  /tmp/lh_small_uint64 smoke_lead linear,linear 8 \
  --output-dir /tmp/lead_smoke

build/hash_functions_core/LEAD/lead_hasher \
  --model-dir /tmp/lead_smoke --key 123 --key-type u64 --csv
```

## Minimal VortexHash Flow

This assumes SIFT-Small files exist under `vector_datasets/siftsmall/`.

```bash
mkdir -p vortex_v1_output

build/hash_functions_core/Vortex/vortex_v1_cli build_hnsw \
  --dataset vector_datasets/siftsmall/siftsmall_base.fvecs \
  --M 16 --efConstruction 100 --metric L2 \
  --output vortex_v1_output/siftsmall_hnsw.index

build/hash_functions_core/Vortex/vortex_v1_cli extract_nsw \
  --index vortex_v1_output/siftsmall_hnsw.index \
  --target_skeleton 5000 \
  --output vortex_v1_output/siftsmall_nsw.csr

build/hash_functions_core/Vortex/vortex_v1_cli train \
  --dataset vector_datasets/siftsmall/siftsmall_base.fvecs \
  --nsw vortex_v1_output/siftsmall_nsw.csr \
  --K 256 --hash_bits 64 --cdf_models linear,linear \
  --cdf_branch 32 --centroid_knn 32 \
  --output vortex_v1_output/siftsmall_v1.model \
  --codegen --codegen-dir vortex_v1_output/siftsmall_codegen \
  --codegen-name siftsmall_v1

build/hash_functions_core/Vortex/vortex_v1_cli hash \
  --codegen-dir vortex_v1_output/siftsmall_codegen \
  --dataset vector_datasets/siftsmall/siftsmall_query.fvecs \
  --out vortex_v1_output/siftsmall_hashes.txt
```

## More Public Guides

- `docs/public/faiss_provider.md`
- `docs/public/vortex_model_selector.md`
- `docs/public/vortex_benchmark_pack.md`
- `docs/public/vortex_generated_code_api.md`
- Component references under `learned_structures/` and `hash_functions_core/`
