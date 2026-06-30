# VortexHash

`hash_functions_core/Vortex/` contains VortexHash, the vector learned hash
function used by LearnedHash. VortexHash trains a compact model that maps vectors
to an ordered hash space while preserving locality for downstream routing,
placement, and evaluation workflows.

The source directory is named `Vortex`. The public C++ namespace, installed
headers, CMake targets, command-line binary names, and default generated-output
root keep the `vortex_v1` name for compatibility.

## What Is Included

- Training and model serialization for VortexHash v1.
- HNSW/NSW skeleton extraction and model-selection tools.
- Generated-code emission for standalone inference packages.
- Runtime loading, hash inference, and centroid-neighbor queries.
- Evaluation tools for locality, recall, latency, and HNSW baselines.
- Unit and smoke tests for runtime, training, selector, codegen, and CLI paths.

## Build

From the repository root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

The default build enables FAISS-backed Vortex training, selector, and evaluation
components. For runtime/codegen-only consumers that do not need FAISS-backed
training or evaluation, use a lean build:

```bash
cmake -S . -B build_lean -DCMAKE_BUILD_TYPE=Release \
  -DLEARNEDHASH_BUILD_TOOLS=OFF \
  -DLEARNEDHASH_BUILD_VORTEX_FAISS_COMPONENTS=OFF
cmake --build build_lean --config Release
```

FAISS provider modes are documented in `docs/public/faiss_provider.md`.

## Tools

Default full builds place VortexHash binaries under
`build/hash_functions_core/Vortex/`:

- `vortex_v1_cli`: train models, extract NSW skeletons, hash vectors, and emit generated-code packages.
- `vortex_model_selector`: search dataset-aware training configurations.
- `vortex_eval`: evaluate generated-code locality, recall, and latency.
- `hnsw_eval`: build or inspect FAISS HNSW baselines.

Check the public help text after building:

```bash
build/hash_functions_core/Vortex/vortex_v1_cli --help
build/hash_functions_core/Vortex/vortex_model_selector --help
build/hash_functions_core/Vortex/vortex_model_selector --help-advanced
```

Use `LEARNEDHASH_LOG=trace|debug|info|warn|error|off` for diagnostics.

## Minimal Flow

The shortest VortexHash smoke path uses SIFT-Small files materialized under
`vector_datasets/siftsmall/`:

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
  --seed 42 --threads 1 \
  --output vortex_v1_output/siftsmall_v1.model \
  --codegen --codegen-dir vortex_v1_output/siftsmall_codegen \
  --codegen-name siftsmall_v1

build/hash_functions_core/Vortex/vortex_v1_cli hash \
  --codegen-dir vortex_v1_output/siftsmall_codegen \
  --dataset vector_datasets/siftsmall/siftsmall_query.fvecs \
  --out vortex_v1_output/siftsmall_hashes.txt
```

Generated outputs are ignored by git and default to `vortex_v1_output/` unless
an explicit output directory is provided.

## Public API Surface

Installed headers keep the compatibility include prefix:

```cpp
#include <vortex_v1/model.h>
#include <vortex_v1/codegen_loader.h>
```

CMake consumers can link the exported targets:

```cmake
find_package(LearnedHash CONFIG REQUIRED)
target_link_libraries(app PRIVATE LearnedHash::vortex_v1_runtime)
```

Full builds also export `LearnedHash::vortex_v1_core`. Lean builds export only
`LearnedHash::vortex_v1_runtime` for runtime/codegen consumers.

## Validation

Focused Vortex checks from the repository root:

```bash
ctest --test-dir build --output-on-failure -R vortex
```

Full repository release readiness:

```bash
tools/check_release_ready.py --jobs 8
```

For install/export smoke tests after a build:

```bash
tools/check_install_export.py --build-dir build
```

## More Documentation

- `docs/public/quickstart.md`: build, install, and minimal examples.
- `docs/public/vortex_model_selector.md`: selector options, roles, and metrics.
- `docs/public/vortex_benchmark_pack.md`: reproducible benchmark-pack workflow.
- `docs/public/vortex_generated_code_api.md`: generated C++ and C ABI usage.
- `docs/public/tool_reference.md`: command-line tool reference.

## License

Apache-2.0
