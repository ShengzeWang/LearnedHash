# LEAD

LEAD is a scalar learned hash function built on top of `rm_model`. It maps
recursive-model predictions into a configurable hash space while preserving key
order.

## Build

From repository root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

`lead_hashgen` and `lead_hasher` are command-line tools built by default. For
library-only builds, pass `-DLEARNEDHASH_BUILD_TOOLS=OFF`.

LEAD tools use the shared rm_model logging layer. Prefer
`LEARNEDHASH_LOG=trace|debug|info|warn|error|off` for diagnostics. `RUST_LOG`
remains a compatibility fallback, and `LEARNEDHASH_LOG` takes precedence when
both are set.

## Validation

From the repository root, use the standard build and CTest flow:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build --output-on-failure -R lead
```

For package-level checks, run the top-level readiness check:

```bash
tools/check_release_ready.py --jobs 8
```

The LEAD test coverage exercises runtime hashing, generated rm_model loading,
UInt256 edge behavior, strict CLI parsing, and library-only builds.

## End-to-End Example

This flow trains a learned hash function from a binary dataset, generates artifacts, and hashes keys.

1) Train + hashgen:

```bash
build/hash_functions_core/LEAD/lead_hashgen dataset/osm_cellids_200M_uint64 \
  my_lead linear,linear 100 --space-bits 64 --scale-factor 1.0
```

2) Hash a single key:

```bash
build/hash_functions_core/LEAD/lead_hasher --model-dir lead_output/my_lead_linear_linear_bf100 \
  --key 12345
```

3) Hash a dataset to CSV (first 1M rows; CSV is the default for datasets):

```bash
build/hash_functions_core/LEAD/lead_hasher --model-dir lead_output/my_lead_linear_linear_bf100 \
  --dataset dataset/osm_cellids_200M_uint64 --limit 1000000 --out hashes.csv
```

4) Hash a dataset to raw binary:

```bash
build/hash_functions_core/LEAD/lead_hasher --model-dir lead_output/my_lead_linear_linear_bf100 \
  --dataset dataset/osm_cellids_200M_uint64 --format raw --out hashes.bin
```

## Quick Validation

Minimal reproducible smoke check from repository root:

```bash
python3 - <<'PY'
import struct
vals=[i*3 for i in range(1024)]
with open('/tmp/lh_small_uint64','wb') as f:
    f.write(struct.pack('<Q',len(vals)))
    for v in vals:
        f.write(struct.pack('<Q',v))
PY

build/hash_functions_core/LEAD/lead_hashgen /tmp/lh_small_uint64 smoke_lead linear,linear 8 --output-dir /tmp/lead_smoke
build/hash_functions_core/LEAD/lead_hasher --model-dir /tmp/lead_smoke --key 123 --key-type u64 --csv
```

## Data Format

The dataset format matches `rm_model`:
- first 8 bytes: `uint64_t` count in little-endian
- payload: `count` items in little-endian (`uint64_t`, `uint32_t`, or `double`)

The dataset must be sorted in ascending key order (rm_model requirement). Duplicates are allowed.

Type inference is based on the input path:
- contains `uint64` -> `uint64_t`
- contains `uint32` -> `uint32_t`
- contains `f64` -> `double`

## Hashgen

```bash
build/hash_functions_core/LEAD/lead_hashgen <input> [namespace] [models] [branching factor] [options]
```

Selection modes:
- Explicit models: provide `models` and `branching factor` positionally.
- Size bounded: use `--max-size <bytes>` (cannot be combined with explicit models).
- Auto: omit models/branching factor to run the selector and choose a balanced config.

Key options:
- `--output-dir <dir>`: output directory (default `lead_output/<namespace>_<models>_bf<branch>`).
- `--data-path <dir>`: output directory for rm_model parameter blobs.
- `--max-size <bytes>`: choose a configuration smaller than the specified size.
- `--size-weight <w>` and `--error-weight <w>`: selector weights (non-negative).
- `--selector-limit <n>`: number of candidate configs to evaluate (>=2).
- `--space-bits <bits>`: hash space size in [1, 256] (default 64).
- `--scale-factor <f>`: scale factor in (0, 1], default 1.0 (clamped).
- `--threads <count>`: training threads (default 4).
- `--no-code`: skip emitting model artifacts (output is not loadable).
- `--no-errors`: omit last-layer error metadata in generated code.
- `--no-calibration`: disable prediction calibration (enabled by default).
- `--zero-build-time`: zero `BUILD_TIME_NS` for reproducible outputs.

Examples:

Train with an explicit output directory and custom data path:

```bash
build/hash_functions_core/LEAD/lead_hashgen dataset/osm_cellids_200M_uint64 \
  my_lead linear,linear 100 --output-dir /tmp/my_lead_out --data-path /tmp/my_lead_data
```

Train with selector weights and a larger candidate pool:

```bash
build/hash_functions_core/LEAD/lead_hashgen dataset/osm_cellids_200M_uint64 \
  my_lead --size-weight 0.7 --error-weight 0.3 --selector-limit 20
```

Train with a size budget and pinned thread count:

```bash
build/hash_functions_core/LEAD/lead_hashgen dataset/osm_cellids_200M_uint64 \
  my_lead_auto --max-size 65536 --threads 8
```

Train with a smaller hash space and scale factor:

```bash
build/hash_functions_core/LEAD/lead_hashgen dataset/osm_cellids_200M_uint64 \
  my_lead linear,linear 100 --space-bits 32 --scale-factor 0.9
```

## Hasher

```bash
build/hash_functions_core/LEAD/lead_hasher --model-dir <lead_output_dir> \
  (--dataset <path> | --key <value>) [options]
```

Options:
- `--key-type <u64|u32|f64>`: optional, must match the model key type.
- `--format <hex|raw>`: output raw binary (big-endian) or text (CSV for datasets, hex for single-key).
- `--out <file>`: write hashes to a file (stdout if omitted for dataset mode).
- `--limit <n>`: limit dataset hashing to the first `n` items.
- `--csv`: output CSV for single-key mode.
- `--csv-pred`: include the rm_model prediction as `rm_pred` in CSV output (implies `--csv`).
- `--bench`: emit throughput and per-key timing stats (p50/p95/p99).

Notes:
- `--model-dir` must point to the LEAD output directory that contains `lead.json`.
- Output width is set at hashgen via `--space-bits`; `lead_hasher` outputs the full configured space.
- `--csv/--csv-pred` require `--format hex` (dataset mode uses CSV by default).
- `--csv-pred` reports raw predictions when available; otherwise it falls back to clamped predictions.

CSV schema:
- If `space_bits <= 64`: `key_dec,hash_dec,hash_hex` (plus `rm_pred` when `--csv-pred`).
- If `space_bits > 64`: `key_dec,hash_msb64_hex,hash_hex` (plus `rm_pred` when `--csv-pred`).

Examples:

Hash a key with explicit key type:

```bash
build/hash_functions_core/LEAD/lead_hasher --model-dir lead_output/my_lead_linear_linear_bf100 \
  --key 12345 --key-type u64
```

Hash the first 1M items to CSV:

```bash
build/hash_functions_core/LEAD/lead_hasher --model-dir lead_output/my_lead_linear_linear_bf100 \
  --dataset dataset/osm_cellids_200M_uint64 --limit 1000000 --out hashes.csv
```

Write raw hashes (big-endian) to a file:

```bash
build/hash_functions_core/LEAD/lead_hasher --model-dir lead_output/my_lead_linear_linear_bf100 \
  --dataset dataset/osm_cellids_200M_uint64 --format raw --out hashes.bin
```

Include the raw rm_model prediction in CSV output:

```bash
build/hash_functions_core/LEAD/lead_hasher --model-dir lead_output/my_lead_linear_linear_bf100 \
  --dataset dataset/osm_cellids_200M_uint64 --limit 1000000 --csv-pred --out hashes.csv
```

## Mapping

LEAD maps rm_model predictions to a hash space while preserving order:
- `space_bits` selects the hash space size (default 64).
- `scale_factor` is clamped to (0, 1] and applied in fixed-point (1/2^32 steps)
  to scale the maximum output range.
- Mapping uses the floating prediction output from rm_model.
- Predictions are clamped to `[0, max_index]` in the generated model; LEAD supports
  raw predictions via hashgen-injected wrappers for calibration.

## Prediction Calibration

Calibration linearly rescales raw predictions from the training set into
`[0, max_index]` before mapping, reducing “all-zero” clusters. It performs a
full pass over the training data during hashgen:

```
pred_cal = (pred_raw - min_pred) * (max_index / (max_pred - min_pred))
```

Notes:
- Enabled by default; disable with `--no-calibration`.
- Calibration uses raw (pre-clamp) prediction ranges from training.
- If raw predict symbols are unavailable at load time, calibration is disabled.

## Output Layout

`lead_hashgen` emits:

```
lead_output/<namespace>_<models>_bf<branch>/
  lead.json
  rm_model/
    include/
    src/
    data/   (unless --data-path is set)
    lib/    (auto-built on first load)
    model.json
```

## C++ API

```cpp
#include "lead/lead.h"

lead::Hasher hasher;
hasher.load("lead_output/my_lead_linear_linear_bf100");

uint64_t h64 = hasher.hash64(12345ULL);
auto h128 = hasher.hash128(12345ULL); // {low64, high64}
auto h256 = hasher.hash(12345ULL);    // HashOutput with bits + hex helpers
```

`hash64` and `hash128` return the most significant bits of the configured hash space
(set by `--space-bits` at hashgen).

## Runtime Library Notes

LEAD loads the generated rm_model shared library at runtime. If the shared library is
missing, it is built automatically from the generated source:
- Compiler: `RM_MODEL_CXX` (preferred) or `CXX`; defaults to `c++` on Unix and `cl` on Windows.
- Outputs: `rm_model/lib/<namespace>.so` (Linux), `rm_model/lib/<namespace>.dylib` (macOS),
  or `rm_model/lib/<namespace>.dll` (Windows).

## Citation

If you use LEAD in research or in a published system, please cite:

```bibtex
@INPROCEEDINGS{11192384,
  author={Wang, Shengze and Liu, Yi and Zhang, Xiaoxue and Hu, Liting and Qian, Chen},
  booktitle={2025 IEEE 33rd International Conference on Network Protocols (ICNP)}, 
  title={A Distributed Learned Hash Table}, 
  year={2025},
  pages={1-11},
  doi={10.1109/ICNP65844.2025.11192384}
}

@INPROCEEDINGS{10858529,
  author={Wang, Shengze and Liu, Yi and Zhang, Xiaoxue and Hu, Liting and Qian, Chen},
  booktitle={2024 IEEE 32nd International Conference on Network Protocols (ICNP)}, 
  title={Poster: Distributed Learned Hash Table}, 
  year={2024},
  pages={1-2},
  doi={10.1109/ICNP61940.2024.10858529}
}
```

## License

Apache-2.0
