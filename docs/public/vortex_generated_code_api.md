# Vortex Generated-Code API

Vortex codegen emits a portable, inference-only C++ shared library around a
trained VortexHash model.

## Generate And Build

```bash
build/hash_functions_core/Vortex/vortex_v1_cli codegen \
  --model vortex_v1_output/siftsmall_v1.model \
  --output-dir vortex_v1_output/siftsmall_codegen \
  --name siftsmall_v1

cmake -S vortex_v1_output/siftsmall_codegen/codegen \
  -B vortex_v1_output/siftsmall_codegen/codegen/build \
  -DCMAKE_BUILD_TYPE=Release
cmake --build vortex_v1_output/siftsmall_codegen/codegen/build --config Release
```

The generated package embeds the inference runtime sources it needs. If codegen
runs outside the repository tree, set `VORTEX_ROOT` to the repository root so the
runtime can be copied.

## Load Options

Centroid-index loading and query-cache tokens are enabled by default.

```cpp
vortex_codegen::siftsmall_v1::Hasher hasher;
vortex_codegen::siftsmall_v1::LoadOptions opts;
opts.enable_centroid_index = true;
opts.enable_query_cache = true;
hasher.load_from_dir("vortex_v1_output/siftsmall_codegen", opts);
```

Disable either option only for A/B validation or debugging:

```cpp
opts.enable_centroid_index = false;
opts.enable_query_cache = false;
```

## Hash And Centroid Query

```cpp
std::vector<float> query(hasher.dim(), 0.0f);
auto hc = hasher.hash_with_centroid(query.data());

std::vector<vortex_codegen::siftsmall_v1::AnnNeighbor> centroids;
hasher.ann_query(query.data(), 8, &centroids, hc.cache_token);
```

`hash_with_centroid` returns the hash, nearest active centroid id, exact L2
distance, and a cache token. Reuse the token for later top-N centroid search on
the same query pointer and loaded hasher state to avoid recomputing query-pivot
distances.

## C ABI

The generated header also exposes a C ABI with one process-global hasher:

```c
int ok = siftsmall_v1_vortex_infer_load_ex("vortex_v1_output/siftsmall_codegen", 1, 16);
uint64_t words[4] = {0, 0, 0, 0};
uint32_t nearest = 0;
float nearest_d2 = 0.0f;
uint64_t cache_token = 0;
ok = siftsmall_v1_vortex_infer_hash_with_centroid(
    query, words, &nearest, &nearest_d2, &cache_token);
```

Use the C ABI when integrating from languages that can call C shared libraries.
Treat load and cleanup as process-global operations.

## Runtime Behavior

- `hash64()` is valid only when `hash_bits <= 64`.
- `hash()` always returns four little-endian 64-bit words.
- The centroid index is prebuilt/serialized during training and loaded once when
  the hasher loads. If it is missing or incompatible, the generated runtime can
  rebuild the in-memory index.
- Active and pivot centroid vectors are packed into aligned contiguous buffers at
  load time for cache/SIMD-friendly dot-product scans.
- ANN centroid queries use pivot-bound pruning and exact L2 re-ranking; they do
  not change hash semantics.
