# FAISS Provider And Vendor Audit

Vortex v1 uses FAISS for CPU HNSW, Flat L2 search, clustering, and index I/O.
The repository defaults to a trimmed bundled CPU FAISS snapshot so a fresh
checkout can build without a system FAISS install. GPU, Python wrapper,
cppcontrib, and FAISS docs subtrees are intentionally not vendored because they
are not part of the current C++ build or transitive include closure.

## Provider Options

Set `LEARNEDHASH_FAISS_PROVIDER` at configure time:

- `bundled`: build `include/faiss` from this repository. This is the default.
- `system`: require an installed FAISS CMake package and fail if it does not
  expose `faiss` or `faiss::faiss`.
- `auto`: prefer system FAISS and fall back to the bundled provider.

Examples:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DLEARNEDHASH_FAISS_PROVIDER=bundled

cmake -S . -B build_system -DCMAKE_BUILD_TYPE=Release \
  -DLEARNEDHASH_FAISS_PROVIDER=system \
  -DCMAKE_PREFIX_PATH=/path/to/faiss/install

cmake -S . -B build_auto -DCMAKE_BUILD_TYPE=Release \
  -DLEARNEDHASH_FAISS_PROVIDER=auto
```

To build only the first-party runtime/codegen surface and skip FAISS entirely,
disable the FAISS-backed Vortex layer:

```bash
cmake -S . -B build_lean -DCMAKE_BUILD_TYPE=Release \
  -DLEARNEDHASH_BUILD_TOOLS=OFF \
  -DLEARNEDHASH_BUILD_VORTEX_FAISS_COMPONENTS=OFF
```

This mode builds and exports `LearnedHash::vortex_v1_runtime`, but not
`LearnedHash::vortex_v1_core`, `vortex_model_selector`, `vortex_eval`, or
`hnsw_eval`.

## Vendor Audit

Before changing the bundled FAISS tree, run:

```bash
tools/audit_faiss_vendor.py
tools/audit_faiss_vendor.py --json
```

The audit reports:

- direct first-party FAISS includes;
- FAISS CMake source and header lists;
- transitive FAISS include closure from those lists and first-party includes;
- optional subtree remaining counts for GPU, Python, contrib, and docs material;
- local license and attribution files.

The audit should report `closure_ready=true`, `trim_ready=true`, and zero
optional files remaining. After changing the bundled FAISS tree, rerun a
bundled-provider build, CTest, and `tools/check_source_package.py`.

## Installed Package Notes

When the bundled provider is installed through the top-level project, the prefix
contains both `share/faiss/faiss-config.cmake` and
`lib/cmake/LearnedHash/LearnedHashConfig.cmake`. The FAISS config exports the
OpenMP dependency needed by the static FAISS library and applies the same
AppleClang/Homebrew OpenMP discovery hint used by the in-tree build.
