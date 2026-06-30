# Packaging Checklist

Use this checklist before preparing a source archive.

## Source Package Contents

Include:

- Root source files, `CMakeLists.txt`, `CMakePresets.json`, `LICENSE`, and `LearnedHash.svg`.
- First-party source, headers, tests, and command-line tools.
- Documentation:
  - `README.md`
  - focused guides under `docs/public/`
  - component READMEs next to `rm_model`, `LEAD`, and `Vortex`
  - `dataset/README.md` and `dataset/SOURCES.md`
- Required third-party attribution, including `include/faiss/LICENSE` when the
  bundled FAISS provider is shipped.
- Public maintenance scripts listed in `docs/public/tool_reference.md`.

Generated and environment-specific material is outside the source archive:

- Raw dataset payloads.
- Generated models, generated-code build directories, benchmark reports, and caches.
- Build directories and local CMake files.
- Local auxiliary files such as notes, logs, and environment-specific paths.

`.gitignore` and `.gitattributes` encode the package boundary checked by the
validation commands below.

## Validation

Run the full readiness check from fresh build directories:

```bash
tools/check_release_ready.py --jobs 8
```

Save a machine-readable report when needed:

```bash
tools/check_release_ready.py --jobs 8 --report-json /tmp/learnedhash-readiness.json
```

After committing a release candidate, verify the committed archive:

```bash
tools/check_release_ready.py --jobs 8 --strict-source-archive
```

Focused checks:

```bash
tools/check_source_package.py
tools/check_git_hygiene.py --apply-local-config --strict-local-config
tools/check_install_export.py --build-dir build
tools/audit_faiss_vendor.py
```

The readiness check covers Python maintenance-script syntax, whitespace checks,
Git hygiene/path checks, package-boundary checks, cleanup-helper safety, bundled
FAISS attribution, benchmark and dataset schema checks, Release builds, CTest,
and installed-package smoke tests.

## Build Smoke

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build --output-on-failure -j8
```

With CMake 3.21 or newer:

```bash
cmake --preset release-bundled
cmake --build --preset release-bundled
ctest --preset release-bundled
```

Library-only and lean-runtime presets are documented in `docs/public/quickstart.md`.

## Dataset And Benchmark Evidence

Use `dataset/SOURCES.md` and `tools/vortex_dataset_prepare.py` to materialize
and validate public datasets. Use `tools/vortex_sift_benchmark_pack.py` for
benchmark-pack runs and `tools/vortex_selector_scale_validation.py` for bounded
selector scale-policy validation.

Before moving benchmark numbers into a report, create a portable artifact bundle
from the final scale-validation directory:

```bash
tools/vortex_paper_artifact_bundle.py \
  --scale-validation-dir vortex_v1_output/selector_scale_validation/<run> \
  --bundle-name <benchmark-run-name> \
  --copy-reports
```

For committed release candidates, add `--require-clean-git` so the bundle records
a clean source state. The bundle records manifests, commands, environment, git
state, checksums, report paths, and final metrics without copying raw datasets
or generated model/cache payloads.
