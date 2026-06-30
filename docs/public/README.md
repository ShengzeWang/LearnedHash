# Public Documentation

This directory contains the public workflow guides for LearnedHash. Start with
`README.md` at the repository root, then use the focused guides below when you
need more detail.

## Guides

- `docs/public/quickstart.md`: build, test, install, and minimal LEAD/VortexHash examples.
- `docs/public/faiss_provider.md`: bundled, system, and lean-runtime FAISS modes.
- `docs/public/vortex_model_selector.md`: VortexHash selector strategy, roles, metrics, and JSON output.
- `docs/public/vortex_benchmark_pack.md`: reproducible benchmark-pack workflow and output schema.
- `docs/public/vortex_generated_code_api.md`: generated VortexHash C++ and C ABI usage.
- `docs/public/tool_reference.md`: command-line tools and maintenance checks.
- `docs/public/vortex_project.md`: Vortex project note and citation.
- `docs/public/release_checklist.md`: packaging and validation checklist.

Component references live next to the implementation:

- `learned_structures/rm_model/README.md`
- `hash_functions_core/LEAD/README.md`
- `hash_functions_core/Vortex/README.md`

Dataset provenance lives in `dataset/SOURCES.md`. Raw datasets, generated
models, build outputs, benchmark reports, and local logs are excluded from the
source package.

## Validation

Run the readiness check before publishing or packaging:

```bash
tools/check_release_ready.py --jobs 8
```

After committing a release candidate, add `--strict-source-archive` to verify
that `git archive HEAD` contains the expected source package.
