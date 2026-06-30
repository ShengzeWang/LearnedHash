#!/usr/bin/env python3
"""Prepare and validate Vortex reported benchmark vector datasets.

This public entrypoint is kept for CLI and import compatibility. The
implementation lives in ``vortex_dataset_prepare_lib`` split by dataset specs,
download/materialization, validation, conversion, manifest writing, and CLI
routing.
"""

from __future__ import annotations

from vortex_dataset_prepare_lib.specs import *  # noqa: F401,F403
from vortex_dataset_prepare_lib.paths import *  # noqa: F401,F403
from vortex_dataset_prepare_lib.binary_io import *  # noqa: F401,F403
from vortex_dataset_prepare_lib.contracts import *  # noqa: F401,F403
from vortex_dataset_prepare_lib.file_ops import *  # noqa: F401,F403
from vortex_dataset_prepare_lib.conversion import *  # noqa: F401,F403
from vortex_dataset_prepare_lib.validation import *  # noqa: F401,F403
from vortex_dataset_prepare_lib.manifest import *  # noqa: F401,F403
from vortex_dataset_prepare_lib.download import *  # noqa: F401,F403
from vortex_dataset_prepare_lib.materialize import *  # noqa: F401,F403
from vortex_dataset_prepare_lib.cli import build_parser, main  # noqa: F401
from vortex_dataset_prepare_lib import download as _download
from vortex_dataset_prepare_lib import materialize as _materialize


def _sync_compat_overrides() -> None:
    """Preserve historical monkeypatch behavior for the public wrapper module."""
    _download.run_curl = run_curl
    _download.run_curl_range = run_curl_range
    _download.copy_file = copy_file
    _materialize.materialize_prefix_base = materialize_prefix_base
    _materialize.materialize_query_and_groundtruth = materialize_query_and_groundtruth
    _materialize.ensure_texmex_archive = ensure_texmex_archive
    _materialize.extract_texmex_archive = extract_texmex_archive


def convert_materialized_dataset(*args, **kwargs):
    _sync_compat_overrides()
    return _materialize.convert_materialized_dataset(*args, **kwargs)


def materialize_dataset(*args, **kwargs):
    _sync_compat_overrides()
    return _materialize.materialize_dataset(*args, **kwargs)


def materialize_texmex(*args, **kwargs):
    _sync_compat_overrides()
    return _materialize.materialize_texmex(*args, **kwargs)


if __name__ == "__main__":
    raise SystemExit(main())
