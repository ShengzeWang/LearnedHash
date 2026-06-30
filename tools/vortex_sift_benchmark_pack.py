#!/usr/bin/env python3
"""Run a reproducible Vortex v1 benchmark pack on SIFT-style datasets.

The pack is intentionally orchestration-only: it records selector, training,
codegen load, hash latency, model-size, and memory metrics without changing
selector search heuristics or hash algorithms.
"""

from __future__ import annotations

import sys

from vortex_benchmark_pack.cli import main


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
