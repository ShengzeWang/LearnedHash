"""Dataset validation and exact-search spot-check helpers."""

from __future__ import annotations

import heapq
import mmap
import random
import struct
from pathlib import Path

from .binary_io import (
    read_groundtruth_ids,
    read_xbin_vector,
    squared_l2,
    validate_groundtruth,
    validate_xbin_file,
)
from .conversion import validate_vecs_file
from .paths import (
    converted_base_path,
    converted_groundtruth_path,
    converted_query_path,
    materialized_base_path,
    materialized_groundtruth_path,
    materialized_query_path,
    texmex_expected_size,
    texmex_file_path,
)
from .specs import DatasetSpec, TexMexFileSpec, TexMexSpec, ValidationError


def validate_bundle(
    spec: DatasetSpec,
    *,
    base: Path,
    query: Path,
    groundtruth: Path,
    scan_groundtruth: bool,
) -> dict[str, object]:
    validate_xbin_file(
        base, count=spec.base_count, dim=spec.dim, dtype=spec.base_dtype, label="base"
    )
    validate_xbin_file(
        query,
        count=spec.query_count,
        dim=spec.dim,
        dtype=spec.query_dtype,
        label="query",
    )
    gt = validate_groundtruth(
        groundtruth,
        query_count=spec.query_count,
        topk=spec.groundtruth_topk,
        base_count=spec.base_count,
        scan=scan_groundtruth,
    )
    return {
        "dataset": spec.name,
        "base": {
            "path": str(base),
            "count": spec.base_count,
            "dim": spec.dim,
            "dtype": spec.base_dtype,
        },
        "query": {
            "path": str(query),
            "count": spec.query_count,
            "dim": spec.dim,
            "dtype": spec.query_dtype,
        },
        "groundtruth": gt,
        "ok": True,
    }


def sampled_query_indexes(
    query_count: int, sample_queries: int, seed: int
) -> list[int]:
    if sample_queries <= 0:
        return []
    sample_count = min(sample_queries, query_count)
    return sorted(random.Random(seed).sample(range(query_count), sample_count))


def _exact_spot_check_numpy(
    np,
    spec: DatasetSpec,
    *,
    base: Path,
    query: Path,
    groundtruth: Path,
    sample_queries: int,
    seed: int,
    chunk_rows: int,
    tolerance: float,
) -> dict[str, object]:
    dtype_map = {
        "uint8": np.uint8,
        "int8": np.int8,
        "float32": np.float32,
    }
    base_mm = np.memmap(
        base,
        dtype=dtype_map[spec.base_dtype],
        mode="r",
        offset=8,
        shape=(spec.base_count, spec.dim),
    )
    query_mm = np.memmap(
        query,
        dtype=dtype_map[spec.query_dtype],
        mode="r",
        offset=8,
        shape=(spec.query_count, spec.dim),
    )
    gt_ids = np.memmap(
        groundtruth,
        dtype="<u4",
        mode="r",
        offset=8,
        shape=(spec.query_count, spec.groundtruth_topk),
    )
    sampled = sampled_query_indexes(spec.query_count, sample_queries, seed)
    failures = []
    topk = spec.groundtruth_topk
    for q_index in sampled:
        q = np.asarray(query_mm[q_index], dtype=np.float32)
        best_dist = np.empty(0, dtype=np.float64)
        best_ids = np.empty(0, dtype=np.int64)
        for start in range(0, spec.base_count, chunk_rows):
            end = min(start + chunk_rows, spec.base_count)
            chunk = np.asarray(base_mm[start:end], dtype=np.float32)
            diff = chunk - q
            distances = np.einsum("ij,ij->i", diff, diff).astype(np.float64, copy=False)
            local_k = min(topk, distances.shape[0])
            local = np.argpartition(distances, local_k - 1)[:local_k]
            candidate_dist = np.concatenate((best_dist, distances[local]))
            candidate_ids = np.concatenate(
                (best_ids, local.astype(np.int64, copy=False) + start)
            )
            keep_k = min(topk, candidate_dist.shape[0])
            keep = np.argpartition(candidate_dist, keep_k - 1)[:keep_k]
            best_dist = candidate_dist[keep]
            best_ids = candidate_ids[keep]
        order = np.lexsort((best_ids, best_dist))
        best_dist = best_dist[order]
        threshold = float(best_dist[min(topk - 1, best_dist.shape[0] - 1)])
        official_ids = np.asarray(gt_ids[q_index], dtype=np.int64)
        official_vectors = np.asarray(base_mm[official_ids], dtype=np.float32)
        diff = official_vectors - q
        official_distances = np.einsum("ij,ij->i", diff, diff).astype(
            np.float64, copy=False
        )
        allowed = threshold + max(tolerance, abs(threshold) * 1e-5)
        bad = np.flatnonzero(official_distances > allowed)
        if bad.size:
            rank = int(bad[0])
            failures.append(
                {
                    "query_index": q_index,
                    "rank": rank,
                    "groundtruth_id": int(official_ids[rank]),
                    "groundtruth_distance": float(official_distances[rank]),
                    "exact_kth_distance": threshold,
                    "allowed_distance": allowed,
                }
            )
    if failures:
        raise ValidationError(
            f"{groundtruth}: exact spot-check failed for {len(failures)} queries; "
            f"first failure={failures[0]}"
        )
    return {
        "ok": True,
        "mode": "numpy",
        "seed": seed,
        "sampled_queries": sampled,
        "checked_queries": len(sampled),
        "topk": topk,
        "chunk_rows": chunk_rows,
        "distance": "squared_L2_ranking_equivalent",
        "tie_policy": "accept official ids whose exact distance is tied with kth exact distance",
    }


def _exact_spot_check_python(
    spec: DatasetSpec,
    *,
    base: Path,
    query: Path,
    groundtruth: Path,
    sample_queries: int,
    seed: int,
    chunk_rows: int,
    tolerance: float,
) -> dict[str, object]:
    sampled = sampled_query_indexes(spec.query_count, sample_queries, seed)
    topk = spec.groundtruth_topk
    with base.open("rb") as base_fh, query.open("rb") as query_fh, groundtruth.open(
        "rb"
    ) as gt_fh:
        base_mm = mmap.mmap(base_fh.fileno(), 0, access=mmap.ACCESS_READ)
        query_mm = mmap.mmap(query_fh.fileno(), 0, access=mmap.ACCESS_READ)
        gt_mm = mmap.mmap(gt_fh.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            for q_index in sampled:
                q = read_xbin_vector(
                    query_mm,
                    index=q_index,
                    dim=spec.dim,
                    dtype=spec.query_dtype,
                )
                heap: list[tuple[float, int]] = []
                for base_index in range(spec.base_count):
                    candidate = read_xbin_vector(
                        base_mm,
                        index=base_index,
                        dim=spec.dim,
                        dtype=spec.base_dtype,
                    )
                    dist = squared_l2(q, candidate)
                    item = (-dist, -base_index)
                    if len(heap) < topk:
                        heapq.heappush(heap, item)
                    elif item > heap[0]:
                        heapq.heapreplace(heap, item)
                exact = sorted((-dist, -idx) for dist, idx in heap)
                threshold = exact[-1][0]
                allowed = threshold + max(tolerance, abs(threshold) * 1e-5)
                for rank, base_index in enumerate(
                    read_groundtruth_ids(gt_mm, query_index=q_index, topk=topk)
                ):
                    candidate = read_xbin_vector(
                        base_mm,
                        index=base_index,
                        dim=spec.dim,
                        dtype=spec.base_dtype,
                    )
                    dist = squared_l2(q, candidate)
                    if dist > allowed:
                        raise ValidationError(
                            f"{groundtruth}: exact spot-check failed at query={q_index}, "
                            f"rank={rank}, id={base_index}; distance={dist}, "
                            f"exact_kth_distance={threshold}, allowed={allowed}"
                        )
        finally:
            base_mm.close()
            query_mm.close()
            gt_mm.close()
    return {
        "ok": True,
        "mode": "python",
        "seed": seed,
        "sampled_queries": sampled,
        "checked_queries": len(sampled),
        "topk": topk,
        "chunk_rows": chunk_rows,
        "distance": "squared_L2_ranking_equivalent",
        "tie_policy": "accept official ids whose exact distance is tied with kth exact distance",
    }


def exact_spot_check_groundtruth(
    spec: DatasetSpec,
    *,
    base: Path,
    query: Path,
    groundtruth: Path,
    sample_queries: int,
    seed: int,
    chunk_rows: int,
    tolerance: float,
) -> dict[str, object]:
    if chunk_rows <= 0:
        raise ValidationError("--spot-check-base-chunk must be positive")
    validate_xbin_file(
        base, count=spec.base_count, dim=spec.dim, dtype=spec.base_dtype, label="base"
    )
    validate_xbin_file(
        query,
        count=spec.query_count,
        dim=spec.dim,
        dtype=spec.query_dtype,
        label="query",
    )
    validate_groundtruth(
        groundtruth,
        query_count=spec.query_count,
        topk=spec.groundtruth_topk,
        base_count=spec.base_count,
        scan=True,
    )
    try:
        import numpy as np  # type: ignore
    except ImportError:
        return _exact_spot_check_python(
            spec,
            base=base,
            query=query,
            groundtruth=groundtruth,
            sample_queries=sample_queries,
            seed=seed,
            chunk_rows=chunk_rows,
            tolerance=tolerance,
        )
    return _exact_spot_check_numpy(
        np,
        spec,
        base=base,
        query=query,
        groundtruth=groundtruth,
        sample_queries=sample_queries,
        seed=seed,
        chunk_rows=chunk_rows,
        tolerance=tolerance,
    )


def validate_texmex_vector_file(
    path: Path, file_spec: TexMexFileSpec, *, scan_headers: bool = True
) -> dict[str, object]:
    if not path.is_file():
        raise ValidationError(f"{file_spec.output_name} is missing: {path}")
    expected_size = texmex_expected_size(file_spec)
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValidationError(
            f"{path}: wrong byte size; expected {expected_size}, found {actual_size}"
        )
    row_bytes = 4 + file_spec.dim * 4
    with path.open("rb") as fh:
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            if scan_headers:
                for row in range(file_spec.count):
                    dim = struct.unpack_from("<i", mm, row * row_bytes)[0]
                    if dim != file_spec.dim:
                        raise ValidationError(
                            f"{path}: row {row} has dim={dim}, expected {file_spec.dim}"
                        )
            if file_spec.kind == "ivecs" and file_spec.id_bound is not None:
                for row in range(file_spec.count):
                    offset = row * row_bytes + 4
                    for rank in range(file_spec.dim):
                        value = struct.unpack_from("<i", mm, offset + rank * 4)[0]
                        if value < 0 or value >= file_spec.id_bound:
                            raise ValidationError(
                                f"{path}: id out of bounds at row={row}, rank={rank}: "
                                f"{value} not in [0, {file_spec.id_bound})"
                            )
        finally:
            mm.close()
    return {
        "path": str(path),
        "kind": file_spec.kind,
        "count": file_spec.count,
        "dim": file_spec.dim,
        "bytes": actual_size,
        "headers_checked": scan_headers,
        "id_bound": file_spec.id_bound,
    }


def validate_texmex_dataset(
    spec: TexMexSpec, output_root: Path, *, scan_headers: bool = True
) -> dict[str, object]:
    files = []
    for file_spec in spec.files:
        files.append(
            validate_texmex_vector_file(
                texmex_file_path(spec, output_root, file_spec),
                file_spec,
                scan_headers=scan_headers,
            )
        )
    return {
        "dataset": spec.name,
        "display_name": spec.display_name,
        "metric": spec.metric,
        "files": files,
        "ok": True,
    }


def validate_materialized_dataset(
    spec: DatasetSpec,
    output_root: Path,
    *,
    scan_groundtruth: bool,
    spot_check_queries: int = 0,
    spot_check_seed: int = 20260101,
    spot_check_base_chunk: int = 65_536,
    spot_check_tolerance: float = 1e-4,
) -> dict[str, object]:
    base = materialized_base_path(spec, output_root)
    query = materialized_query_path(spec, output_root)
    groundtruth = materialized_groundtruth_path(spec, output_root)
    validation = validate_bundle(
        spec,
        base=base,
        query=query,
        groundtruth=groundtruth,
        scan_groundtruth=scan_groundtruth,
    )
    if spot_check_queries > 0:
        validation["exact_spot_check"] = exact_spot_check_groundtruth(
            spec,
            base=base,
            query=query,
            groundtruth=groundtruth,
            sample_queries=spot_check_queries,
            seed=spot_check_seed,
            chunk_rows=spot_check_base_chunk,
            tolerance=spot_check_tolerance,
        )
    return validation


def validate_converted_dataset(
    spec: DatasetSpec,
    output_root: Path,
    *,
    scan_headers: bool,
) -> dict[str, object]:
    base = validate_vecs_file(
        converted_base_path(spec, output_root),
        count=spec.base_count,
        dim=spec.dim,
        kind="fvecs",
        scan_headers=scan_headers,
    )
    query = validate_vecs_file(
        converted_query_path(spec, output_root),
        count=spec.query_count,
        dim=spec.dim,
        kind="fvecs",
        scan_headers=scan_headers,
    )
    groundtruth = validate_vecs_file(
        converted_groundtruth_path(spec, output_root),
        count=spec.query_count,
        dim=spec.groundtruth_topk,
        kind="ivecs",
        id_bound=spec.base_count,
        scan_headers=scan_headers,
    )
    return {
        "dataset": spec.name,
        "base": base,
        "query": query,
        "groundtruth": groundtruth,
        "ok": True,
    }
