"""Binary format, checksum, and low-level validation helpers."""

from __future__ import annotations

import hashlib
import mmap
import struct
from pathlib import Path

from .specs import DatasetSpec, ValidationError, dtype_size, xbin_size


def read_xbin_header(path: Path) -> tuple[int, int]:
    with path.open("rb") as fh:
        raw = fh.read(8)
    if len(raw) != 8:
        raise ValidationError(f"{path}: file is too small for an xbin header")
    return struct.unpack("<II", raw)


def validate_xbin_file(
    path: Path, *, count: int, dim: int, dtype: str, label: str
) -> None:
    if not path.is_file():
        raise ValidationError(f"{label} file is missing: {path}")
    expected_size = xbin_size(count, dim, dtype)
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValidationError(
            f"{path}: wrong {label} byte size; expected {expected_size}, found {actual_size}"
        )
    actual_count, actual_dim = read_xbin_header(path)
    if actual_count != count or actual_dim != dim:
        raise ValidationError(
            f"{path}: wrong {label} header; expected ({count}, {dim}), "
            f"found ({actual_count}, {actual_dim})"
        )


def patch_prefix_header(
    spec: DatasetSpec, path: Path, *, dry_run: bool = False
) -> dict[str, object]:
    if not path.is_file():
        raise ValidationError(f"base prefix file is missing: {path}")
    expected_size = spec.prefix_bytes
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValidationError(
            f"{path}: wrong prefix byte size; expected {expected_size}, found {actual_size}. "
            f"Use byte range {spec.byte_range}."
        )
    old_count, dim = read_xbin_header(path)
    if dim != spec.dim:
        raise ValidationError(f"{path}: expected dim={spec.dim}, found dim={dim}")
    if old_count not in {spec.source_count, spec.base_count}:
        raise ValidationError(
            f"{path}: expected header count {spec.source_count} before patch or "
            f"{spec.base_count} after patch, found {old_count}"
        )
    if not dry_run and old_count != spec.base_count:
        with path.open("r+b") as fh:
            fh.write(struct.pack("<I", spec.base_count))
    new_count, new_dim = (old_count, dim) if dry_run else read_xbin_header(path)
    if not dry_run and (new_count, new_dim) != (spec.base_count, spec.dim):
        raise ValidationError(f"{path}: header patch did not persist")
    return {
        "path": str(path),
        "dry_run": dry_run,
        "old_count": old_count,
        "old_dim": dim,
        "new_count": spec.base_count if dry_run else new_count,
        "new_dim": spec.dim if dry_run else new_dim,
        "expected_size": expected_size,
    }


def read_groundtruth_header(path: Path) -> tuple[int, int]:
    with path.open("rb") as fh:
        raw = fh.read(8)
    if len(raw) != 8:
        raise ValidationError(f"{path}: file is too small for a ground-truth header")
    return struct.unpack("<II", raw)


def validate_groundtruth(
    path: Path,
    *,
    query_count: int,
    topk: int,
    base_count: int,
    scan: bool = True,
) -> dict[str, object]:
    if not path.is_file():
        raise ValidationError(f"ground-truth file is missing: {path}")
    actual_query_count, actual_topk = read_groundtruth_header(path)
    if (actual_query_count, actual_topk) != (query_count, topk):
        raise ValidationError(
            f"{path}: wrong ground-truth header; expected ({query_count}, {topk}), "
            f"found ({actual_query_count}, {actual_topk})"
        )
    ids_bytes = query_count * topk * 4
    distances_bytes = query_count * topk * 4
    expected_size = 8 + ids_bytes + distances_bytes
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValidationError(
            f"{path}: wrong ground-truth byte size; expected {expected_size}, found {actual_size}"
        )
    result = {
        "path": str(path),
        "query_count": query_count,
        "topk": topk,
        "bytes": actual_size,
        "id_bounds_checked": False,
        "distance_order_checked": False,
    }
    if not scan:
        return result
    with path.open("rb") as fh:
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            ids_offset = 8
            distances_offset = ids_offset + ids_bytes
            for i in range(query_count * topk):
                value = struct.unpack_from("<I", mm, ids_offset + i * 4)[0]
                if value >= base_count:
                    q, rank = divmod(i, topk)
                    raise ValidationError(
                        f"{path}: ground-truth id out of bounds at query={q}, rank={rank}: "
                        f"{value} >= {base_count}"
                    )
            for q in range(query_count):
                row_offset = distances_offset + q * topk * 4
                prev = None
                for rank in range(topk):
                    value = struct.unpack_from("<f", mm, row_offset + rank * 4)[0]
                    if prev is not None and value < prev:
                        raise ValidationError(
                            f"{path}: distances decrease at query={q}, rank={rank}: "
                            f"{value} < {prev}"
                        )
                    prev = value
        finally:
            mm.close()
    result["id_bounds_checked"] = True
    result["distance_order_checked"] = True
    return result


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(8 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def md5_file(path: Path) -> str:
    h = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(8 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def maybe_sha256(path: Path | None, *, enabled: bool = True) -> str | None:
    if not enabled or path is None or not path.is_file():
        return None
    return sha256_file(path)


def xbin_struct_format(dtype: str) -> str:
    if dtype == "uint8":
        return "B"
    if dtype == "int8":
        return "b"
    if dtype == "float32":
        return "f"
    raise ValidationError(f"unsupported dtype: {dtype}")


def read_xbin_vector(mm: mmap.mmap, *, index: int, dim: int, dtype: str) -> list[float]:
    offset = 8 + index * dim * dtype_size(dtype)
    if dtype == "uint8":
        return [float(value) for value in mm[offset : offset + dim]]
    fmt = f"<{dim}{xbin_struct_format(dtype)}"
    return [float(value) for value in struct.unpack_from(fmt, mm, offset)]


def squared_l2(a: list[float], b: list[float]) -> float:
    return sum((av - bv) * (av - bv) for av, bv in zip(a, b, strict=True))


def read_groundtruth_ids(
    mm: mmap.mmap, *, query_index: int, topk: int
) -> tuple[int, ...]:
    offset = 8 + query_index * topk * 4
    return struct.unpack_from(f"<{topk}I", mm, offset)
