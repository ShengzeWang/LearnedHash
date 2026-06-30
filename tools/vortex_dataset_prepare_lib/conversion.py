"""Streaming conversion to Vortex .fvecs/.ivecs formats."""

from __future__ import annotations

import mmap
import struct
from pathlib import Path

from .binary_io import validate_groundtruth, validate_xbin_file, xbin_struct_format
from .file_ops import ensure_convertible_output
from .paths import fvecs_size, ivecs_size
from .specs import ValidationError, dtype_size


def dtype_cast_label(dtype: str) -> str:
    if dtype == "float32":
        return "float32_preserved"
    if dtype in {"uint8", "int8"}:
        return f"{dtype}_to_float32_no_normalization"
    raise ValidationError(f"unsupported dtype: {dtype}")


def _write_numpy_rows(np, out, values, *, dim: int) -> None:
    rows = int(values.shape[0])
    row_bytes = 4 + dim * 4
    buf = bytearray(rows * row_bytes)
    headers = np.ndarray(
        shape=(rows,),
        dtype="<i4",
        buffer=buf,
        offset=0,
        strides=(row_bytes,),
    )
    headers.fill(dim)
    payload = np.ndarray(
        shape=(rows, dim),
        dtype="<f4",
        buffer=buf,
        offset=4,
        strides=(row_bytes, 4),
    )
    payload[...] = values
    out.write(buf)


def _convert_xbin_to_fvecs_numpy(
    np,
    source: Path,
    destination: Path,
    *,
    count: int,
    dim: int,
    dtype: str,
    chunk_rows: int,
) -> None:
    dtype_map = {
        "uint8": np.dtype("u1"),
        "int8": np.dtype("i1"),
        "float32": np.dtype("<f4"),
    }
    source_mm = np.memmap(
        source,
        dtype=dtype_map[dtype],
        mode="r",
        offset=8,
        shape=(count, dim),
    )
    tmp = destination.with_name(destination.name + ".tmp")
    try:
        with tmp.open("wb") as out:
            for start in range(0, count, chunk_rows):
                end = min(start + chunk_rows, count)
                _write_numpy_rows(np, out, source_mm[start:end], dim=dim)
        tmp.replace(destination)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    finally:
        del source_mm


def _convert_xbin_to_fvecs_python(
    source: Path,
    destination: Path,
    *,
    count: int,
    dim: int,
    dtype: str,
) -> None:
    value_bytes = dtype_size(dtype)
    row_input_bytes = dim * value_bytes
    fmt = f"<{dim}{xbin_struct_format(dtype)}"
    tmp = destination.with_name(destination.name + ".tmp")
    try:
        with source.open("rb") as src, tmp.open("wb") as out:
            src.seek(8)
            dim_bytes = struct.pack("<i", dim)
            for row in range(count):
                raw = src.read(row_input_bytes)
                if len(raw) != row_input_bytes:
                    raise ValidationError(
                        f"{source}: short read at row {row}; expected {row_input_bytes} bytes"
                    )
                if dtype == "uint8":
                    values = [float(v) for v in raw]
                else:
                    values = [float(v) for v in struct.unpack(fmt, raw)]
                out.write(dim_bytes)
                out.write(struct.pack(f"<{dim}f", *values))
        tmp.replace(destination)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def convert_xbin_to_fvecs(
    source: Path,
    destination: Path,
    *,
    count: int,
    dim: int,
    dtype: str,
    label: str,
    force: bool,
    chunk_rows: int,
) -> dict[str, object]:
    if chunk_rows <= 0:
        raise ValidationError("--conversion-chunk-rows must be positive")
    validate_xbin_file(source, count=count, dim=dim, dtype=dtype, label=label)
    expected_bytes = fvecs_size(count, dim)
    if ensure_convertible_output(
        destination, expected_bytes=expected_bytes, force=force, label=f"{label} fvecs"
    ):
        status = "existing"
    else:
        try:
            import numpy as np  # type: ignore
        except ImportError:
            _convert_xbin_to_fvecs_python(
                source, destination, count=count, dim=dim, dtype=dtype
            )
            mode = "python"
        else:
            _convert_xbin_to_fvecs_numpy(
                np,
                source,
                destination,
                count=count,
                dim=dim,
                dtype=dtype,
                chunk_rows=chunk_rows,
            )
            mode = "numpy"
        status = "converted"
    return {
        "path": str(destination),
        "source_path": str(source),
        "kind": "fvecs",
        "source_dtype": dtype,
        "dtype_cast": dtype_cast_label(dtype),
        "count": count,
        "dim": dim,
        "bytes": destination.stat().st_size,
        "expected_bytes": expected_bytes,
        "status": status,
        "mode": "existing" if status == "existing" else mode,
        "chunk_rows": chunk_rows,
    }


def _write_numpy_id_rows(np, out, ids, *, topk: int) -> None:
    rows = int(ids.shape[0])
    row_bytes = 4 + topk * 4
    buf = bytearray(rows * row_bytes)
    headers = np.ndarray(
        shape=(rows,),
        dtype="<i4",
        buffer=buf,
        offset=0,
        strides=(row_bytes,),
    )
    headers.fill(topk)
    payload = np.ndarray(
        shape=(rows, topk),
        dtype="<i4",
        buffer=buf,
        offset=4,
        strides=(row_bytes, 4),
    )
    payload[...] = ids
    out.write(buf)


def _convert_groundtruth_to_ivecs_numpy(
    np,
    source: Path,
    destination: Path,
    *,
    query_count: int,
    topk: int,
    chunk_rows: int,
) -> None:
    ids = np.memmap(
        source,
        dtype="<u4",
        mode="r",
        offset=8,
        shape=(query_count, topk),
    )
    tmp = destination.with_name(destination.name + ".tmp")
    try:
        with tmp.open("wb") as out:
            for start in range(0, query_count, chunk_rows):
                end = min(start + chunk_rows, query_count)
                _write_numpy_id_rows(np, out, ids[start:end], topk=topk)
        tmp.replace(destination)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    finally:
        del ids


def _convert_groundtruth_to_ivecs_python(
    source: Path,
    destination: Path,
    *,
    query_count: int,
    topk: int,
) -> None:
    row_bytes = topk * 4
    tmp = destination.with_name(destination.name + ".tmp")
    try:
        with source.open("rb") as src, tmp.open("wb") as out:
            src.seek(8)
            dim_bytes = struct.pack("<i", topk)
            for row in range(query_count):
                raw = src.read(row_bytes)
                if len(raw) != row_bytes:
                    raise ValidationError(
                        f"{source}: short ground-truth ID read at row {row}"
                    )
                ids = struct.unpack(f"<{topk}I", raw)
                out.write(dim_bytes)
                out.write(struct.pack(f"<{topk}i", *ids))
        tmp.replace(destination)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def convert_groundtruth_to_ivecs(
    source: Path,
    destination: Path,
    *,
    query_count: int,
    topk: int,
    base_count: int,
    force: bool,
    chunk_rows: int,
) -> dict[str, object]:
    if base_count > 2_147_483_647:
        raise ValidationError("ivecs conversion requires ids to fit in signed int32")
    validate_groundtruth(
        source,
        query_count=query_count,
        topk=topk,
        base_count=base_count,
        scan=True,
    )
    expected_bytes = ivecs_size(query_count, topk)
    if ensure_convertible_output(
        destination,
        expected_bytes=expected_bytes,
        force=force,
        label="groundtruth ivecs",
    ):
        status = "existing"
    else:
        try:
            import numpy as np  # type: ignore
        except ImportError:
            _convert_groundtruth_to_ivecs_python(
                source, destination, query_count=query_count, topk=topk
            )
            mode = "python"
        else:
            _convert_groundtruth_to_ivecs_numpy(
                np,
                source,
                destination,
                query_count=query_count,
                topk=topk,
                chunk_rows=chunk_rows,
            )
            mode = "numpy"
        status = "converted"
    return {
        "path": str(destination),
        "source_path": str(source),
        "kind": "ivecs",
        "source_format": "bigann_groundtruth_ids_distances",
        "count": query_count,
        "dim": topk,
        "bytes": destination.stat().st_size,
        "expected_bytes": expected_bytes,
        "status": status,
        "mode": "existing" if status == "existing" else mode,
        "chunk_rows": chunk_rows,
    }


def _validate_vecs_headers_numpy(
    np,
    path: Path,
    *,
    count: int,
    dim: int,
    kind: str,
    id_bound: int | None,
) -> None:
    row_bytes = 4 + dim * 4
    with path.open("rb") as fh:
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            headers = np.ndarray(
                shape=(count,),
                dtype="<i4",
                buffer=mm,
                offset=0,
                strides=(row_bytes,),
            )
            bad = np.flatnonzero(headers != dim)
            if bad.size:
                row = int(bad[0])
                raise ValidationError(
                    f"{path}: row {row} has dim={int(headers[row])}, expected {dim}"
                )
            if kind == "ivecs" and id_bound is not None:
                ids = np.ndarray(
                    shape=(count, dim),
                    dtype="<i4",
                    buffer=mm,
                    offset=4,
                    strides=(row_bytes, 4),
                )
                if ids.size:
                    min_id = int(ids.min())
                    max_id = int(ids.max())
                    if min_id < 0 or max_id >= id_bound:
                        raise ValidationError(
                            f"{path}: id bounds check failed; min={min_id}, "
                            f"max={max_id}, expected [0, {id_bound})"
                        )
                del ids
            del headers
        finally:
            mm.close()


def _validate_vecs_headers_python(
    path: Path,
    *,
    count: int,
    dim: int,
    kind: str,
    id_bound: int | None,
) -> None:
    row_bytes = 4 + dim * 4
    with path.open("rb") as fh:
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            for row in range(count):
                offset = row * row_bytes
                actual_dim = struct.unpack_from("<i", mm, offset)[0]
                if actual_dim != dim:
                    raise ValidationError(
                        f"{path}: row {row} has dim={actual_dim}, expected {dim}"
                    )
                if kind == "ivecs" and id_bound is not None:
                    ids_offset = offset + 4
                    for rank in range(dim):
                        value = struct.unpack_from("<i", mm, ids_offset + rank * 4)[0]
                        if value < 0 or value >= id_bound:
                            raise ValidationError(
                                f"{path}: id out of bounds at row={row}, rank={rank}: "
                                f"{value} not in [0, {id_bound})"
                            )
        finally:
            mm.close()


def validate_vecs_file(
    path: Path,
    *,
    count: int,
    dim: int,
    kind: str,
    id_bound: int | None = None,
    scan_headers: bool = True,
) -> dict[str, object]:
    if kind not in {"fvecs", "ivecs"}:
        raise ValidationError(f"unsupported vecs kind: {kind}")
    if not path.is_file():
        raise ValidationError(f"{kind} file is missing: {path}")
    expected_bytes = (
        fvecs_size(count, dim) if kind == "fvecs" else ivecs_size(count, dim)
    )
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValidationError(
            f"{path}: wrong {kind} byte size; expected {expected_bytes}, found {actual_bytes}"
        )
    if scan_headers:
        try:
            import numpy as np  # type: ignore
        except ImportError:
            _validate_vecs_headers_python(
                path, count=count, dim=dim, kind=kind, id_bound=id_bound
            )
        else:
            _validate_vecs_headers_numpy(
                np, path, count=count, dim=dim, kind=kind, id_bound=id_bound
            )
    return {
        "path": str(path),
        "kind": kind,
        "count": count,
        "dim": dim,
        "bytes": actual_bytes,
        "headers_checked": scan_headers,
        "id_bound": id_bound,
    }
