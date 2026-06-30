#!/usr/bin/env python3
"""Regression checks for the Vortex dataset preparation helper."""

from __future__ import annotations

import hashlib
import json
import struct
import subprocess
import tarfile
import tempfile
from pathlib import Path

import vortex_dataset_prepare as prep


def mock_spec() -> prep.DatasetSpec:
    return prep.DatasetSpec(
        name="mock10m",
        display_name="MOCK10M",
        role="test",
        source_family="mock",
        base_url="https://example.invalid/base.u8bin",
        query_url="https://example.invalid/query.u8bin",
        groundtruth_url="https://example.invalid/gt",
        base_dtype="uint8",
        query_dtype="uint8",
        dim=2,
        base_count=3,
        source_count=100,
        query_count=2,
        groundtruth_topk=2,
    )


def write_xbin(
    path: Path,
    *,
    header_count: int,
    dim: int,
    dtype: str,
    payload_count: int | None = None,
) -> None:
    payload_count = header_count if payload_count is None else payload_count
    path.write_bytes(
        struct.pack("<II", header_count, dim)
        + bytes(payload_count * dim * prep.dtype_size(dtype))
    )


def write_xbin_values(
    path: Path,
    *,
    header_count: int,
    dim: int,
    dtype: str,
    values: list[list[float]],
) -> None:
    payload = bytearray(struct.pack("<II", header_count, dim))
    fmt = {"uint8": "B", "int8": "b", "float32": "f"}[dtype]
    for row in values:
        if len(row) != dim:
            raise AssertionError("wrong fixture dimension")
        if dtype == "float32":
            payload.extend(struct.pack(f"<{dim}{fmt}", *row))
        else:
            payload.extend(struct.pack(f"<{dim}{fmt}", *[int(v) for v in row]))
    path.write_bytes(payload)


def write_groundtruth(
    path: Path, *, query_count: int, topk: int, ids: list[int], distances: list[float]
) -> None:
    if len(ids) != query_count * topk:
        raise AssertionError("wrong id fixture length")
    if len(distances) != query_count * topk:
        raise AssertionError("wrong distance fixture length")
    payload = bytearray(struct.pack("<II", query_count, topk))
    payload.extend(struct.pack(f"<{len(ids)}I", *ids))
    payload.extend(struct.pack(f"<{len(distances)}f", *distances))
    path.write_bytes(payload)


def read_fvecs(path: Path, *, count: int, dim: int) -> list[list[float]]:
    rows = []
    with path.open("rb") as fh:
        for _ in range(count):
            actual_dim = struct.unpack("<i", fh.read(4))[0]
            if actual_dim != dim:
                raise AssertionError(f"wrong fvecs dim: {actual_dim}")
            rows.append(list(struct.unpack(f"<{dim}f", fh.read(dim * 4))))
    return rows


def read_ivecs(path: Path, *, count: int, dim: int) -> list[list[int]]:
    rows = []
    with path.open("rb") as fh:
        for _ in range(count):
            actual_dim = struct.unpack("<i", fh.read(4))[0]
            if actual_dim != dim:
                raise AssertionError(f"wrong ivecs dim: {actual_dim}")
            rows.append(list(struct.unpack(f"<{dim}i", fh.read(dim * 4))))
    return rows


def write_texmex_vectors(path: Path, *, count: int, dim: int, kind: str) -> None:
    payload = bytearray()
    for row in range(count):
        payload.extend(struct.pack("<i", dim))
        if kind == "fvecs":
            payload.extend(
                struct.pack(f"<{dim}f", *[float(row + i) for i in range(dim)])
            )
        elif kind == "ivecs":
            payload.extend(struct.pack(f"<{dim}i", *range(dim)))
        else:
            raise AssertionError(f"unsupported kind: {kind}")
    path.write_bytes(payload)


def mock_texmex_spec() -> prep.TexMexSpec:
    return prep.TexMexSpec(
        name="mocktex",
        display_name="MockTex",
        role="test",
        url="ftp://example.invalid/mock.tar.gz",
        archive_bytes=0,
        md5="unused",
        metric="L2",
        files=(
            prep.TexMexFileSpec(
                "mock/mock_base.fvecs", "mock_base.fvecs", 3, 2, "fvecs"
            ),
            prep.TexMexFileSpec(
                "mock/mock_query.fvecs", "mock_query.fvecs", 2, 2, "fvecs"
            ),
            prep.TexMexFileSpec(
                "mock/mock_learn.fvecs", "mock_learn.fvecs", 4, 2, "fvecs"
            ),
            prep.TexMexFileSpec(
                "mock/mock_groundtruth.ivecs",
                "mock_groundtruth.ivecs",
                2,
                2,
                "ivecs",
                id_bound=3,
            ),
        ),
    )


def expect_validation_error(fn, contains: str) -> None:
    try:
        fn()
    except prep.ValidationError as exc:
        if contains not in str(exc):
            raise AssertionError(
                f"expected error containing {contains!r}, got {exc!r}"
            ) from exc
        return
    raise AssertionError(f"expected ValidationError containing {contains!r}")


def test_patch_prefix_header() -> None:
    spec = mock_spec()
    with tempfile.TemporaryDirectory(prefix="vortex-dataset-prepare-") as tmp:
        base = Path(tmp) / "base.10M.u8bin"
        write_xbin(
            base,
            header_count=spec.source_count,
            dim=spec.dim,
            dtype=spec.base_dtype,
            payload_count=spec.base_count,
        )
        dry = prep.patch_prefix_header(spec, base, dry_run=True)
        if (
            dry["old_count"] != spec.source_count
            or prep.read_xbin_header(base)[0] != spec.source_count
        ):
            raise AssertionError("dry-run patch changed the header")
        patched = prep.patch_prefix_header(spec, base)
        if patched["new_count"] != spec.base_count:
            raise AssertionError("patch did not report the selected prefix count")
        prep.validate_xbin_file(
            base,
            count=spec.base_count,
            dim=spec.dim,
            dtype=spec.base_dtype,
            label="base",
        )


def test_validate_bundle_and_query_match() -> None:
    spec = mock_spec()
    with tempfile.TemporaryDirectory(prefix="vortex-dataset-prepare-") as tmp:
        root = Path(tmp)
        base = root / "base.u8bin"
        query = root / "query.u8bin"
        gt = root / "gt.bin"
        write_xbin(
            base, header_count=spec.base_count, dim=spec.dim, dtype=spec.base_dtype
        )
        write_xbin(
            query, header_count=spec.query_count, dim=spec.dim, dtype=spec.query_dtype
        )
        write_groundtruth(
            gt,
            query_count=spec.query_count,
            topk=spec.groundtruth_topk,
            ids=[0, 1, 1, 2],
            distances=[0.0, 1.0, 0.5, 2.0],
        )
        result = prep.validate_bundle(
            spec, base=base, query=query, groundtruth=gt, scan_groundtruth=True
        )
        if not result["ok"]:
            raise AssertionError("valid bundle rejected")

        bad_query = root / "bad_query.u8bin"
        write_xbin(
            bad_query,
            header_count=spec.query_count,
            dim=spec.dim + 1,
            dtype=spec.query_dtype,
        )
        expect_validation_error(
            lambda: prep.validate_bundle(
                spec, base=base, query=bad_query, groundtruth=gt, scan_groundtruth=True
            ),
            "wrong query byte size",
        )


def test_groundtruth_safety_checks() -> None:
    spec = mock_spec()
    with tempfile.TemporaryDirectory(prefix="vortex-dataset-prepare-") as tmp:
        root = Path(tmp)
        gt_bad_id = root / "gt_bad_id.bin"
        write_groundtruth(
            gt_bad_id,
            query_count=spec.query_count,
            topk=spec.groundtruth_topk,
            ids=[0, 3, 1, 2],
            distances=[0.0, 1.0, 0.5, 2.0],
        )
        expect_validation_error(
            lambda: prep.validate_groundtruth(
                gt_bad_id,
                query_count=spec.query_count,
                topk=spec.groundtruth_topk,
                base_count=spec.base_count,
                scan=True,
            ),
            "out of bounds",
        )

        gt_bad_distance = root / "gt_bad_distance.bin"
        write_groundtruth(
            gt_bad_distance,
            query_count=spec.query_count,
            topk=spec.groundtruth_topk,
            ids=[0, 1, 1, 2],
            distances=[1.0, 0.0, 0.5, 2.0],
        )
        expect_validation_error(
            lambda: prep.validate_groundtruth(
                gt_bad_distance,
                query_count=spec.query_count,
                topk=spec.groundtruth_topk,
                base_count=spec.base_count,
                scan=True,
            ),
            "distances decrease",
        )


def test_real_specs_encode_validated_ranges() -> None:
    expected = {
        "sift10m": ("0-1280000007", 1_280_000_008, 10_000, 128, "uint8"),
        "deep10m_l2": ("0-3840000007", 3_840_000_008, 10_000, 96, "float32"),
        "spacev10m": ("0-1000000007", 1_000_000_008, 29_316, 100, "int8"),
    }
    for name, (byte_range, size, query_count, dim, dtype) in expected.items():
        spec = prep.DATASETS[name]
        if (
            spec.byte_range,
            spec.prefix_bytes,
            spec.query_count,
            spec.dim,
            spec.base_dtype,
        ) != (
            byte_range,
            size,
            query_count,
            dim,
            dtype,
        ):
            raise AssertionError(f"unexpected spec for {name}: {spec}")


def test_workload_contracts_are_l2_canonical_and_exact() -> None:
    result = prep.validate_workload_contracts()
    if not result["ok"]:
        raise AssertionError(f"dataset contract check failed: {result['errors']}")
    rows = result["workloads"]
    expected_profiles = {
        "sift10m",
        "deep10m_l2",
        "spacev10m",
        "siftsmall",
        "sift",
        "gist1m",
    }
    actual_profiles = {row["profile"] for row in rows}
    if actual_profiles != expected_profiles:
        raise AssertionError(f"unexpected workload contract set: {actual_profiles}")
    for row in rows:
        if row["metric"] != "L2":
            raise AssertionError(f"{row['profile']} is not L2")
        if not row["groundtruth_exact"]:
            raise AssertionError(f"{row['profile']} does not require exact GT")
        if row["slice_based"]:
            if row["groundtruth_scope"] != "official_exact_top100_for_evaluated_slice":
                raise AssertionError(f"{row['profile']} GT is not slice-scoped")
            if row["base_split"] != "canonical_prefix_slice_from_official_1b_base":
                raise AssertionError(
                    f"{row['profile']} is not a canonical prefix split"
                )
        else:
            if row["base_split"] != "official_texmex_base_file":
                raise AssertionError(f"{row['profile']} does not use TexMex base split")
            if row["query_split"] != "official_texmex_query_file":
                raise AssertionError(
                    f"{row['profile']} does not use TexMex query split"
                )


def test_dataset_profiles_expose_benchmark_safe_layout() -> None:
    rows = prep.dataset_profile_rows()
    by_profile = {row["profile"]: row for row in rows}
    expected_profiles = {
        "siftsmall",
        "sift",
        "gist1m",
        "sift10m",
        "deep10m_l2",
        "spacev10m",
    }
    if set(by_profile) != expected_profiles:
        raise AssertionError(f"unexpected dataset profiles: {set(by_profile)}")

    if by_profile["sift"]["base_file"] != "sift_base.fvecs":
        raise AssertionError("SIFT profile does not expose TexMex base filename")
    if by_profile["sift"]["manifest_required"]:
        raise AssertionError("SIFT manifest must remain optional for compatibility")
    if not by_profile["gist1m"]["manifest_required"]:
        raise AssertionError("GIST1M extension profile must require provenance")
    if by_profile["sift10m"]["base_file"] != "sift10m_base.fvecs":
        raise AssertionError("SIFT10M profile does not expose converted base filename")
    for row in rows:
        if row["metric"] != "L2":
            raise AssertionError(f"{row['profile']} is not L2")
        for key in ("base_file", "query_file", "groundtruth_file"):
            if "/" in row[key]:
                raise AssertionError(f"{row['profile']} {key} is not a local filename")


def test_materialize_prefix_dataset_and_spot_check() -> None:
    spec = mock_spec()
    with tempfile.TemporaryDirectory(prefix="vortex-prefix-materialize-") as tmp:
        root = Path(tmp)
        fixtures = root / "fixtures"
        fixtures.mkdir()
        source_base = fixtures / "base.1B.u8bin"
        source_query = fixtures / "query.u8bin"
        source_gt = fixtures / "gt"
        write_xbin_values(
            source_base,
            header_count=spec.source_count,
            dim=spec.dim,
            dtype=spec.base_dtype,
            values=[[0, 0], [10, 0], [0, 10]],
        )
        write_xbin_values(
            source_query,
            header_count=spec.query_count,
            dim=spec.dim,
            dtype=spec.query_dtype,
            values=[[1, 0], [9, 0]],
        )
        write_groundtruth(
            source_gt,
            query_count=spec.query_count,
            topk=spec.groundtruth_topk,
            ids=[0, 1, 1, 0],
            distances=[1.0, 81.0, 1.0, 81.0],
        )

        original_run_curl_range = prep.run_curl_range
        original_run_curl = prep.run_curl

        def fake_run_curl_range(
            url,
            output,
            *,
            byte_range,
            expected_bytes,
            force_download=False,
        ):
            if url != spec.base_url:
                raise AssertionError(f"unexpected range URL: {url}")
            prep.copy_file(source_base, output)
            if output.stat().st_size != expected_bytes:
                raise AssertionError("fixture base size mismatch")
            return {
                "path": str(output),
                "url": url,
                "byte_range": byte_range,
                "bytes": expected_bytes,
                "expected_bytes": expected_bytes,
                "status": "downloaded",
            }

        def fake_run_curl(url, output, *, force=False):
            if url == spec.query_url:
                prep.copy_file(source_query, output)
            elif url == spec.groundtruth_url:
                prep.copy_file(source_gt, output)
            else:
                raise AssertionError(f"unexpected URL: {url}")

        prep.run_curl_range = fake_run_curl_range
        prep.run_curl = fake_run_curl
        try:
            output_root = root / "vector_datasets"
            result = prep.materialize_dataset(
                spec,
                output_root=output_root,
                force_download=False,
                scan_groundtruth=True,
                include_file_sha256=False,
                spot_check_queries=2,
                spot_check_seed=7,
                spot_check_base_chunk=2,
            )
        finally:
            prep.run_curl_range = original_run_curl_range
            prep.run_curl = original_run_curl

        if not result["ok"]:
            raise AssertionError("materialization did not report ok")
        if prep.read_xbin_header(prep.materialized_base_path(spec, output_root))[0] != 3:
            raise AssertionError("materialized base header was not patched")
        if not (output_root / spec.name / "manifest.json").is_file():
            raise AssertionError("materialized manifest was not written")
        spot = result["validation"]["exact_spot_check"]
        if spot["checked_queries"] != 2 or not spot["ok"]:
            raise AssertionError(f"unexpected spot-check result: {spot}")


def test_convert_materialized_dataset_to_vortex_formats() -> None:
    spec = mock_spec()
    with tempfile.TemporaryDirectory(prefix="vortex-prefix-convert-") as tmp:
        output_root = Path(tmp) / "vector_datasets"
        output_dir = prep.dataset_output_dir(spec, output_root)
        output_dir.mkdir(parents=True)
        write_xbin_values(
            prep.materialized_base_path(spec, output_root),
            header_count=spec.base_count,
            dim=spec.dim,
            dtype=spec.base_dtype,
            values=[[0, 0], [10, 0], [0, 10]],
        )
        write_xbin_values(
            prep.materialized_query_path(spec, output_root),
            header_count=spec.query_count,
            dim=spec.dim,
            dtype=spec.query_dtype,
            values=[[1, 0], [9, 0]],
        )
        write_groundtruth(
            prep.materialized_groundtruth_path(spec, output_root),
            query_count=spec.query_count,
            topk=spec.groundtruth_topk,
            ids=[0, 1, 1, 0],
            distances=[1.0, 81.0, 1.0, 81.0],
        )

        result = prep.convert_materialized_dataset(
            spec,
            output_root=output_root,
            force_convert=False,
            scan_converted=True,
            include_file_sha256=True,
            chunk_rows=2,
            command_line="test convert",
        )
        if not result["ok"]:
            raise AssertionError("conversion did not report ok")

        base_rows = read_fvecs(
            prep.converted_base_path(spec, output_root),
            count=spec.base_count,
            dim=spec.dim,
        )
        query_rows = read_fvecs(
            prep.converted_query_path(spec, output_root),
            count=spec.query_count,
            dim=spec.dim,
        )
        gt_rows = read_ivecs(
            prep.converted_groundtruth_path(spec, output_root),
            count=spec.query_count,
            dim=spec.groundtruth_topk,
        )
        if base_rows != [[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]]:
            raise AssertionError(f"unexpected base fvecs rows: {base_rows}")
        if query_rows != [[1.0, 0.0], [9.0, 0.0]]:
            raise AssertionError(f"unexpected query fvecs rows: {query_rows}")
        if gt_rows != [[0, 1], [1, 0]]:
            raise AssertionError(f"unexpected ivecs rows: {gt_rows}")

        manifest = json.loads((output_dir / "manifest.json").read_text())
        conversion = manifest["vortex_conversion"]
        if conversion["base"]["dtype_cast"] != "uint8_to_float32_no_normalization":
            raise AssertionError("manifest did not record base dtype cast")
        if "sha256" not in conversion["base"] or "sha256" not in conversion["groundtruth"]:
            raise AssertionError("manifest did not record output checksums")


def test_exact_spot_check_rejects_wrong_slice_groundtruth() -> None:
    spec = mock_spec()
    with tempfile.TemporaryDirectory(prefix="vortex-prefix-spot-") as tmp:
        root = Path(tmp)
        base = root / "base.u8bin"
        query = root / "query.u8bin"
        gt = root / "gt"
        write_xbin_values(
            base,
            header_count=spec.base_count,
            dim=spec.dim,
            dtype=spec.base_dtype,
            values=[[0, 0], [10, 0], [0, 10]],
        )
        write_xbin_values(
            query,
            header_count=spec.query_count,
            dim=spec.dim,
            dtype=spec.query_dtype,
            values=[[1, 0], [9, 0]],
        )
        write_groundtruth(
            gt,
            query_count=spec.query_count,
            topk=spec.groundtruth_topk,
            ids=[2, 1, 1, 0],
            distances=[1.0, 81.0, 1.0, 81.0],
        )
        expect_validation_error(
            lambda: prep.exact_spot_check_groundtruth(
                spec,
                base=base,
                query=query,
                groundtruth=gt,
                sample_queries=2,
                seed=7,
                chunk_rows=2,
                tolerance=1e-4,
            ),
            "exact spot-check failed",
        )


def test_texmex_validation_and_safe_extract() -> None:
    spec = mock_texmex_spec()
    with tempfile.TemporaryDirectory(prefix="vortex-texmex-") as tmp:
        root = Path(tmp)
        staging = root / "staging" / "mock"
        staging.mkdir(parents=True)
        for file_spec in spec.files:
            write_texmex_vectors(
                staging / Path(file_spec.archive_member).name,
                count=file_spec.count,
                dim=file_spec.dim,
                kind=file_spec.kind,
            )

        archive = root / "mock.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            for file_spec in spec.files:
                tar.add(
                    staging / Path(file_spec.archive_member).name,
                    arcname=file_spec.archive_member,
                )

        output_root = root / "vector_datasets"
        extracted = prep.extract_texmex_archive(spec, archive, output_root)
        if len(extracted) != len(spec.files):
            raise AssertionError("not all TexMex members were extracted")
        validation = prep.validate_texmex_dataset(spec, output_root)
        if not validation["ok"]:
            raise AssertionError("valid TexMex fixture rejected")


def test_texmex_partial_archive_resumes_before_md5() -> None:
    final_payload = b"complete archive payload"
    spec = prep.TexMexSpec(
        name="mocktex",
        display_name="MockTex",
        role="test",
        url="ftp://example.invalid/mock.tar.gz",
        archive_bytes=len(final_payload),
        md5=hashlib.md5(final_payload, usedforsecurity=False).hexdigest(),
        metric="L2",
        files=(),
    )
    with tempfile.TemporaryDirectory(prefix="vortex-texmex-resume-") as tmp:
        source_root = Path(tmp)
        archive = prep.texmex_archive_path(spec, source_root)
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_bytes(final_payload[:4])

        calls = []
        original_run = prep.subprocess.run

        def fake_run(cmd, check):
            calls.append(cmd)
            archive.write_bytes(final_payload)
            return subprocess.CompletedProcess(cmd, 0)

        prep.subprocess.run = fake_run
        try:
            info = prep.ensure_texmex_archive(spec, source_root)
        finally:
            prep.subprocess.run = original_run

        if not calls:
            raise AssertionError("partial archive was not resumed")
        if info["bytes"] != len(final_payload):
            raise AssertionError("resumed archive byte size was not reported")


def test_texmex_groundtruth_id_bounds() -> None:
    spec = prep.TexMexFileSpec("mock/gt.ivecs", "gt.ivecs", 1, 2, "ivecs", id_bound=2)
    with tempfile.TemporaryDirectory(prefix="vortex-texmex-") as tmp:
        path = Path(tmp) / "gt.ivecs"
        payload = struct.pack("<i", 2) + struct.pack("<2i", 0, 2)
        path.write_bytes(payload)
        expect_validation_error(
            lambda: prep.validate_texmex_vector_file(path, spec),
            "id out of bounds",
        )


def main() -> int:
    test_patch_prefix_header()
    test_validate_bundle_and_query_match()
    test_groundtruth_safety_checks()
    test_real_specs_encode_validated_ranges()
    test_workload_contracts_are_l2_canonical_and_exact()
    test_dataset_profiles_expose_benchmark_safe_layout()
    test_materialize_prefix_dataset_and_spot_check()
    test_convert_materialized_dataset_to_vortex_formats()
    test_exact_spot_check_rejects_wrong_slice_groundtruth()
    test_texmex_validation_and_safe_extract()
    test_texmex_partial_archive_resumes_before_md5()
    test_texmex_groundtruth_id_bounds()
    print("vortex dataset prepare checks: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
