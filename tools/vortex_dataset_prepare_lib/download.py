"""Download, range-crop, and archive-extraction helpers."""

from __future__ import annotations

import subprocess
import tarfile
from pathlib import Path

from .binary_io import md5_file, patch_prefix_header
from .file_ops import copy_file
from .paths import (
    dataset_source_dir,
    expected_groundtruth_bytes,
    expected_query_bytes,
    materialized_base_path,
    materialized_groundtruth_path,
    materialized_query_path,
    source_base_path,
    source_groundtruth_path,
    source_query_path,
    texmex_archive_path,
    texmex_file_path,
    texmex_output_dir,
)
from .specs import DatasetSpec, TexMexSpec, ValidationError
from .validation import validate_texmex_vector_file


def curl_commands(spec: DatasetSpec, output_root: Path) -> list[str]:
    source_dir = output_root / "_sources" / spec.source_family
    base_suffix = {
        "uint8": "u8bin",
        "int8": "i8bin",
        "float32": "fbin",
    }[spec.base_dtype]
    return [
        (
            f"curl -L --fail --retry 10 -r {spec.byte_range} "
            f"-o {source_dir / ('base.10M.' + base_suffix)} {spec.base_url}"
        ),
        f"curl -L --fail --retry 10 -o {source_dir / Path(spec.query_url).name} {spec.query_url}",
        f"curl -L --fail --retry 10 -o {source_dir / Path(spec.groundtruth_url).name} {spec.groundtruth_url}",
    ]


def ensure_curl_file(
    url: str,
    output: Path,
    *,
    expected_bytes: int,
    force_download: bool = False,
) -> dict[str, object]:
    output.parent.mkdir(parents=True, exist_ok=True)
    status = "existing"
    if force_download and output.exists():
        output.unlink()
    if output.exists():
        actual_size = output.stat().st_size
        if actual_size == expected_bytes:
            return {
                "path": str(output),
                "url": url,
                "bytes": actual_size,
                "expected_bytes": expected_bytes,
                "status": status,
            }
        if actual_size > expected_bytes:
            raise ValidationError(
                f"{output}: file is larger than expected; expected {expected_bytes}, "
                f"found {actual_size}. Re-run with --force-download to replace it."
            )
        status = "resumed"
    else:
        status = "downloaded"
    run_curl(url, output, force=False)
    actual_size = output.stat().st_size
    if actual_size != expected_bytes:
        raise ValidationError(
            f"{output}: wrong byte size after download; expected {expected_bytes}, "
            f"found {actual_size}. Re-run the same command to resume or use "
            "--force-download to replace it."
        )
    return {
        "path": str(output),
        "url": url,
        "bytes": actual_size,
        "expected_bytes": expected_bytes,
        "status": status,
    }


def run_curl_range(
    url: str,
    output: Path,
    *,
    byte_range: str,
    expected_bytes: int,
    force_download: bool = False,
) -> dict[str, object]:
    output.parent.mkdir(parents=True, exist_ok=True)
    range_start_s, range_end_s = byte_range.split("-", maxsplit=1)
    range_start = int(range_start_s)
    range_end = int(range_end_s)
    if range_start != 0 or range_end + 1 != expected_bytes:
        raise ValidationError(
            f"unsupported range {byte_range}; expected 0-{expected_bytes - 1}"
        )
    if force_download and output.exists():
        output.unlink()
    existing = output.stat().st_size if output.exists() else 0
    if existing == expected_bytes:
        return {
            "path": str(output),
            "url": url,
            "byte_range": byte_range,
            "bytes": existing,
            "expected_bytes": expected_bytes,
            "status": "existing",
        }
    if existing > expected_bytes:
        raise ValidationError(
            f"{output}: prefix file is larger than expected; expected {expected_bytes}, "
            f"found {existing}. Re-run with --force-download to replace it."
        )

    resume_start = range_start + existing
    resume_range = f"{resume_start}-{range_end}"
    tmp = output.with_name(output.name + ".part")
    if tmp.exists():
        tmp.unlink()
    cmd = [
        "curl",
        "-L",
        "--fail",
        "--retry",
        "10",
        "--retry-delay",
        "2",
        "-r",
        resume_range,
        "-o",
        str(tmp),
        url,
    ]
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise ValidationError(
            f"range download failed with exit code {result.returncode}: {' '.join(cmd)}"
        )
    remaining = expected_bytes - existing
    downloaded = tmp.stat().st_size
    if downloaded != remaining:
        tmp.unlink(missing_ok=True)
        raise ValidationError(
            f"{url}: range server returned {downloaded} bytes for {resume_range}; "
            f"expected {remaining}. Refusing to append an unsafe prefix."
        )
    with output.open("ab") as dst, tmp.open("rb") as src:
        while True:
            chunk = src.read(8 * 1024 * 1024)
            if not chunk:
                break
            dst.write(chunk)
    tmp.unlink(missing_ok=True)
    actual_size = output.stat().st_size
    if actual_size != expected_bytes:
        raise ValidationError(
            f"{output}: wrong prefix byte size after range download; expected "
            f"{expected_bytes}, found {actual_size}"
        )
    return {
        "path": str(output),
        "url": url,
        "byte_range": byte_range,
        "bytes": actual_size,
        "expected_bytes": expected_bytes,
        "status": "resumed" if existing else "downloaded",
    }


def materialize_prefix_base(
    spec: DatasetSpec,
    output_root: Path,
    *,
    force_download: bool,
) -> dict[str, object]:
    source_path = source_base_path(spec, output_root)
    materialized_path = materialized_base_path(spec, output_root)
    download = run_curl_range(
        spec.base_url,
        source_path,
        byte_range=spec.byte_range,
        expected_bytes=spec.prefix_bytes,
        force_download=force_download,
    )
    copy_file(source_path, materialized_path)
    patch = patch_prefix_header(spec, materialized_path)
    return {
        "source": download,
        "path": str(materialized_path),
        "patch": patch,
    }


def materialize_query_and_groundtruth(
    spec: DatasetSpec,
    output_root: Path,
    *,
    force_download: bool,
) -> dict[str, object]:
    source_query = source_query_path(spec, output_root)
    source_gt = source_groundtruth_path(spec, output_root)
    query_download = ensure_curl_file(
        spec.query_url,
        source_query,
        expected_bytes=expected_query_bytes(spec),
        force_download=force_download,
    )
    gt_download = ensure_curl_file(
        spec.groundtruth_url,
        source_gt,
        expected_bytes=expected_groundtruth_bytes(spec),
        force_download=force_download,
    )
    materialized_query = materialized_query_path(spec, output_root)
    materialized_gt = materialized_groundtruth_path(spec, output_root)
    copy_file(source_query, materialized_query)
    copy_file(source_gt, materialized_gt)
    return {
        "query": query_download | {"path": str(materialized_query)},
        "groundtruth": gt_download | {"path": str(materialized_gt)},
        "source_query": str(source_query),
        "source_groundtruth": str(source_gt),
    }


def run_curl(url: str, output: Path, *, force: bool = False) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and force:
        output.unlink()
    cmd = [
        "curl",
        "-L",
        "--fail",
        "--retry",
        "10",
        "--retry-delay",
        "2",
        "-C",
        "-",
        "-o",
        str(output),
        url,
    ]
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise ValidationError(
            f"download failed with exit code {result.returncode}: {' '.join(cmd)}"
        )


def ensure_texmex_archive(
    spec: TexMexSpec, source_root: Path, *, force_download: bool = False
) -> dict[str, object]:
    archive = texmex_archive_path(spec, source_root)
    if force_download or not archive.exists():
        run_curl(spec.url, archive, force=force_download)
    else:
        actual_size = archive.stat().st_size
        if actual_size < spec.archive_bytes:
            run_curl(spec.url, archive, force=False)
        elif actual_size > spec.archive_bytes:
            raise ValidationError(
                f"{archive}: archive is larger than expected; expected {spec.archive_bytes}, "
                f"found {actual_size}. Re-run with --force-download to replace it."
            )
        else:
            actual_md5 = md5_file(archive)
            if actual_md5 != spec.md5:
                raise ValidationError(
                    f"{archive}: MD5 mismatch before download; expected {spec.md5}, "
                    f"found {actual_md5}. Re-run with --force-download to replace it."
                )
    actual_size = archive.stat().st_size
    if actual_size != spec.archive_bytes:
        raise ValidationError(
            f"{archive}: wrong archive byte size after download; expected {spec.archive_bytes}, "
            f"found {actual_size}. "
            "The download may still be incomplete; re-run the same command to resume."
        )
    actual_md5 = md5_file(archive)
    if actual_md5 != spec.md5:
        raise ValidationError(
            f"{archive}: MD5 mismatch; expected {spec.md5}, found {actual_md5}. "
            "Re-run with --force-download to replace it."
        )
    return {
        "path": str(archive),
        "url": spec.url,
        "md5": actual_md5,
        "bytes": actual_size,
        "expected_bytes": spec.archive_bytes,
    }


def safe_extract_member(
    tar: tarfile.TarFile, member_name: str, destination: Path
) -> None:
    member = tar.getmember(member_name)
    if not member.isfile():
        raise ValidationError(f"archive member is not a regular file: {member_name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    src = tar.extractfile(member)
    if src is None:
        raise ValidationError(f"could not open archive member: {member_name}")
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    with src, tmp.open("wb") as out:
        while True:
            chunk = src.read(8 * 1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    tmp.replace(destination)


def extract_texmex_archive(
    spec: TexMexSpec,
    archive: Path,
    output_root: Path,
    *,
    force_extract: bool = False,
) -> list[dict[str, object]]:
    output_dir = texmex_output_dir(spec, output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted = []
    with tarfile.open(archive, "r:gz") as tar:
        names = set(tar.getnames())
        for file_spec in spec.files:
            if file_spec.archive_member not in names:
                raise ValidationError(
                    f"{archive}: missing expected member {file_spec.archive_member}"
                )
            destination = texmex_file_path(spec, output_root, file_spec)
            if destination.exists() and not force_extract:
                validate_texmex_vector_file(destination, file_spec, scan_headers=False)
                extracted.append({"path": str(destination), "status": "existing"})
                continue
            safe_extract_member(tar, file_spec.archive_member, destination)
            extracted.append({"path": str(destination), "status": "extracted"})
    return extracted
