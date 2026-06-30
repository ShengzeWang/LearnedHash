"""Filesystem helpers for streaming dataset materialization."""

from __future__ import annotations

from pathlib import Path

from .specs import ValidationError


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.name + ".tmp")
    with source.open("rb") as src, tmp.open("wb") as dst:
        while True:
            chunk = src.read(8 * 1024 * 1024)
            if not chunk:
                break
            dst.write(chunk)
    tmp.replace(destination)


def ensure_convertible_output(
    path: Path,
    *,
    expected_bytes: int,
    force: bool,
    label: str,
) -> bool:
    """Return True when an existing output can be reused."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        return False
    actual = path.stat().st_size
    if actual == expected_bytes and not force:
        return True
    if not force:
        raise ValidationError(
            f"{path}: existing {label} has wrong byte size; expected {expected_bytes}, "
            f"found {actual}. Re-run with --force-convert to replace it."
        )
    path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    return False
