"""Shared path-name checks for generated local artifacts.

The repository intentionally keeps generated datasets, benchmark outputs, and
codegen caches out of source control. These helpers keep those runtime-created
paths portable across common macOS, Linux, and Windows checkouts.
"""

from __future__ import annotations

import re
from pathlib import Path


INVALID_COMPONENT_CHARS = frozenset('"*:<>?/\\|')
RESERVED_WINDOWS_NAMES = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        "COM0",
        "COM1",
        "COM2",
        "COM3",
        "COM4",
        "COM5",
        "COM6",
        "COM7",
        "COM8",
        "COM9",
        "LPT0",
        "LPT1",
        "LPT2",
        "LPT3",
        "LPT4",
        "LPT5",
        "LPT6",
        "LPT7",
        "LPT8",
        "LPT9",
    }
)
BAD_SYNC_TOKENS = (
    "conflicted copy",
    "conflict copy",
    "shengze",
    "laptop",
)
MAX_SEGMENT_CHARS = 255
MAX_SAFE_NAME_CHARS = 128
MAX_SAFE_RELATIVE_CHARS = 240
MAX_SYNC_ABSOLUTE_CHARS = 400


class PathSafetyError(ValueError):
    """Raised when an output path is likely to fail on synced/Windows paths."""


def _has_control_character(value: str) -> bool:
    return any(ord(char) < 32 for char in value)


def _reserved_stem(component: str) -> str:
    return component.split(".", 1)[0].upper()


def component_issues(component: str, *, max_chars: int = MAX_SEGMENT_CHARS) -> list[str]:
    issues: list[str] = []
    if component in {"", ".", ".."}:
        issues.append("component is empty or relative")
    if len(component) > max_chars:
        issues.append(f"segment exceeds {max_chars} characters")
    if any(char in INVALID_COMPONENT_CHARS for char in component):
        issues.append("component contains a cross-platform invalid character")
    if _has_control_character(component):
        issues.append("component contains a control character")
    if component.endswith(" ") or component.endswith("."):
        issues.append("component ends with a space or dot")
    stem = _reserved_stem(component)
    if stem in RESERVED_WINDOWS_NAMES:
        issues.append(f"component uses reserved Windows name {stem}")
    lowered = component.lower()
    if any(token in lowered for token in BAD_SYNC_TOKENS):
        issues.append("component looks like a sync/conflict copy name")
    return issues


def validate_component(
    value: str,
    *,
    label: str,
    max_chars: int = MAX_SAFE_NAME_CHARS,
) -> str:
    if value != Path(value).name:
        raise PathSafetyError(f"{label} must be a single path component: {value!r}")
    issues = component_issues(value, max_chars=max_chars)
    if issues:
        raise PathSafetyError(f"{label} is not path-safe: {value!r}: {'; '.join(issues)}")
    return value


def safe_component(
    value: str,
    *,
    fallback: str = "artifact",
    max_chars: int = MAX_SAFE_NAME_CHARS,
) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._- ")
    if not cleaned:
        cleaned = fallback
    if any(token in cleaned.lower() for token in BAD_SYNC_TOKENS):
        cleaned = fallback
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rstrip("._- ")
    if not cleaned:
        cleaned = fallback
    if _reserved_stem(cleaned) in RESERVED_WINDOWS_NAMES:
        cleaned = f"{fallback}_{cleaned}"
    return cleaned


def _effective_absolute_path(path: Path, root: Path | None) -> Path:
    if path.is_absolute():
        return path.expanduser().resolve(strict=False)
    base = root.resolve(strict=False) if root is not None else Path.cwd()
    return (base / path).expanduser().resolve(strict=False)


def _relative_to_root(path: Path, root: Path | None) -> str | None:
    if root is None:
        return None
    try:
        return str(path.relative_to(root.resolve(strict=False)))
    except ValueError:
        return None


def path_issues(
    path: Path,
    *,
    root: Path | None = None,
    max_relative_chars: int = MAX_SAFE_RELATIVE_CHARS,
    max_absolute_chars: int = MAX_SYNC_ABSOLUTE_CHARS,
) -> list[str]:
    absolute = _effective_absolute_path(path, root)
    issues: list[str] = []
    relative = _relative_to_root(absolute, root)
    if relative is not None and len(relative) > max_relative_chars:
        issues.append(f"relative path exceeds {max_relative_chars} characters")
    absolute_text = str(absolute)
    if len(absolute_text) > max_absolute_chars:
        issues.append(f"absolute path exceeds {max_absolute_chars} characters")
    for component in absolute.parts:
        if component in {absolute.anchor, ""}:
            continue
        issues.extend(component_issues(component))
    return issues


def relative_path_issues(
    rel: str,
    absolute_root: Path,
    *,
    max_relative_chars: int = MAX_SAFE_RELATIVE_CHARS,
    max_absolute_chars: int = MAX_SYNC_ABSOLUTE_CHARS,
) -> list[str]:
    issues: list[str] = []
    if len(rel) > max_relative_chars:
        issues.append(f"relative path exceeds {max_relative_chars} characters")
    absolute_len = len(str(absolute_root / rel))
    if absolute_len > max_absolute_chars:
        issues.append(f"absolute path exceeds {max_absolute_chars} characters")
    for component in Path(rel).parts:
        issues.extend(component_issues(component))
    return issues


def validate_output_path(
    path: Path,
    *,
    root: Path | None,
    label: str,
    max_relative_chars: int = MAX_SAFE_RELATIVE_CHARS,
    max_absolute_chars: int = MAX_SYNC_ABSOLUTE_CHARS,
) -> Path:
    issues = path_issues(
        path,
        root=root,
        max_relative_chars=max_relative_chars,
        max_absolute_chars=max_absolute_chars,
    )
    if issues:
        raise PathSafetyError(f"{label} is not path-safe: {path}: {'; '.join(issues)}")
    return _effective_absolute_path(path, root)
