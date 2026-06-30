"""Subprocess execution and log-capture helpers."""

from __future__ import annotations

import os
import platform
import re
import shlex
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from .common import ROOT, safe_label


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if value == "true":
        return True
    if value == "false":
        return False
    try:
        if re.fullmatch(r"[-+]?\d+", value):
            return int(value)
        if re.fullmatch(r"[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?", value):
            return float(value)
    except ValueError:
        pass
    return value

def parse_key_value_output(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for line in text.splitlines():
        match = re.match(r"\s*([A-Za-z0-9_@().-]+):\s*(.*)\s*$", line)
        if match:
            out[match.group(1)] = parse_scalar(match.group(2))
    return out

def time_command_prefix() -> tuple[list[str], str | None]:
    time_bin = Path("/usr/bin/time")
    if not time_bin.exists():
        return [], None
    system = platform.system()
    if system == "Darwin":
        return [str(time_bin), "-l"], "darwin"
    if system == "Linux":
        return [str(time_bin), "-v"], "gnu"
    return [], None

def parse_time_peak_rss_kib(text: str, flavor: str | None) -> int | None:
    if flavor == "darwin":
        for line in text.splitlines():
            match = re.match(r"\s*(\d+)\s+maximum resident set size", line)
            if match:
                return max(1, (int(match.group(1)) + 1023) // 1024)
    if flavor == "gnu":
        for line in text.splitlines():
            match = re.match(r"\s*Maximum resident set size .*:\s*(\d+)\s*$", line)
            if match:
                return int(match.group(1))
    return None

def process_children(pid: int) -> list[int]:
    if shutil.which("pgrep") is None:
        return []
    try:
        proc = subprocess.run(
            ["pgrep", "-P", str(pid)],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return []
    children = []
    for line in proc.stdout.splitlines():
        try:
            children.append(int(line.strip()))
        except ValueError:
            continue
    return children

def process_tree(pid: int) -> list[int]:
    seen: set[int] = set()
    stack = [pid]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(process_children(cur))
    return sorted(seen)

def rss_kib(pid: int) -> int:
    try:
        proc = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return 0
    total = 0
    for line in proc.stdout.splitlines():
        try:
            total += int(line.strip())
        except ValueError:
            continue
    return total

def sample_peak_rss(root_pid: int, stop: threading.Event, peak: list[int]) -> None:
    while not stop.is_set():
        current = 0
        for pid in process_tree(root_pid):
            current += rss_kib(pid)
        if current > peak[0]:
            peak[0] = current
        stop.wait(0.1)

def terminate_tree(pid: int) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for child in reversed(process_tree(pid)):
            try:
                os.kill(child, sig)
            except OSError:
                pass
        time.sleep(0.5)

def run_command(
    label: str,
    argv: list[str],
    logs_dir: Path,
    timeout: int | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{safe_label(label)}.log"
    time_prefix, time_flavor = time_command_prefix()
    exec_argv = [*time_prefix, *argv]
    started = time.perf_counter()
    proc = subprocess.Popen(
        exec_argv,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )
    peak = [rss_kib(proc.pid)]
    stop = threading.Event()
    sampler = threading.Thread(target=sample_peak_rss, args=(proc.pid, stop, peak), daemon=True)
    sampler.start()
    timed_out = False
    try:
        stdout, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        terminate_tree(proc.pid)
        stdout, _ = proc.communicate()
    finally:
        stop.set()
        sampler.join(timeout=1.0)
    wall_seconds = time.perf_counter() - started
    command_line = shlex.join(argv)
    measured_line = shlex.join(exec_argv) if time_prefix else command_line
    log_path.write_text(f"$ {command_line}\n# measured_by: {measured_line}\n\n{stdout}",
                        errors="replace")
    time_peak = parse_time_peak_rss_kib(stdout, time_flavor)
    peak_rss = time_peak if time_peak is not None else (peak[0] or None)
    result = {
        "label": label,
        "command": argv,
        "command_line": command_line,
        "returncode": proc.returncode,
        "wall_seconds": wall_seconds,
        "peak_rss_kib": peak_rss,
        "log": str(log_path),
        "timed_out": timed_out,
    }
    if proc.returncode != 0 or timed_out:
        status = "timed out" if timed_out else f"exited {proc.returncode}"
        raise RuntimeError(f"{label} {status}; see {log_path}")
    result["parsed_output"] = parse_key_value_output(stdout)
    return result
