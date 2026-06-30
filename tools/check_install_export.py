#!/usr/bin/env python3
"""Smoke-test the installed LearnedHash CMake package.

The check installs an existing build tree, configures a tiny external consumer
with find_package(LearnedHash CONFIG), builds separate LEAD and Vortex consumers,
and runs them. It catches package-export regressions that in-tree builds cannot
observe.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


CONSUMER_CMAKE = r"""
cmake_minimum_required(VERSION 3.16)
project(LearnedHashInstallConsumer LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(LearnedHash CONFIG REQUIRED)

add_executable(rm_lead_consumer rm_lead.cpp)
target_link_libraries(rm_lead_consumer PRIVATE LearnedHash::lead_core)

if (TARGET LearnedHash::vortex_v1_core)
  add_executable(vortex_consumer vortex_core.cpp)
  target_link_libraries(vortex_consumer PRIVATE LearnedHash::vortex_v1_core)
else()
  add_executable(vortex_consumer vortex_runtime.cpp)
  target_link_libraries(vortex_consumer PRIVATE LearnedHash::vortex_v1_runtime)
endif()
"""


RM_LEAD_CONSUMER = r"""
#include <iostream>

#include "rm_model/uint256.h"
#include "lead/hash.h"

int main() {
  const auto value = rm_model::UInt256::from_u64(42);
  lead::HashOutput out;
  out.value = lead::UInt256::from_u64(7);
  out.bits = 64;
  std::cout << rm_model::to_hex(value, 64) << " " << out.to_hex() << "\n";
  return 0;
}
"""


VORTEX_CORE_CONSUMER = r"""
#include <iostream>

#include "vortex_v1/model.h"
#include "vortex_v1/training.h"

int main() {
  vortex::VortexModel model;
  model.dim = 128;
  std::cout << model.dim << "\n";
  return 0;
}
"""


VORTEX_RUNTIME_CONSUMER = r"""
#include <iostream>

#include "vortex_v1/codegen.h"
#include "vortex_v1/model.h"

int main() {
  vortex::VortexModel model;
  model.dim = 128;
  vortex::CodegenOptions options;
  options.name = "install_export_smoke";
  std::cout << model.dim << " " << options.name << "\n";
  return 0;
}
"""


def run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(cmd), flush=True)
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {result.returncode}: {' '.join(cmd)}"
        )
    return result


def executable_path(build_dir: Path, name: str, config: str) -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    candidates = [
        build_dir / f"{name}{suffix}",
        build_dir / config / f"{name}{suffix}",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"built executable not found: {name}")


def write_consumer_project(source_dir: Path) -> None:
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "CMakeLists.txt").write_text(
        CONSUMER_CMAKE.lstrip(), encoding="utf-8"
    )
    (source_dir / "rm_lead.cpp").write_text(RM_LEAD_CONSUMER.lstrip(), encoding="utf-8")
    (source_dir / "vortex_core.cpp").write_text(
        VORTEX_CORE_CONSUMER.lstrip(), encoding="utf-8"
    )
    (source_dir / "vortex_runtime.cpp").write_text(
        VORTEX_RUNTIME_CONSUMER.lstrip(), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path("build"),
        help="Existing LearnedHash build directory to install (default: build).",
    )
    parser.add_argument(
        "--install-prefix",
        type=Path,
        help="Install prefix. If omitted, a temporary prefix is created.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="External consumer work directory. If omitted, a temporary directory is created.",
    )
    parser.add_argument(
        "--config", default="Release", help="CMake build/install config."
    )
    parser.add_argument("--cmake", default="cmake", help="CMake executable.")
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary install/work directories for debugging.",
    )
    parser.add_argument(
        "--force-clean",
        action="store_true",
        help="Remove explicit install/work directories before use.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_dir = args.build_dir.resolve()
    if not build_dir.exists():
        print(f"build directory does not exist: {build_dir}", file=sys.stderr)
        return 2

    temp_dirs: list[str] = []
    try:
        if args.install_prefix is None:
            install_prefix = Path(
                tempfile.mkdtemp(prefix="learnedhash-install-")
            ).resolve()
            temp_dirs.append(str(install_prefix))
        else:
            install_prefix = args.install_prefix.resolve()
            if install_prefix.exists():
                if not args.force_clean:
                    print(
                        f"install prefix already exists; pass --force-clean to remove it: {install_prefix}",
                        file=sys.stderr,
                    )
                    return 2
                shutil.rmtree(install_prefix)

        if args.work_dir is None:
            work_dir = Path(tempfile.mkdtemp(prefix="learnedhash-consumer-")).resolve()
            temp_dirs.append(str(work_dir))
        else:
            work_dir = args.work_dir.resolve()
            if work_dir.exists():
                if not args.force_clean:
                    print(
                        f"work directory already exists; pass --force-clean to remove it: {work_dir}",
                        file=sys.stderr,
                    )
                    return 2
                shutil.rmtree(work_dir)

        install_cmd = [
            args.cmake,
            "--install",
            str(build_dir),
            "--prefix",
            str(install_prefix),
        ]
        if args.config:
            install_cmd.extend(["--config", args.config])
        run(install_cmd)

        source_dir = work_dir / "src"
        consumer_build_dir = work_dir / "build"
        write_consumer_project(source_dir)

        run(
            [
                args.cmake,
                "-S",
                str(source_dir),
                "-B",
                str(consumer_build_dir),
                "-DCMAKE_BUILD_TYPE=Release",
                f"-DCMAKE_PREFIX_PATH={install_prefix}",
            ]
        )
        build_cmd = [args.cmake, "--build", str(consumer_build_dir), "--parallel", "1"]
        if args.config:
            build_cmd.extend(["--config", args.config])
        run(build_cmd)

        run([str(executable_path(consumer_build_dir, "rm_lead_consumer", args.config))])
        run([str(executable_path(consumer_build_dir, "vortex_consumer", args.config))])

        print("LearnedHash install/export smoke: ok")
        if args.keep_temp or args.install_prefix is not None:
            print(f"Install prefix: {install_prefix}")
        if args.keep_temp or args.work_dir is not None:
            print(f"Consumer work dir: {work_dir}")
        if temp_dirs and not args.keep_temp:
            print("Temporary install/work directories cleaned.")
        return 0
    except Exception as exc:
        print(f"LearnedHash install/export smoke failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if not args.keep_temp:
            for path in temp_dirs:
                shutil.rmtree(path, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
