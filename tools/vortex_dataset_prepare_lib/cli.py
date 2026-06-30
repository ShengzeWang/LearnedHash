"""Command-line interface for the Vortex dataset preparation helper."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .binary_io import patch_prefix_header
from .contracts import dataset_profile_rows, validate_workload_contracts
from .download import curl_commands
from .manifest import manifest, texmex_manifest
from .materialize import (
    convert_materialized_dataset,
    materialize_dataset,
    materialize_texmex,
)
from .specs import DATASETS, TEXMEX_DATASETS, DatasetSpec, TexMexSpec, ValidationError
from .validation import (
    validate_bundle,
    validate_converted_dataset,
    validate_materialized_dataset,
    validate_texmex_dataset,
)


def texmex_arg(value: str) -> TexMexSpec:
    key = value.lower()
    if key not in TEXMEX_DATASETS:
        raise argparse.ArgumentTypeError(
            f"unknown TexMex dataset {value}; choose one of {', '.join(sorted(TEXMEX_DATASETS))}"
        )
    return TEXMEX_DATASETS[key]


def dataset_arg(value: str) -> DatasetSpec:
    key = value.lower()
    if key not in DATASETS:
        raise argparse.ArgumentTypeError(
            f"unknown dataset {value}; choose one of {', '.join(sorted(DATASETS))}"
        )
    return DATASETS[key]


def add_dataset_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset", type=dataset_arg, required=True, help="10M dataset profile"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="List supported 10M prefix dataset specs.")
    p_list.add_argument("--json", action="store_true", help="Emit JSON.")

    p_profiles = sub.add_parser(
        "profiles",
        help="List all Vortex dataset profiles and local vector_datasets layout.",
    )
    p_profiles.add_argument("--json", action="store_true", help="Emit JSON.")

    p_contract = sub.add_parser(
        "contract-check",
        help="Verify L2 metric, canonical split, and exact-GT contracts.",
    )
    p_contract.add_argument("--json", action="store_true", help="Emit JSON.")

    p_commands = sub.add_parser(
        "range-commands", help="Print official range-download commands."
    )
    add_dataset_arg(p_commands)
    p_commands.add_argument("--output-root", type=Path, default=Path("vector_datasets"))

    p_manifest = sub.add_parser(
        "manifest", help="Emit a manifest template with optional SHA-256 values."
    )
    add_dataset_arg(p_manifest)
    p_manifest.add_argument("--base", type=Path)
    p_manifest.add_argument("--query", type=Path)
    p_manifest.add_argument("--groundtruth", type=Path)

    p_patch = sub.add_parser(
        "patch-xbin-header",
        help="Patch a cropped 10M base header after range download.",
    )
    add_dataset_arg(p_patch)
    p_patch.add_argument("--base", type=Path, required=True)
    p_patch.add_argument("--dry-run", action="store_true")

    p_validate = sub.add_parser(
        "validate", help="Validate a base/query/ground-truth triple."
    )
    add_dataset_arg(p_validate)
    p_validate.add_argument("--base", type=Path, required=True)
    p_validate.add_argument("--query", type=Path, required=True)
    p_validate.add_argument("--groundtruth", type=Path, required=True)
    p_validate.add_argument("--skip-groundtruth-scan", action="store_true")

    p_validate_materialized = sub.add_parser(
        "validate-materialized",
        help="Validate a locally materialized 10M prefix dataset.",
    )
    add_dataset_arg(p_validate_materialized)
    p_validate_materialized.add_argument(
        "--output-root", type=Path, default=Path("vector_datasets")
    )
    p_validate_materialized.add_argument("--skip-groundtruth-scan", action="store_true")
    p_validate_materialized.add_argument(
        "--spot-check-queries",
        type=int,
        default=0,
        help="Run an exact L2 spot check for N fixed-seed queries.",
    )
    p_validate_materialized.add_argument(
        "--spot-check-seed", type=int, default=20260101
    )
    p_validate_materialized.add_argument(
        "--spot-check-base-chunk",
        type=int,
        default=65_536,
        help="Base rows per NumPy exact-search chunk.",
    )
    p_validate_materialized.add_argument(
        "--spot-check-tolerance",
        type=float,
        default=1e-4,
        help="Absolute tolerance for exact squared-L2 tie checks.",
    )

    p_materialize = sub.add_parser(
        "materialize",
        help="Download, crop, patch, validate, and manifest a 10M prefix dataset.",
    )
    add_dataset_arg(p_materialize)
    p_materialize.add_argument(
        "--output-root", type=Path, default=Path("vector_datasets")
    )
    p_materialize.add_argument("--force-download", action="store_true")
    p_materialize.add_argument("--skip-groundtruth-scan", action="store_true")
    p_materialize.add_argument("--include-file-sha256", action="store_true")
    p_materialize.add_argument(
        "--spot-check-queries",
        type=int,
        default=0,
        help="Run an exact L2 spot check for N fixed-seed queries after validation.",
    )
    p_materialize.add_argument("--spot-check-seed", type=int, default=20260101)
    p_materialize.add_argument(
        "--spot-check-base-chunk",
        type=int,
        default=65_536,
        help="Base rows per NumPy exact-search chunk.",
    )
    p_materialize.add_argument(
        "--spot-check-tolerance",
        type=float,
        default=1e-4,
        help="Absolute tolerance for exact squared-L2 tie checks.",
    )

    p_validate_converted = sub.add_parser(
        "validate-converted",
        help="Validate Vortex .fvecs/.ivecs outputs for a materialized 10M dataset.",
    )
    add_dataset_arg(p_validate_converted)
    p_validate_converted.add_argument(
        "--output-root", type=Path, default=Path("vector_datasets")
    )
    p_validate_converted.add_argument("--skip-header-scan", action="store_true")

    p_convert = sub.add_parser(
        "convert",
        help="Stream-convert validated 10M xbin/fbin/i8bin files to Vortex vecs.",
    )
    add_dataset_arg(p_convert)
    p_convert.add_argument("--output-root", type=Path, default=Path("vector_datasets"))
    p_convert.add_argument("--force-convert", action="store_true")
    p_convert.add_argument("--include-file-sha256", action="store_true")
    p_convert.add_argument(
        "--conversion-chunk-rows",
        type=int,
        default=65_536,
        help="Rows per streaming conversion chunk.",
    )
    p_convert.add_argument("--skip-converted-header-scan", action="store_true")

    p_texmex_list = sub.add_parser(
        "texmex-list", help="List supported TexMex dataset specs."
    )
    p_texmex_list.add_argument("--json", action="store_true", help="Emit JSON.")

    p_texmex_manifest = sub.add_parser(
        "texmex-manifest",
        help="Emit a TexMex manifest template, using local file checksums when present.",
    )
    p_texmex_manifest.add_argument("--dataset", type=texmex_arg, required=True)
    p_texmex_manifest.add_argument(
        "--output-root", type=Path, default=Path("vector_datasets")
    )
    p_texmex_manifest.add_argument("--include-file-sha256", action="store_true")

    p_texmex_validate = sub.add_parser(
        "texmex-validate", help="Validate an existing materialized TexMex dataset."
    )
    p_texmex_validate.add_argument("--dataset", type=texmex_arg, required=True)
    p_texmex_validate.add_argument(
        "--output-root", type=Path, default=Path("vector_datasets")
    )
    p_texmex_validate.add_argument("--skip-header-scan", action="store_true")

    p_texmex_materialize = sub.add_parser(
        "texmex-materialize",
        help="Download, extract, validate, and manifest an official TexMex dataset.",
    )
    p_texmex_materialize.add_argument("--dataset", type=texmex_arg, required=True)
    p_texmex_materialize.add_argument(
        "--source-root",
        type=Path,
        default=Path("vector_datasets") / "_sources" / "texmex",
    )
    p_texmex_materialize.add_argument(
        "--output-root", type=Path, default=Path("vector_datasets")
    )
    p_texmex_materialize.add_argument("--force-download", action="store_true")
    p_texmex_materialize.add_argument("--force-extract", action="store_true")
    p_texmex_materialize.add_argument("--skip-header-scan", action="store_true")
    p_texmex_materialize.add_argument("--include-file-sha256", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "list":
            rows = [
                asdict(spec)
                | {
                    "byte_range": spec.byte_range,
                    "expected_prefix_bytes": spec.prefix_bytes,
                }
                for spec in DATASETS.values()
            ]
            if args.json:
                print(json.dumps(rows, indent=2, sort_keys=True))
            else:
                for row in rows:
                    print(
                        f"{row['name']}: dim={row['dim']} dtype={row['base_dtype']} "
                        f"base={row['base_count']} query={row['query_count']} "
                        f"metric={row['metric']} range={row['byte_range']} "
                        f"gt={row['groundtruth_url']}"
                    )
            return 0
        if args.command == "profiles":
            rows = dataset_profile_rows()
            if args.json:
                print(json.dumps(rows, indent=2, sort_keys=True))
            else:
                for row in rows:
                    manifest = (
                        "required" if row["manifest_required"] else "optional"
                    )
                    print(
                        f"{row['profile']}: {row['display_name']} "
                        f"family={row['family']} dim={row['dim']} "
                        f"base={row['base_count']} query={row['query_count']} "
                        f"metric={row['metric']} manifest={manifest}"
                    )
            return 0
        if args.command == "contract-check":
            result = validate_workload_contracts()
            if args.json:
                print(json.dumps(result, indent=2, sort_keys=True))
            else:
                for row in result["workloads"]:
                    print(
                        f"{row['profile']}: metric={row['metric']} "
                        f"base_split={row['base_split']} "
                        f"query_split={row['query_split']} "
                        f"gt={row['groundtruth_scope']}"
                    )
                if result["errors"]:
                    for error in result["errors"]:
                        print(f"error: {error}", file=sys.stderr)
            return 0 if result["ok"] else 1
        if args.command == "range-commands":
            print("\n".join(curl_commands(args.dataset, args.output_root)))
            return 0
        if args.command == "manifest":
            print(
                json.dumps(
                    manifest(args.dataset, args.base, args.query, args.groundtruth),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "patch-xbin-header":
            print(
                json.dumps(
                    patch_prefix_header(args.dataset, args.base, dry_run=args.dry_run),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "validate":
            print(
                json.dumps(
                    validate_bundle(
                        args.dataset,
                        base=args.base,
                        query=args.query,
                        groundtruth=args.groundtruth,
                        scan_groundtruth=not args.skip_groundtruth_scan,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "validate-materialized":
            print(
                json.dumps(
                    validate_materialized_dataset(
                        args.dataset,
                        args.output_root,
                        scan_groundtruth=not args.skip_groundtruth_scan,
                        spot_check_queries=args.spot_check_queries,
                        spot_check_seed=args.spot_check_seed,
                        spot_check_base_chunk=args.spot_check_base_chunk,
                        spot_check_tolerance=args.spot_check_tolerance,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "materialize":
            print(
                json.dumps(
                    materialize_dataset(
                        args.dataset,
                        output_root=args.output_root,
                        force_download=args.force_download,
                        scan_groundtruth=not args.skip_groundtruth_scan,
                        include_file_sha256=args.include_file_sha256,
                        spot_check_queries=args.spot_check_queries,
                        spot_check_seed=args.spot_check_seed,
                        spot_check_base_chunk=args.spot_check_base_chunk,
                        spot_check_tolerance=args.spot_check_tolerance,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "validate-converted":
            print(
                json.dumps(
                    validate_converted_dataset(
                        args.dataset,
                        args.output_root,
                        scan_headers=not args.skip_header_scan,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "convert":
            print(
                json.dumps(
                    convert_materialized_dataset(
                        args.dataset,
                        output_root=args.output_root,
                        force_convert=args.force_convert,
                        scan_converted=not args.skip_converted_header_scan,
                        include_file_sha256=args.include_file_sha256,
                        chunk_rows=args.conversion_chunk_rows,
                        command_line=" ".join(sys.argv if argv is None else argv),
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "texmex-list":
            rows = [
                {
                    "name": spec.name,
                    "display_name": spec.display_name,
                    "role": spec.role,
                    "url": spec.url,
                    "archive_bytes": spec.archive_bytes,
                    "md5": spec.md5,
                    "metric": spec.metric,
                    "slice_based": spec.slice_based,
                    "base_split": spec.base_split,
                    "query_split": spec.query_split,
                    "groundtruth_scope": spec.groundtruth_scope,
                    "groundtruth_exact": spec.groundtruth_exact,
                    "files": [asdict(file_spec) for file_spec in spec.files],
                }
                for spec in TEXMEX_DATASETS.values()
            ]
            if args.json:
                print(json.dumps(rows, indent=2, sort_keys=True))
            else:
                for row in rows:
                    base = next(
                        item
                        for item in row["files"]
                        if item["output_name"].endswith("_base.fvecs")
                    )
                    query = next(
                        item
                        for item in row["files"]
                        if item["output_name"].endswith("_query.fvecs")
                    )
                    print(
                        f"{row['name']}: dim={base['dim']} base={base['count']} "
                        f"query={query['count']} metric={row['metric']} "
                        f"archive_bytes={row['archive_bytes']} "
                        f"md5={row['md5']} url={row['url']}"
                    )
            return 0
        if args.command == "texmex-manifest":
            print(
                json.dumps(
                    texmex_manifest(
                        args.dataset,
                        args.output_root,
                        archive_info=None,
                        validation=None,
                        include_file_sha256=args.include_file_sha256,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "texmex-validate":
            validation = validate_texmex_dataset(
                args.dataset,
                args.output_root,
                scan_headers=not args.skip_header_scan,
            )
            print(json.dumps(validation, indent=2, sort_keys=True))
            return 0
        if args.command == "texmex-materialize":
            print(
                json.dumps(
                    materialize_texmex(
                        args.dataset,
                        source_root=args.source_root,
                        output_root=args.output_root,
                        force_download=args.force_download,
                        force_extract=args.force_extract,
                        scan_headers=not args.skip_header_scan,
                        include_file_sha256=args.include_file_sha256,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
    except ValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    parser.error(f"unhandled command {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
