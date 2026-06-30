"""CSV, Markdown, and HTML report rendering."""

from __future__ import annotations

import csv
import html
from pathlib import Path
from typing import Any


def write_summary_csv(path: Path, metrics: dict[str, Any]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "value"])
        for key in sorted(metrics):
            writer.writerow([key, metrics[key]])


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f"{value:,.2f}"
        return f"{value:.6g}"
    return str(value)


def _metric(metrics: dict[str, Any], key: str) -> str:
    return html.escape(_fmt(metrics.get(key)))


def _card(label: str, value: str, detail: str = "") -> str:
    detail_html = f"<div class=\"detail\">{html.escape(detail)}</div>" if detail else ""
    return (
        "<div class=\"card\">"
        f"<div class=\"label\">{html.escape(label)}</div>"
        f"<div class=\"value\">{html.escape(value)}</div>"
        f"{detail_html}</div>"
    )


def _metric_table(metrics: dict[str, Any], keys: list[str]) -> str:
    rows = []
    for key in keys:
        if key in metrics:
            rows.append(
                "<tr>"
                f"<th>{html.escape(key)}</th>"
                f"<td>{html.escape(_fmt(metrics[key]))}</td>"
                "</tr>"
            )
    if not rows:
        return "<div class=\"empty\">No metrics available for this section.</div>"
    return "<table>" + "\n".join(rows) + "</table>"


def _all_metrics_table(metrics: dict[str, Any]) -> str:
    rows = "\n".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(_fmt(value))}</td></tr>"
        for key, value in sorted(metrics.items())
    )
    return f"<table>{rows}</table>"


def write_report(run_dir: Path, manifest: dict[str, Any], metrics: dict[str, Any]) -> None:
    validation_status = metrics.get("materialized_recall_validation_status")
    validation_delta = metrics.get("materialized_selector_vs_eval_recall_delta")
    validation_abs_delta = metrics.get("materialized_selector_vs_eval_recall_abs_delta")
    dominant = manifest["dominant_cost"]
    quality_keys = [
        "best_recall_at_k_in_window",
        "materialized_overlay_match_score",
        "eval_overlay_match_score",
        "eval_node_locality_score",
        "eval_recall_at_k_in_window",
        "materialized_selector_vs_eval_recall_delta",
        "materialized_selector_vs_eval_recall_abs_delta",
        "materialized_recall_validation_status",
    ]
    cost_keys = [
        "selector_wall_seconds",
        "selector_peak_rss_kib",
        "candidate_ok_count",
        "candidate_count",
        "dominant_cost",
        "dominant_cost_wall_seconds",
        "train_threads",
        "train_threads_fallback",
    ]
    artifact_keys = [
        "model_file_bytes",
        "codegen_model_bin_bytes",
        "codegen_centroid_index_bytes",
        "codegen_package_bytes",
        "codegen_cold_load_time_ms",
        "codegen_warm_load_time_ms",
        "hash_latency_avg_ms",
        "hash_latency_p95_ms",
        "hash_latency_p99_ms",
    ]
    provenance_keys = [
        "dataset_manifest_available",
        "dataset_manifest_required",
        "dataset_manifest_path",
        "dataset_manifest_sha256",
        "dataset_name",
        "dataset_metric",
        "dataset_source_family",
        "dataset_base_split",
        "dataset_query_split",
        "dataset_groundtruth_scope",
        "dataset_raw_validation_ok",
        "dataset_conversion_available",
        "dataset_conversion_validated",
        "dataset_base_manifest_sha256",
        "dataset_query_manifest_sha256",
        "dataset_groundtruth_manifest_sha256",
    ]
    report_html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Vortex SIFT Benchmark Pack</title>
<style>
:root {{ --page:#f6f7f3; --card:#ffffff; --ink:#111827; --muted:#64748b; --line:#d8ded2; --blue:#1d4ed8; --amber:#b45309; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; padding: 24px; background: var(--page); color: var(--ink); font: 14px/1.45 "Avenir Next", "Segoe UI", system-ui, sans-serif; font-variant-numeric: tabular-nums; }}
.wrap {{ max-width: 1280px; margin: 0 auto; display: grid; gap: 16px; }}
.hero {{ background: #111827; color: #f8fafc; border-radius: 8px; padding: 20px 22px; box-shadow: 0 18px 40px rgba(15, 23, 42, 0.16); }}
h1 {{ margin: 0 0 8px; font-size: 24px; }}
h2 {{ margin: 0 0 10px; font-size: 17px; }}
.meta {{ color: #cbd5e1; display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 6px 16px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 10px; }}
.card, .panel {{ background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 14px; }}
.card {{ background: #fbfcf9; }}
.label {{ color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 700; }}
.value {{ font-size: 20px; font-weight: 750; line-height: 1.2; margin-top: 4px; }}
.detail, .note {{ color: var(--muted); font-size: 12px; margin-top: 4px; }}
.callout {{ background: #fff7ed; border: 1px solid #fed7aa; border-left: 4px solid var(--amber); border-radius: 8px; padding: 12px 14px; }}
table {{ border-collapse: collapse; width: 100%; margin: 0.4rem 0 1rem; }}
th, td {{ border-bottom: 1px solid var(--line); padding: 0.45rem 0.6rem; text-align: left; vertical-align: top; }}
th {{ background: #f8fafc; color: #334155; width: 24rem; }}
code {{ background: #eef1ea; padding: 0.1rem 0.25rem; border-radius: 0.2rem; }}
details summary {{ cursor: pointer; font-weight: 700; }}
a {{ color: var(--blue); }}
@media(max-width: 760px) {{ body {{ padding: 12px; }} th {{ width: auto; }} }}
</style>
</head>
<body><div class="wrap">
<section class="hero">
<h1>Vortex SIFT Benchmark Pack</h1>
<div class="meta">
<div><strong>Profile:</strong> {html.escape(str(manifest.get("profile")))}</div>
<div><strong>Preset:</strong> {html.escape(str(manifest.get("preset")))}</div>
<div><strong>Materialized role:</strong> {html.escape(str(manifest.get("materialized_role")))} requested {html.escape(str(manifest.get("materialized_role_requested")))}</div>
<div><strong>Git:</strong> {html.escape(str(manifest.get("git", {}).get("commit")))} dirty={html.escape(str(manifest.get("git", {}).get("dirty")))}</div>
</div>
</section>

<section class="grid">
{_card("Selector Wall", _fmt(metrics.get("selector_wall_seconds", "n/a")) + " s", "selector end-to-end")}
{_card("Candidates", f"{_fmt(metrics.get('candidate_ok_count'))} / {_fmt(metrics.get('candidate_count'))}", "ok / total")}
{_card("Eval Recall", _fmt(metrics.get("eval_recall_at_k_in_window")), "generated-code vortex_eval")}
{_card("Overlay Match", _fmt(metrics.get("eval_overlay_match_score")), "generated-code placement")}
{_card("Hash Latency Avg", _fmt(metrics.get("hash_latency_avg_ms")) + " ms", "generated-code inference")}
{_card("Package Size", _fmt(metrics.get("codegen_package_bytes")) + " B", "generated package")}
</section>

<section class="callout"><strong>Dominant cost:</strong> {html.escape(str(dominant.get("name")))}
({_fmt(dominant.get("wall_seconds"))} seconds).<br>
{html.escape(str(dominant.get("recommendation")))}</section>

<section class="callout"><strong>Materialized recall validation:</strong>
{html.escape(str(validation_status))}. <code>vortex_eval</code> recall minus selector materialized recall:
{html.escape(_fmt(validation_delta))}; absolute delta: {html.escape(_fmt(validation_abs_delta))}.</section>

<section class="panel"><h2>Dataset Provenance</h2>{_metric_table(metrics, provenance_keys)}</section>
<section class="panel"><h2>Quality Metrics</h2>{_metric_table(metrics, quality_keys)}</section>
<section class="panel"><h2>Cost and Search Metrics</h2>{_metric_table(metrics, cost_keys)}</section>
<section class="panel"><h2>Artifacts and Runtime Metrics</h2>{_metric_table(metrics, artifact_keys)}</section>

<details class="panel"><summary>All Summary Metrics</summary>
<p class="note">Complete flat metric table for spreadsheets and regression checks.</p>
{_all_metrics_table(metrics)}
</details>

<section class="panel"><h2>Artifacts</h2>
<ul>
<li><code>manifest.json</code>: full machine-readable run metadata.</li>
<li><code>summary.csv</code>: flat metric table for spreadsheets.</li>
<li><code>selector.json</code> and <code>selector.html</code>: selector output and focused visual report.</li>
<li><code>logs/</code>: raw command output and exact command lines.</li>
</ul>
</section>
</div></body>
</html>
"""
    (run_dir / "report.html").write_text(report_html, encoding="utf-8")
    report_md = [
        "# Vortex SIFT Benchmark Pack",
        "",
        f"- profile: `{manifest['profile']}`",
        f"- preset: `{manifest['preset']}`",
        f"- materialized_role_requested: `{manifest.get('materialized_role_requested')}`",
        f"- materialized_role: `{manifest.get('materialized_role')}`",
        f"- git_commit: `{manifest['git']['commit']}`",
        f"- git_dirty: `{manifest['git']['dirty']}`",
        f"- dominant_cost: `{manifest['dominant_cost'].get('name')}`",
        f"- recommendation: {manifest['dominant_cost'].get('recommendation')}",
        f"- dataset_manifest_path: `{metrics.get('dataset_manifest_path')}`",
        f"- dataset_manifest_sha256: `{metrics.get('dataset_manifest_sha256')}`",
        f"- materialized_recall_validation_status: `{validation_status}`",
        f"- materialized_selector_vs_eval_recall_delta: `{validation_delta}`",
        f"- materialized_selector_vs_eval_recall_abs_delta: `{validation_abs_delta}`",
        "",
        "## Summary Metrics",
        "",
    ]
    for key in sorted(metrics):
        report_md.append(f"- `{key}`: {metrics[key]}")
    (run_dir / "README.md").write_text("\n".join(report_md) + "\n")
