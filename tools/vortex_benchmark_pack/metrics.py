"""Compatibility re-exports for benchmark-pack metric helpers."""

from __future__ import annotations

from .dominant_cost import dominant_cost
from .metric_flattening import flatten_metrics
from .selector_summary import summarize_selector

__all__ = ["dominant_cost", "flatten_metrics", "summarize_selector"]
