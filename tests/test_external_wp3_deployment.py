"""Synthetic-data checks that do not open external outcome files."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluate_external_wp3_deployment import metric_row, paired_bootstrap  # noqa: E402


def _config() -> dict:
    return {"evaluation": {
        "log_loss_clip": 1e-6, "top_fractions": [0.05, 0.1],
        "bootstrap_replicates": 50, "bootstrap_max_attempts": 1000,
        "bootstrap_min_valid": 45, "bootstrap_seed": 7,
        "insufficient_cluster_threshold": 2, "wide_ci_threshold": 1.0,
    }}


def test_metric_row_ranking_ties_use_stable_identity():
    frame = pd.DataFrame({
        "label": [0, 1, 0, 1], "assay_chembl_id": ["A"] * 4,
        "standardized_inchikey": ["D", "A", "B", "C"],
    })
    result = metric_row(frame, np.array([0.8, 0.8, 0.2, 0.2]), _config())
    assert result["k_top_5pct"] == 1
    assert result["precision_top_5pct"] == 1.0  # ID A wins the 0.8 tie.
    assert result["prevalence"] == 0.5
    assert result["ef_top_5pct"] == 2.0


def test_paired_bootstrap_uses_scaffold_clusters_and_valid_classes():
    frame = pd.DataFrame({
        "scaffold": ["s1", "s1", "s2", "s2", "s3", "s3", "s4", "s4"],
        "label": [0, 1, 0, 1, 0, 1, 0, 1],
        "c1_probability": [0.2, 0.8] * 4,
        "d2_probability": [0.1, 0.9] * 4,
    })
    result = paired_bootstrap(frame, _config())
    assert result["valid"] == 50
    assert result["n_scaffolds"] == 4
    assert result["informative"] is True
    assert result["ci_lower"] == 0.0
    assert result["ci_upper"] == 0.0


def test_paired_bootstrap_single_class_resamples_are_redrawn():
    frame = pd.DataFrame({
        "scaffold": ["positive", "negative"],
        "label": [1, 0],
        "c1_probability": [0.6, 0.4],
        "d2_probability": [0.7, 0.3],
    })
    result = paired_bootstrap(frame, _config())
    assert result["valid"] == 50
    assert result["rejected_single_class"] > 0
