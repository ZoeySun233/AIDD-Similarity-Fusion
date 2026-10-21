"""Synthetic-data tests for fixed-budget exploratory deferral calculations."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from analyze_wp4_screening import analyze_assay  # noqa: E402


class Wp4ScreeningTests(unittest.TestCase):
    def setUp(self) -> None:
        n = 20
        self.frame = pd.DataFrame({
            "assay_chembl_id": ["SYNTHETIC"] * n,
            "standardized_inchikey": [f"ID{i:02d}" for i in range(n)],
            "label": [1 if i % 3 == 0 else 0 for i in range(n)],
            "c1_probability": np.linspace(0.05, 0.95, n),
            "rf_probability": np.linspace(0.1, 0.9, n),
            "chem_probability": np.linspace(0.9, 0.1, n),
            "max_similarity": np.linspace(0.2, 0.8, n),
        })
        self.config = {
            "screening_fraction_of_original_cohort": 0.05,
            "coverage_fractions": [1.0, 0.5],
            "deferral_rules": ["c1_boundary_distance", "expert_agreement", "training_similarity"],
        }

    def test_full_coverage_is_same_for_every_rule(self) -> None:
        rows = analyze_assay(self.frame, self.config)
        full = [row for row in rows if row["target_coverage"] == 1.0]
        self.assertEqual(len(full), 3)
        self.assertEqual({row["retained_n"] for row in full}, {20})
        self.assertEqual({row["fixed_screening_k"] for row in full}, {1})
        self.assertEqual(len({row["retained_brier"] for row in full}), 1)
        self.assertEqual(len({row["precision_at_fixed_top_5pct"] for row in full}), 1)

    def test_deferral_accounts_for_all_molecules_and_keeps_budget_fixed(self) -> None:
        rows = analyze_assay(self.frame, self.config)
        half = [row for row in rows if row["target_coverage"] == 0.5]
        self.assertEqual(len(half), 3)
        for row in half:
            self.assertEqual(row["retained_n"], 10)
            self.assertEqual(row["deferred_n"], 10)
            self.assertEqual(row["fixed_screening_k"], 1)
            self.assertEqual(row["retained_active"] + row["retained_inactive"], 10)
            self.assertEqual(row["deferred_active"] + row["deferred_inactive"], 10)


if __name__ == "__main__":
    unittest.main()
