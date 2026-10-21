import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate_phase6 import selected_f1_threshold, top_k_metrics  # noqa: E402


class Phase6EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inner = pd.read_csv(ROOT / "results_v2/phase6_independent_inner_predictions.csv.gz")
        cls.fit_audit = pd.read_csv(ROOT / "results_v2/phase6_independent_inner_fit_audit.csv")
        cls.selection = pd.read_csv(ROOT / "results_v2/phase6_independent_gate_selection.csv")
        cls.metrics = pd.read_csv(ROOT / "results_v2/phase6_fold_metrics.csv")
        cls.paired = pd.read_csv(ROOT / "results_v2/phase6_paired_differences.csv")
        cls.bootstrap = pd.read_csv(ROOT / "results_v2/phase6_scaffold_bootstrap.csv")
        cls.thresholds = pd.read_csv(ROOT / "results_v2/phase6_threshold_audit.csv")
        cls.reliability = pd.read_csv(ROOT / "results_v2/phase6_reliability_bins.csv")

    def test_inner_fit_never_uses_held_scaffold_or_outer_test(self):
        self.assertEqual(len(self.fit_audit), 360)
        self.assertTrue((self.fit_audit[["source_overlap_n", "scaffold_overlap_n", "outer_test_overlap_n"]] == 0).all().all())
        self.assertEqual(len(self.selection[self.selection.selected_by_one_se]), 30)

    def test_each_independent_validation_row_appears_once_per_method(self):
        keys = ["repeat", "outer_fold", "seed", "method", "source_row"]
        self.assertFalse(self.inner.duplicated(keys).any())
        self.assertTrue(self.inner.probability.between(0, 1).all())
        self.assertEqual(self.inner.groupby(["repeat", "outer_fold", "seed", "method"]).ngroups, 300)
        for (repeat, fold), group in self.inner.groupby(["repeat", "outer_fold"]):
            manifest = pd.read_csv(ROOT / f"data/splits_v2/outer_repeat_{repeat}.csv")
            expected = set(manifest.loc[manifest.outer_fold != fold, "source_row"])
            actual = set(group.loc[(group.seed == 11) & (group.method == "C1_fixed_weight"), "source_row"])
            self.assertEqual(actual, expected)

    def test_top_k_and_threshold_definitions(self):
        y = np.array([1, 0, 1, 0])
        p = np.array([0.9, 0.8, 0.2, 0.1])
        result = top_k_metrics(y, p, np.arange(4), 0.25)
        self.assertEqual(result["selected_n"], 1)
        self.assertEqual(result["selected_active_n"], 1)
        self.assertAlmostEqual(result["enrichment_factor"], 2.0)
        self.assertAlmostEqual(selected_f1_threshold(y, p), 0.2)

    def test_primary_paired_difference_reproduces_fold_scores(self):
        primary = self.paired[self.paired.candidate == "D2_similarity_entropy"]
        self.assertEqual(len(primary), 10)
        for row in primary.itertuples():
            subset = self.metrics[(self.metrics.repeat == row.repeat) & (self.metrics.outer_fold == row.outer_fold)]
            pivot = subset[subset.method.isin((row.candidate, row.baseline))].pivot(index="seed", columns="method", values="average_precision")
            difference = (pivot[row.candidate] - pivot[row.baseline]).mean()
            self.assertAlmostEqual(row.delta_average_precision, difference, places=12)
        self.assertEqual(len(self.thresholds), 300)
        self.assertTrue((self.thresholds.fit_source == "independent_inner_validation").all())

    def test_bootstrap_is_paired_and_complete(self):
        self.assertEqual(len(self.bootstrap), 4000)
        for _, group in self.bootstrap.groupby(["candidate", "baseline"]):
            self.assertEqual(set(group.bootstrap_replicate), set(range(1, 2001)))
            self.assertTrue(np.isfinite(group.delta_average_precision).all())
            self.assertTrue(np.isfinite(group.delta_brier_score).all())

    def test_reliability_bins_are_paired_descriptive_summaries(self):
        self.assertEqual(len(self.reliability), 20)
        for method, group in self.reliability.groupby("method"):
            self.assertEqual(set(group.bin), set(range(1, 11)))
            self.assertEqual(group.n_repeat_molecules.sum(), 2 * 1513)
            self.assertTrue(group.mean_predicted.between(0, 1).all())
        wide = self.reliability.pivot(index="bin", columns="method", values="observed_fraction")
        self.assertTrue(np.allclose(wide.iloc[:, 0], wide.iloc[:, 1]))


if __name__ == "__main__":
    unittest.main()
