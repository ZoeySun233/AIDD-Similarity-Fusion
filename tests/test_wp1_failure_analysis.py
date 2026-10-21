import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyze_failure_wp1 import safe_ap, score_groups  # noqa: E402


class FailureAnalysisTests(unittest.TestCase):
    def test_ap_is_missing_for_single_class_groups(self):
        self.assertTrue(np.isnan(safe_ap(np.array([1, 1]), np.array([0.1, 0.9]))))

    def test_score_groups_uses_probabilities_and_counts(self):
        frame = pd.DataFrame({
            "repeat": [1, 1, 1, 1], "outer_fold": [1] * 4,
            "label": [1, 0, 1, 0], "scaffold": ["a", "a", "b", "b"],
            "rf": [0.9, 0.2, 0.7, 0.3], "chemberta": [0.8, 0.4, 0.6, 0.1],
            "fixed": [0.85, 0.3, 0.65, 0.2], "dynamic": [0.86, 0.29, 0.66, 0.19],
            "error_overlap_at_0_5": ["neither_wrong"] * 4,
            "expert_disagreement": [0.1, 0.2, 0.1, 0.2],
            "rf_advantage_brier": [0.01, 0.12, 0.13, -0.08],
        })
        for name in ("rf", "chemberta", "fixed", "dynamic"):
            frame[f"squared_error_{name}"] = np.square(frame.label - frame[name])
        score = score_groups(frame, ["repeat", "outer_fold"]).iloc[0]
        self.assertEqual(score.n_molecules, 4)
        self.assertEqual(score.n_active, 2)
        self.assertEqual(score.n_scaffolds, 2)
        self.assertAlmostEqual(score.brier_rf, float(np.square(frame.label - frame.rf).mean()))
        self.assertAlmostEqual(score.ap_rf, 1.0)

    def test_generated_atlas_is_paired_and_development_cutoffs_are_respected(self):
        path = ROOT / "results_v2/wp1_error_atlas.csv"
        if not path.exists():
            self.skipTest("Run the WP1 analysis to generate its outputs.")
        atlas = pd.read_csv(path)
        self.assertEqual(len(atlas), 2 * 1513)
        self.assertFalse(atlas.duplicated(["repeat", "source_row"]).any())
        self.assertEqual(set(atlas.repeat), {1, 2})
        self.assertTrue(atlas[["rf", "chemberta", "fixed", "dynamic"]].ge(0).all().all())
        self.assertTrue(atlas[["rf", "chemberta", "fixed", "dynamic"]].le(1).all().all())
        low = atlas.similarity_band == "low"
        high = atlas.similarity_band == "high"
        self.assertTrue((atlas.loc[low, "max_similarity"] <= atlas.loc[low, "low_similarity_cutoff"]).all())
        self.assertTrue((atlas.loc[high, "max_similarity"] >= atlas.loc[high, "high_similarity_cutoff"]).all())
        self.assertTrue(np.allclose(atlas.rf_advantage_brier,
                                    atlas.squared_error_chemberta - atlas.squared_error_rf))


if __name__ == "__main__":
    unittest.main()
