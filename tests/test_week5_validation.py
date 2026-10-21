import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


class Week5ValidationTests(unittest.TestCase):
    def test_all_three_split_scaffold_sets_are_disjoint(self):
        frames = [pd.read_csv(ROOT / f"data/processed/{split}.csv") for split in ("train", "valid", "test")]
        scaffolds = [set(frame["scaffold"].fillna("")) for frame in frames]
        self.assertTrue(scaffolds[0].isdisjoint(scaffolds[1]))
        self.assertTrue(scaffolds[0].isdisjoint(scaffolds[2]))
        self.assertTrue(scaffolds[1].isdisjoint(scaffolds[2]))

    def test_test_feature_caches_align_with_test_csv(self):
        test = pd.read_csv(ROOT / "data/processed/test.csv")
        expected = test["canonical_smiles"].astype(str).to_numpy(dtype=str)
        for name in ("morgan_r2_2048_test.npz", "chemberta_77m_mlm_test.npz"):
            with np.load(ROOT / "data/processed/features" / name) as cache:
                np.testing.assert_array_equal(cache["smiles"], expected)
                self.assertEqual(len(cache["X"]), len(test))

    def test_four_experiments_have_five_test_seeds(self):
        metrics = pd.read_csv(ROOT / "results/week5_test_metrics_by_seed.csv")
        self.assertEqual(set(metrics["experiment"]), {"A", "B", "C", "D"})
        self.assertTrue((metrics.groupby("experiment")["seed"].nunique() == 5).all())
        self.assertTrue((metrics["split"] == "test").all())

    def test_dynamic_formula_reproduces_saved_test_predictions(self):
        predictions = pd.read_csv(ROOT / "results/week5_test_predictions.csv")
        parameters = pd.read_csv(ROOT / "results/week5_fusion_parameters_by_seed.csv").set_index("seed")
        for seed, group in predictions.groupby("seed"):
            row = parameters.loc[seed]
            alpha = 1.0 / (1.0 + np.exp(-(row["dynamic_a"] * group["max_train_tanimoto"] + row["dynamic_b"])))
            expected = alpha * group["rf_probability"] + (1.0 - alpha) * group["chemberta_probability"]
            np.testing.assert_allclose(alpha, group["dynamic_alpha_rf"], atol=1e-12)
            np.testing.assert_allclose(expected, group["dynamic_fusion_probability"], atol=1e-12)

    def test_summary_declares_locked_test_scope(self):
        with (ROOT / "results/week5_test_summary.json").open(encoding="utf-8") as handle:
            summary = json.load(handle)
        self.assertIn("locked-test", summary["scope"])
        self.assertEqual(summary["seeds"], [11, 22, 33, 44, 55])


if __name__ == "__main__":
    unittest.main()
