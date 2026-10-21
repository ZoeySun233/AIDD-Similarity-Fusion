import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import joblib


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate import classification_metrics  # noqa: E402
from fingerprints import MorganConfig, morgan_fingerprints  # noqa: E402


class Week2BaselineTests(unittest.TestCase):
    def test_fingerprint_shape_dtype_and_determinism(self):
        smiles = ["CCO", "c1ccccc1"]
        config = MorganConfig(radius=2, n_bits=128)
        first = morgan_fingerprints(smiles, config)
        second = morgan_fingerprints(smiles, config)
        self.assertEqual(first.shape, (2, 128))
        self.assertEqual(first.dtype, np.uint8)
        np.testing.assert_array_equal(first, second)

    def test_invalid_smiles_is_rejected(self):
        with self.assertRaises(ValueError):
            morgan_fingerprints(["not-a-smiles"], MorganConfig(n_bits=64))

    def test_metric_contract(self):
        y_true = np.array([0, 0, 1, 1])
        probabilities = np.array([0.1, 0.2, 0.8, 0.9])
        metrics = classification_metrics(y_true, probabilities)
        self.assertAlmostEqual(metrics["roc_auc"], 1.0)
        self.assertAlmostEqual(metrics["pr_auc"], 1.0)
        self.assertTrue(0.0 <= metrics["threshold"] <= 1.0)

    def test_train_validation_scaffolds_are_disjoint(self):
        train = pd.read_csv(ROOT / "data/processed/train.csv")
        valid = pd.read_csv(ROOT / "data/processed/valid.csv")
        train_scaffolds = set(train["scaffold"].fillna(""))
        valid_scaffolds = set(valid["scaffold"].fillna(""))
        self.assertTrue(train_scaffolds.isdisjoint(valid_scaffolds))

    def test_saved_model_reproduces_saved_validation_predictions(self):
        model = joblib.load(ROOT / "models/random_forest_morgan.joblib")
        cached = np.load(ROOT / "data/processed/features/morgan_r2_2048_valid.npz")
        expected = pd.read_csv(ROOT / "results/rf_validation_predictions.csv")[
            "probability_active"
        ].to_numpy()
        actual = model.predict_proba(cached["X"])[:, 1]
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
