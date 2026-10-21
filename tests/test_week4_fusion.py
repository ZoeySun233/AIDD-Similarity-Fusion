import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fusion import (  # noqa: E402
    dynamic_fusion,
    dynamic_weights,
    fixed_fusion,
    maximum_tanimoto_similarity,
    tanimoto_similarity_matrix,
)


class Week4FusionTests(unittest.TestCase):
    def test_tanimoto_known_examples(self):
        query = np.array([[1, 1, 0, 0], [0, 0, 1, 0]], dtype=np.uint8)
        reference = np.array([[1, 1, 0, 0], [1, 0, 1, 0]], dtype=np.uint8)
        actual = tanimoto_similarity_matrix(query, reference)
        expected = np.array([[1.0, 1.0 / 3.0], [0.0, 0.5]])
        np.testing.assert_allclose(actual, expected)

    def test_maximum_similarity_and_index(self):
        query = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.uint8)
        reference = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)
        similarity, indices = maximum_tanimoto_similarity(query, reference)
        np.testing.assert_allclose(similarity, [1.0, 0.5])
        np.testing.assert_array_equal(indices, [0, 1])

    def test_fixed_fusion_boundaries_recover_component_models(self):
        rf = np.array([0.2, 0.8])
        chemberta = np.array([0.6, 0.4])
        np.testing.assert_allclose(fixed_fusion(rf, chemberta, 1.0), rf)
        np.testing.assert_allclose(fixed_fusion(rf, chemberta, 0.0), chemberta)

    def test_dynamic_weights_are_bounded_and_directional(self):
        similarity = np.array([0.1, 0.5, 0.9])
        increasing = dynamic_weights(similarity, a=4.0, b=-2.0)
        decreasing = dynamic_weights(similarity, a=-4.0, b=2.0)
        self.assertTrue(np.all((increasing > 0) & (increasing < 1)))
        self.assertTrue(np.all(np.diff(increasing) > 0))
        self.assertTrue(np.all(np.diff(decreasing) < 0))

    def test_saved_predictions_reproduce_dynamic_formula(self):
        with (ROOT / "results/fusion_summary.json").open(encoding="utf-8") as handle:
            summary = json.load(handle)
        predictions = pd.read_csv(ROOT / "results/fusion_validation_predictions.csv")
        actual, alpha = dynamic_fusion(
            predictions["rf_probability"].to_numpy(),
            predictions["chemberta_probability"].to_numpy(),
            predictions["max_train_tanimoto"].to_numpy(),
            summary["dynamic_fusion"]["a"],
            summary["dynamic_fusion"]["b"],
        )
        np.testing.assert_allclose(actual, predictions["dynamic_fusion_probability"], atol=1e-12)
        np.testing.assert_allclose(alpha, predictions["dynamic_alpha_rf"], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
