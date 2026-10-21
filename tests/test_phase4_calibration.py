import json
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


class Phase4CalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = pd.read_csv(ROOT / "results_v2/calibrated_base_model_predictions.csv")
        cls.audit = pd.read_csv(ROOT / "results_v2/calibration_fit_audit.csv")
        cls.metadata = json.loads((ROOT / "results_v2/phase4_calibration_metadata.json").read_text())

    def test_calibrated_predictions_match_phase3_coverage(self):
        self.assertEqual(len(self.data), 90780)
        self.assertTrue(self.data[["probability_raw", "probability_platt", "probability_isotonic"]].apply(lambda column: column.between(0, 1).all()).all())

    def test_each_calibrator_fits_oof_not_outer_test(self):
        self.assertEqual(len(self.audit), 60)
        self.assertTrue((self.audit.calibration_fit_split == "oof_development").all())
        self.assertTrue((self.audit.fit_rows + self.audit.outer_test_rows == 1513).all())

    def test_primary_calibration_changes_at_least_some_probabilities(self):
        self.assertGreater((self.data.probability_raw != self.data.probability_platt).sum(), 0)

    def test_metadata_locks_shared_component_input_for_fusion(self):
        self.assertIn("probability_platt", self.metadata["fusion_policy"])
        self.assertIn("outer-test labels are excluded", self.metadata["fit_policy"])


if __name__ == "__main__":
    unittest.main()
