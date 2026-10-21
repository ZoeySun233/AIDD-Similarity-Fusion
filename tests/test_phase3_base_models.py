import json
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


class Phase3BaseModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictions = pd.read_csv(ROOT / "results_v2/base_model_predictions.csv")
        cls.metadata = json.loads((ROOT / "results_v2/phase3_run_metadata.json").read_text())

    def test_unified_prediction_contract(self):
        expected = {"CID", "repeat", "outer_fold", "seed", "model", "split", "label", "probability"}
        self.assertTrue(expected.issubset(self.predictions.columns))
        self.assertTrue(self.predictions.probability.between(0, 1).all())
        self.assertEqual(set(self.predictions.model), {"morgan_rf", "frozen_chemberta_mlp"})
        self.assertEqual(set(self.predictions.seed), {11, 22, 33})

    def test_each_outer_context_has_complete_oof_and_test_predictions(self):
        keys = ["repeat", "outer_fold", "seed", "model"]
        for _, frame in self.predictions.groupby(keys):
            self.assertEqual(len(frame), 1513)
            self.assertEqual(frame.source_row.nunique(), 1513)
            self.assertEqual(frame[frame.split == "oof_development"].source_row.nunique(), len(frame[frame.split == "oof_development"]))
            self.assertEqual(frame[frame.split == "outer_test"].source_row.nunique(), len(frame[frame.split == "outer_test"]))

    def test_metadata_declares_oof_and_outer_test_boundaries(self):
        self.assertIn("without that sample", self.metadata["oof_policy"])
        self.assertIn("excluded", self.metadata["outer_test_policy"])


if __name__ == "__main__":
    unittest.main()
