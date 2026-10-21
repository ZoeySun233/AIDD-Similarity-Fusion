import json
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


class Phase5FusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictions = pd.read_csv(ROOT / "results_v2/fusion_predictions_v2.csv")
        cls.selection = pd.read_csv(ROOT / "results_v2/gate_selection_v2.csv")
        cls.metadata = json.loads((ROOT / "results_v2/phase5_fusion_metadata.json").read_text())

    def test_required_controls_and_ablations_are_present(self):
        required = {"C0_equal_weight", "C1_fixed_weight", "D0_original_similarity_gate", "E0_logistic_stacking", "D1_shrunk_similarity", "D2_similarity_entropy", "D3_similarity_entropy_topk", "D4_similarity_entropy_topk_variance", "N0_permuted_similarity", "N1_random_matched_gate", "N2_confidence_only"}
        self.assertTrue(required.issubset(set(self.predictions.method)))

    def test_every_method_context_predicts_each_outer_test_row_once(self):
        grouped = self.predictions.groupby(["repeat", "outer_fold", "seed", "method"])
        self.assertTrue((grouped.source_row.nunique() == grouped.size()).all())
        self.assertTrue(self.predictions.probability.between(0, 1).all())

    def test_one_standard_error_selects_one_gate_per_context(self):
        chosen = self.selection[self.selection.selected_by_one_se]
        self.assertEqual(len(chosen), 30)
        self.assertTrue(set(chosen.candidate).issubset({"D1_shrunk_similarity", "D2_similarity_entropy", "D3_similarity_entropy_topk", "D4_similarity_entropy_topk_variance"}))

    def test_metadata_locks_calibrated_component_input(self):
        self.assertIn("probability_platt", self.metadata["component_input"])
        self.assertIn("OOF-development", self.metadata["gate_fit_policy"])


if __name__ == "__main__":
    unittest.main()
