import hashlib
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phase7_dmpnn import DirectedMPNN, collate_graphs, molecular_graph  # noqa: E402
from phase7_models import FeatureBundle, fit_predict  # noqa: E402


class Phase7AblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = pd.read_csv(ROOT / "data/processed_v2/bace_standardized.csv", keep_default_na=False).sort_values("source_row")
        cls.predictions = pd.read_csv(ROOT / "results_v2/phase7_expert_predictions.csv.gz")
        cls.metrics = pd.read_csv(ROOT / "results_v2/phase7_expert_ablations.csv")
        cls.audit = pd.read_csv(ROOT / "results_v2/phase7_fit_audit.csv")
        cls.decisions = pd.read_csv(ROOT / "results_v2/phase7_expert_decisions.csv")

    def test_feature_caches_match_source_rows_and_molecular_identity(self):
        with np.load(ROOT / "data/processed_v2/features/chemberta_phase7_token_hidden.npz") as frozen:
            self.assertTrue(np.array_equal(frozen["source_row"], self.data.source_row.to_numpy()))
            self.assertEqual(frozen["hidden"].shape[0], 1513)
            self.assertEqual(frozen["mask"].shape[0], 1513)
        with np.load(ROOT / "data/processed_v2/features/chemberta_phase7_augmented_mean.npz") as augmented:
            self.assertTrue(np.array_equal(augmented["source_row"], self.data.source_row.to_numpy()))
            self.assertEqual(augmented["X"].shape[:2], (1513, 2))
            for original, variants in zip(self.data.standardized_smiles, augmented["smiles"]):
                expected = Chem.MolToSmiles(Chem.MolFromSmiles(original), canonical=True)
                for variant in variants:
                    self.assertEqual(Chem.MolToSmiles(Chem.MolFromSmiles(variant), canonical=True), expected)

    def test_every_variant_uses_the_same_outer_rows_and_has_valid_probabilities(self):
        expected = set(self.data.source_row)
        reference = self.predictions[self.predictions.variant == "mean_mlp"][
            ["repeat", "seed", "source_row", "outer_fold", "label", "scaffold"]
        ]
        for variant, frame in self.predictions.groupby("variant"):
            self.assertEqual(len(frame), 2 * 1513 * 3, variant)
            self.assertFalse(frame.duplicated(["repeat", "outer_fold", "seed", "source_row"]).any())
            self.assertTrue(frame.probability.between(0, 1).all())
            for repeat in (1, 2):
                for seed in (11, 22, 33):
                    subset = frame[(frame.repeat == repeat) & (frame.seed == seed)]
                    self.assertEqual(set(subset.source_row), expected)
            paired = frame.merge(reference, on=["repeat", "seed", "source_row"],
                                 suffixes=("", "_reference"), validate="one_to_one")
            self.assertEqual(len(paired), len(frame))
            for field in ("outer_fold", "label", "scaffold"):
                self.assertTrue((paired[field].fillna("<empty>") ==
                                 paired[f"{field}_reference"].fillna("<empty>")).all(), (variant, field))
        self.assertEqual(self.metrics.groupby("variant").size().nunique(), 1)
        self.assertTrue((self.metrics.groupby("variant").size() == 90).all())

    def test_all_fits_exclude_query_scaffolds_and_outer_test_training(self):
        self.assertTrue((self.audit[["source_overlap_n", "scaffold_overlap_n", "outer_test_fit_overlap_n"]] == 0).all().all())
        self.assertEqual(set(self.audit.stage), {
            "inner_parameter_search", "calibration_oof_fit", "outer_full_development_fit",
        })

    def test_frozen_mean_baseline_reproduces_phase4_outer_probabilities(self):
        mean = self.predictions[self.predictions.variant == "mean_mlp"]
        original = pd.read_csv(ROOT / "results_v2/calibrated_base_model_predictions.csv")
        original = original[(original.model == "frozen_chemberta_mlp") & (original.split == "outer_test")]
        paired = mean.merge(original[["repeat", "outer_fold", "seed", "source_row", "probability_platt"]],
                            on=["repeat", "outer_fold", "seed", "source_row"], validate="one_to_one")
        self.assertEqual(len(paired), 2 * 1513 * 3)
        self.assertLess(float(np.abs(paired.probability - paired.probability_platt).max()), 1e-6)

    def test_dmpnn_directed_bond_reverse_and_batch_shapes(self):
        graphs = [molecular_graph("CCO"), molecular_graph("c1ccccc1")]
        batch = collate_graphs(graphs, np.array([0, 1]))
        self.assertEqual(len(batch[0]), 9)
        self.assertEqual(len(batch[2]), 16)
        self.assertTrue(np.array_equal(batch[4][batch[4]].numpy(), np.arange(len(batch[4]))))
        model = DirectedMPNN(graphs[0].atom.shape[1], graphs[0].bond.shape[1], 16, 3, 0.1)
        self.assertEqual(tuple(model(batch).shape), (2,))

    def test_decisions_are_exploratory_and_complete(self):
        self.assertEqual(set(self.decisions.variant), set(self.predictions.variant))
        self.assertTrue(self.decisions.low_similarity_win_folds_vs_rf.between(0, 10).all())
        self.assertTrue(self.decisions.overall_win_folds_vs_mean_mlp.between(0, 10).all())
        self.assertTrue(np.isfinite(self.decisions[["mean_overall_ap", "mean_overall_brier",
                                                        "mean_adaptive_ece", "mean_calibration_intercept",
                                                        "mean_calibration_slope"]].to_numpy()).all())
        metadata = json.loads((ROOT / "results_v2/phase7_run_metadata.json").read_text())
        self.assertEqual(metadata["status"], "post_phase5_exploratory_not_independent_confirmation")

    def test_randomized_smiles_cache_is_training_only(self):
        mean = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.4, 0.5, 0.6], [0.6, 0.7, 0.8]], dtype=np.float32)
        augmented = np.repeat(mean[:, None, :], 2, axis=1)
        bundle = FeatureBundle(mean=mean, cls=mean, hidden=np.empty((4, 0, 3)), mask=np.empty((4, 0)),
                               augmented_mean=augmented, input_ids=np.empty((4, 0)),
                               input_mask=np.empty((4, 0)), labels=np.array([0, 1, 0, 1]))
        params = {"hidden_dim": 4, "dropout": 0.1, "learning_rate": 0.0003, "weight_decay": 0.0001}
        train, query = np.array([0, 1]), np.array([2, 3])
        original = fit_predict("mean_mlp_aug2", train, query, bundle, params, 11, 2, None, {}, {})[0]
        bundle.augmented_mean[query] = 1_000
        altered = fit_predict("mean_mlp_aug2", train, query, bundle, params, 11, 2, None, {}, {})[0]
        np.testing.assert_array_equal(original, altered)

    def test_all_variant_metadata_match_the_final_config(self):
        config_hash = hashlib.sha256((ROOT / "configs_v2/phase7_expert_ablations.yaml").read_bytes()).hexdigest()
        feature_hash = hashlib.sha256((ROOT / "results_v2/phase7_feature_metadata.json").read_bytes()).hexdigest()
        for variant in self.predictions.variant.unique():
            metadata = json.loads((ROOT / f"results_v2/phase7_work/{variant}_metadata.json").read_text())
            self.assertEqual(metadata["config_sha256"], config_hash, variant)
            if variant != "dmpnn":
                self.assertEqual(metadata["feature_metadata_sha256"], feature_hash, variant)


if __name__ == "__main__":
    unittest.main()
