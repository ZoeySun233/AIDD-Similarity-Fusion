import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import preprocessing_v2  # noqa: E402


class Phase1StandardizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = preprocessing_v2.yaml.safe_load(
            (ROOT / "configs_v2/phase1_data_standardization.yaml").read_text()
        )
        cls.data = pd.read_csv(ROOT / "data/processed_v2/bace_standardized.csv", keep_default_na=False)
        cls.audit = pd.read_csv(ROOT / "results_v2/data_audit.csv", keep_default_na=False)
        cls.label_audit = pd.read_csv(ROOT / "results_v2/label_consistency_audit.csv", keep_default_na=False)

    def test_standardization_keeps_every_raw_source_row(self):
        raw = pd.read_csv(ROOT / "data/raw/bace.csv")
        self.assertEqual(len(self.data), len(raw))
        self.assertEqual(self.data["source_row"].tolist(), list(range(len(raw))))

    def test_largest_fragment_selection_is_traceable(self):
        example = preprocessing_v2.standardize_smiles("CCN.Cl", self.config)
        self.assertEqual(example["component_count"], 2)
        self.assertEqual(example["largest_fragment_smiles"], "CCN")
        self.assertIn("largest_fragment_selected", example["standardization_action_trace"])

    def test_labels_are_preserved(self):
        raw = pd.read_csv(ROOT / "data/raw/bace.csv")
        self.assertEqual(self.data["Class"].tolist(), raw["Class"].tolist())
        self.assertEqual(self.data["pIC50"].tolist(), raw["pIC50"].tolist())

    def test_every_valid_structure_has_a_standardized_identity(self):
        valid = self.data["is_valid_smiles"].astype(str).str.lower().eq("true")
        self.assertTrue((self.data.loc[valid, "standardized_smiles"] != "").all())
        self.assertTrue((self.data.loc[valid, "standardized_inchikey"] != "").all())

    def test_label_audit_never_recommends_label_change(self):
        if not self.label_audit.empty:
            self.assertTrue((self.label_audit["recommended_action"] == "retain_label_and_review_manually").all())

    def test_cross_split_duplicate_count_is_reported(self):
        audit_value = self.audit.loc[self.audit["metric"] == "cross_legacy_split_duplicate_rows", "value"].iloc[0]
        observed = self.data["cross_legacy_split_duplicate"].astype(str).str.lower().eq("true").sum()
        self.assertEqual(int(audit_value), int(observed))


if __name__ == "__main__":
    unittest.main()
