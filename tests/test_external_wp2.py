"""Offline checks for the frozen Work Package 2 data audit."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from curate_external_wp2 import numeric_value, sha256  # noqa: E402


class ExternalWp2Tests(unittest.TestCase):
    def test_numeric_value_requires_positive_finite(self) -> None:
        self.assertEqual(numeric_value("100.0"), 100.0)
        for value in ("0", "nan", "inf", None):
            self.assertIsNone(numeric_value(value))

    def test_frozen_snapshot_and_cohort_invariants(self) -> None:
        snapshot = ROOT / "data/external_v2/chembl_37_wp2/snapshot_manifest.json"
        if not snapshot.exists():
            self.skipTest("External snapshot has not been fetched")
        manifest = json.loads(snapshot.read_text(encoding="utf-8"))
        self.assertEqual(manifest["chembl_release"], "ChEMBL_37")
        for item in manifest["files"]:
            self.assertEqual(sha256(ROOT / item["path"]), item["sha256"])
        cohort = pd.read_csv(ROOT / "data/external_v2/chembl_37_wp2/eligible_cohort.csv")
        original = pd.read_csv(ROOT / "data/processed_v2/bace_standardized.csv")
        self.assertFalse(cohort.duplicated(["assay_chembl_id", "standardized_smiles"]).any())
        self.assertEqual(set(cohort["label"]), {0, 1})
        self.assertTrue(set(cohort["standardized_smiles"]).isdisjoint(set(original["standardized_smiles"])))
        self.assertTrue(set(cohort["scaffold"]).isdisjoint(set(original["scaffold"])))
        assay_groups = [set(group["standardized_smiles"]) for _, group in cohort.groupby("assay_chembl_id")]
        self.assertTrue(assay_groups[0].isdisjoint(assay_groups[1]))
        for _, group in cohort.groupby("assay_chembl_id"):
            self.assertGreaterEqual(group["label"].value_counts().min(), 30)
        summary = json.loads((ROOT / "results_v2/external_wp2_audit_summary.json").read_text())
        self.assertTrue(all(
            item["original_structure_overlap_records"] == 0
            and item["original_parent_chembl_overlap_records"] == 0
            for item in summary["assays"].values()
        ))
        source_summary = json.loads((ROOT / "results_v2/external_wp2_original_label_summary.json").read_text())
        self.assertEqual(source_summary["original_si_ic50_threshold_disagreements"], 0)
        self.assertEqual(source_summary["deepchem_class_threshold_disagreements"], 17)
        lock_path = ROOT / "data/external_v2/chembl_37_wp2/cohort_lock.json"
        if lock_path.exists():
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
            self.assertEqual(lock["primary_assay"], "CHEMBL5736880")
            for path, checksum in lock["files"].items():
                self.assertEqual(sha256(ROOT / path), checksum)


if __name__ == "__main__":
    unittest.main()
