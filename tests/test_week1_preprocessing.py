import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import preprocessing  # noqa: E402


class Week1DataContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = pd.read_csv(ROOT / "data/processed/bace_scaffold_split.csv")

    def test_every_row_has_valid_canonical_smiles(self):
        self.assertTrue(self.data["canonical_smiles"].notna().all())

    def test_canonical_smiles_are_unique(self):
        self.assertEqual(int(self.data["canonical_smiles"].duplicated().sum()), 0)

    def test_split_fractions_are_within_one_percentage_point(self):
        observed = self.data["split"].value_counts(normalize=True)
        for name, expected in zip(
            preprocessing.SPLIT_NAMES, preprocessing.SPLIT_FRACTIONS
        ):
            self.assertLessEqual(abs(float(observed[name]) - float(expected)), 0.01)

    def test_scaffolds_do_not_cross_splits(self):
        split_sets = {
            name: set(self.data.loc[self.data["split"] == name, "scaffold"])
            for name in preprocessing.SPLIT_NAMES
        }
        self.assertTrue(split_sets["train"].isdisjoint(split_sets["valid"]))
        self.assertTrue(split_sets["train"].isdisjoint(split_sets["test"]))
        self.assertTrue(split_sets["valid"].isdisjoint(split_sets["test"]))


if __name__ == "__main__":
    unittest.main()
