import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import splits_v2  # noqa: E402


class Phase2NestedScaffoldCVTests(unittest.TestCase):
    def test_outer_repeats_are_complete_and_scaffold_disjoint(self):
        for repeat in (1, 2):
            data = pd.read_csv(ROOT / f"data/splits_v2/outer_repeat_{repeat}.csv", keep_default_na=False)
            self.assertEqual(len(data), 1513)
            self.assertEqual(set(data["outer_fold"]), {1, 2, 3, 4, 5})
            self.assertEqual(splits_v2.scaffold_overlap_count(data, "outer_fold", "scaffold"), 0)
            self.assertEqual(splits_v2.structure_group_overlap_count(data, "outer_fold", "v2_structure_group_id"), 0)

    def test_outer_repeats_are_distinct(self):
        repeat_one = pd.read_csv(ROOT / "data/splits_v2/outer_repeat_1.csv").sort_values("source_row")
        repeat_two = pd.read_csv(ROOT / "data/splits_v2/outer_repeat_2.csv").sort_values("source_row")
        self.assertTrue((repeat_one["outer_fold"] != repeat_two["outer_fold"]).any())

    def test_inner_folds_exclude_the_corresponding_outer_test_fold(self):
        for repeat in (1, 2):
            outer = pd.read_csv(ROOT / f"data/splits_v2/outer_repeat_{repeat}.csv", keep_default_na=False)
            for outer_fold in range(1, 6):
                inner = pd.read_csv(ROOT / f"data/splits_v2/inner_repeat_{repeat}_outer_fold_{outer_fold}.csv", keep_default_na=False)
                self.assertEqual(set(inner["source_row"]), set(outer.loc[outer.outer_fold != outer_fold, "source_row"]))
                self.assertEqual(set(inner["inner_fold"]), {1, 2, 3})
                self.assertEqual(splits_v2.scaffold_overlap_count(inner, "inner_fold", "scaffold"), 0)
                self.assertEqual(splits_v2.structure_group_overlap_count(inner, "inner_fold", "v2_structure_group_id"), 0)

    def test_split_audit_confirms_primary_scaffold_integrity(self):
        audit = pd.read_csv(ROOT / "results_v2/split_audit.csv")
        primary = audit[audit.split_family.str.startswith("primary_")]
        self.assertTrue((primary.scaffold_groups_crossing_folds == 0).all())
        self.assertTrue((primary.structure_groups_crossing_folds == 0).all())
        self.assertTrue((primary.n_compounds > 0).all())


if __name__ == "__main__":
    unittest.main()
