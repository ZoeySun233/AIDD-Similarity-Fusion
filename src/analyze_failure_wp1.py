"""Post-hoc error atlas from frozen Phase 4-6 outer-test predictions.

This script is descriptive. It neither fits a model nor chooses a new gate.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

from evaluate_phase6 import read_outer_predictions, scaffold_similarity_frames
from fusion_v2 import entropy
from train_base_models_v2 import sha256_file


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs_v2/phase6_statistical_evaluation.yaml"
METHODS = {
    "A1_calibrated_rf": "rf",
    "B1_calibrated_chemberta": "chemberta",
    "C1_fixed_weight": "fixed",
    "D2_similarity_entropy": "dynamic",
}
OUT = ROOT / "results_v2"


def safe_ap(y: np.ndarray, p: np.ndarray) -> float:
    return float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def make_atlas(outer: pd.DataFrame, similarity: pd.DataFrame, development: dict, data: pd.DataFrame) -> pd.DataFrame:
    selected = outer[outer.method.isin(METHODS)].copy()
    keys = ["repeat", "outer_fold", "source_row", "label", "method"]
    if selected.groupby(keys).seed.nunique().ne(3).any():
        raise ValueError("Expected three seeds for each frozen prediction identity.")
    averaged = selected.groupby(keys, as_index=False).probability.mean()
    wide = averaged.pivot(index=["repeat", "outer_fold", "source_row", "label"],
                          columns="method", values="probability").reset_index()
    if wide[list(METHODS)].isna().any().any() or len(wide) != 2 * len(data):
        raise ValueError("Incomplete paired outer-test prediction matrix.")
    wide = wide.rename(columns=METHODS)
    if wide.duplicated(["repeat", "source_row"]).any():
        raise ValueError("Each molecule must be tested once per repeat.")
    annotation = data[["source_row", "CID", "Class", "pIC50", "scaffold", "v2_structure_group_id",
                       "standardized_structure_conflict", "class_pic50_consistency_review"]]
    wide = wide.merge(annotation, on="source_row", validate="many_to_one")
    if not np.array_equal(wide.label.to_numpy(int), wide.Class.to_numpy(int)):
        raise ValueError("Labels disagree with standardized data.")
    wide = wide.merge(similarity, on=["repeat", "outer_fold", "source_row"], validate="one_to_one")
    cutoffs = pd.DataFrame([{"repeat": r, "outer_fold": f,
                            "high_similarity_cutoff": float(frame.max_similarity.quantile(0.75))}
                            for (r, f), frame in development.items()])
    wide = wide.merge(cutoffs, on=["repeat", "outer_fold"], validate="many_to_one")
    wide["similarity_band"] = np.select(
        [wide.max_similarity <= wide.low_similarity_cutoff,
         wide.max_similarity >= wide.high_similarity_cutoff],
        ["low", "high"], default="middle")
    wide["topk_label_purity"] = np.maximum(wide.topk_active_fraction, 1 - wide.topk_active_fraction)
    wide["topk_label_agreement_posthoc"] = np.where(wide.label == 1, wide.topk_active_fraction,
                                                       1 - wide.topk_active_fraction)
    wide["expert_disagreement"] = np.abs(wide.rf - wide.chemberta)
    wide["entropy_difference_chem_minus_rf"] = entropy(wide.chemberta.to_numpy()) - entropy(wide.rf.to_numpy())
    for name in METHODS.values():
        wide[f"squared_error_{name}"] = np.square(wide.label - wide[name])
    wide["rf_advantage_brier"] = wide.squared_error_chemberta - wide.squared_error_rf
    wide["dynamic_minus_fixed_brier"] = wide.squared_error_dynamic - wide.squared_error_fixed
    wide["rf_wrong_at_0_5"] = (wide.rf >= 0.5).astype(int) != wide.label
    wide["chemberta_wrong_at_0_5"] = (wide.chemberta >= 0.5).astype(int) != wide.label
    wide["error_overlap_at_0_5"] = np.select(
        [wide.rf_wrong_at_0_5 & wide.chemberta_wrong_at_0_5,
         wide.rf_wrong_at_0_5 & ~wide.chemberta_wrong_at_0_5,
         ~wide.rf_wrong_at_0_5 & wide.chemberta_wrong_at_0_5],
        ["both_wrong", "rf_only_wrong", "chemberta_only_wrong"], default="neither_wrong")
    return wide.sort_values(["repeat", "source_row"]).reset_index(drop=True)


def score_groups(atlas: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for key, frame in atlas.groupby(group_cols, sort=True, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        y = frame.label.to_numpy(int)
        item = dict(zip(group_cols, key))
        item.update(n_molecules=len(frame), n_active=int(y.sum()), active_fraction=float(y.mean()),
                    n_scaffolds=int(frame.scaffold.nunique()),
                    both_wrong_at_0_5=int((frame.error_overlap_at_0_5 == "both_wrong").sum()),
                    rf_only_wrong_at_0_5=int((frame.error_overlap_at_0_5 == "rf_only_wrong").sum()),
                    chemberta_only_wrong_at_0_5=int((frame.error_overlap_at_0_5 == "chemberta_only_wrong").sum()),
                    mean_expert_disagreement=float(frame.expert_disagreement.mean()),
                    rf_better_molecule_fraction=float((frame.rf_advantage_brier > 0).mean()))
        for name in METHODS.values():
            item[f"brier_{name}"] = float(frame[f"squared_error_{name}"].mean())
            item[f"ap_{name}"] = safe_ap(y, frame[name].to_numpy(float))
        item["ap_rf_minus_chemberta"] = item["ap_rf"] - item["ap_chemberta"]
        item["ap_dynamic_minus_fixed"] = item["ap_dynamic"] - item["ap_fixed"]
        item["brier_rf_minus_chemberta"] = item["brier_rf"] - item["brier_chemberta"]
        item["brier_dynamic_minus_fixed"] = item["brier_dynamic"] - item["brier_fixed"]
        rows.append(item)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def mechanism_table(atlas: pd.DataFrame) -> pd.DataFrame:
    predictors = ("max_similarity", "topk_mean_similarity", "topk_label_purity",
                  "topk_label_agreement_posthoc", "entropy_difference_chem_minus_rf",
                  "expert_disagreement")
    rows = []
    for (repeat, fold), frame in atlas.groupby(["repeat", "outer_fold"], sort=True):
        item = {"repeat": repeat, "outer_fold": fold, "n_molecules": len(frame),
                "n_active": int(frame.label.sum()),
                "residual_pearson_rf_chemberta": float(np.corrcoef(
                    frame.label - frame.rf, frame.label - frame.chemberta)[0, 1]),
                "both_wrong_at_0_5": int((frame.error_overlap_at_0_5 == "both_wrong").sum()),
                "either_wrong_at_0_5": int((frame.rf_wrong_at_0_5 | frame.chemberta_wrong_at_0_5).sum()),
                "rf_better_molecule_fraction": float((frame.rf_advantage_brier > 0).mean()),
                "mean_absolute_dynamic_fixed_probability_gap": float(np.abs(frame.dynamic - frame.fixed).mean()),
                "max_absolute_dynamic_fixed_probability_gap": float(np.abs(frame.dynamic - frame.fixed).max())}
        for predictor in predictors:
            item[f"rho_{predictor}_vs_rf_advantage"] = safe_spearman(
                frame[predictor].to_numpy(float), frame.rf_advantage_brier.to_numpy(float))
        rows.append(item)
    return pd.DataFrame(rows)


def issue_review(atlas: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    flags = pd.read_csv(OUT / "label_consistency_audit.csv", keep_default_na=False)
    if not set(flags.source_row).issubset(set(data.source_row)):
        raise ValueError("Label audit contains an unknown source row.")
    per_molecule = atlas.groupby("source_row", as_index=False).agg(
        CID=("CID", "first"), Class=("Class", "first"), pIC50=("pIC50", "first"),
        scaffold=("scaffold", "first"), brier_rf=("squared_error_rf", "mean"),
        brier_chemberta=("squared_error_chemberta", "mean"),
        brier_fixed=("squared_error_fixed", "mean"),
        brier_dynamic=("squared_error_dynamic", "mean"),
        mean_max_similarity=("max_similarity", "mean"))
    return flags.merge(per_molecule, on=["source_row", "CID", "Class", "pIC50"],
                       validate="many_to_one", suffixes=("", "_atlas"))


def main() -> None:
    config = yaml.safe_load(CONFIG.read_text())
    input_paths = [ROOT / config["inputs"][name] for name in
                   ("standardized_csv", "phase5_predictions_csv", "calibrated_base_predictions_csv",
                    "phase1_activity_cliffs_csv")]
    input_paths.extend([CONFIG, OUT / "label_consistency_audit.csv"])
    input_paths.extend(ROOT / config["inputs"]["split_directory"] / f"outer_repeat_{repeat}.csv"
                       for repeat in (1, 2))
    input_paths.extend(ROOT / config["inputs"]["split_directory"] /
                       f"inner_repeat_{repeat}_outer_fold_{fold}.csv"
                       for repeat in (1, 2) for fold in range(1, 6))
    data = pd.read_csv(input_paths[0], keep_default_na=False).sort_values("source_row").reset_index(drop=True)
    outer = read_outer_predictions(config, data)
    similarity, development = scaffold_similarity_frames(data, config)
    atlas = make_atlas(outer, similarity, development, data)
    strata = score_groups(atlas, ["repeat", "outer_fold", "similarity_band"])
    scaffolds = score_groups(atlas, ["repeat", "scaffold"])
    mechanisms = mechanism_table(atlas)
    issues = issue_review(atlas, data)
    cliffs = pd.read_csv(ROOT / config["inputs"]["phase1_activity_cliffs_csv"])
    cliff_sources = set(cliffs.loc[cliffs.potency_cliff_ge_1_log10.astype(bool), "source_row_a"]) | set(
        cliffs.loc[cliffs.potency_cliff_ge_1_log10.astype(bool), "source_row_b"])
    atlas["potency_cliff_posthoc"] = atlas.source_row.isin(cliff_sources)
    cliff_metrics = score_groups(atlas, ["repeat", "outer_fold", "potency_cliff_posthoc"])
    outputs = {
        "wp1_error_atlas.csv": atlas,
        "wp1_similarity_strata.csv": strata,
        "wp1_scaffold_atlas.csv": scaffolds,
        "wp1_mechanism_by_fold.csv": mechanisms,
        "wp1_label_issue_review.csv": issues,
        "wp1_cliff_strata.csv": cliff_metrics,
    }
    for name, frame in outputs.items():
        frame.to_csv(OUT / name, index=False)
    metadata = {
        "analysis": "posthoc_failure_analysis_only", "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed_policy": "average three seeds within each outer repeat and fold before error comparison",
        "repeat_policy": "same molecules occur in both repeats; rows are not independent biological samples",
        "similarity_band_policy": "outer-development inner-OOF max Tanimoto q25 and q75; no test-label selection",
        "label_agreement_policy": "topk_label_agreement_posthoc uses the query label and is not an inference feature",
        "cliff_policy": "label-derived Phase 1 diagnostic, not a prospective risk feature",
        "threshold_policy": "0.5 only for descriptive hard-error overlap; not a selected decision threshold",
        "input_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in input_paths},
        "output_rows": {name: len(frame) for name, frame in outputs.items()},
    }
    (OUT / "wp1_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata["output_rows"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    main()
