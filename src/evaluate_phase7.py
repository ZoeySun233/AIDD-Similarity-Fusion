"""Consolidate Phase 7 exploratory expert ablations on paired outer folds."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression

from train_base_models_v2 import sha256_file


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_VARIANTS = (
    "mean_mlp", "cls_mlp", "attention_mlp", "mean_logistic", "mean_mlp_aug2",
    "mean_mlp_ft_last0", "mean_mlp_ft_last1", "mean_mlp_ft_last2",
)


def adaptive_ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    order = np.argsort(p, kind="mergesort")
    groups = np.array_split(order, min(bins, len(order)))
    return float(sum(len(group) / len(y) * abs(float(y[group].mean()) - float(p[group].mean()))
                     for group in groups if len(group)))


def calibration_line(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    clipped = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(logit, y)
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def run(config_path: Path, variants: tuple[str, ...]) -> dict[str, Path]:
    config = yaml.safe_load(config_path.read_text())
    work = ROOT / "results_v2/phase7_work"
    metrics, predictions, searches, audits = [], [], [], []
    config_sha256 = sha256_file(config_path)
    feature_sha256 = sha256_file(ROOT / config["outputs"]["feature_metadata_json"])
    variant_metadata_sha256 = {}
    for variant in variants:
        paths = {name: work / f"{variant}_{name}.csv" for name in ("metrics", "selection", "fit_audit")}
        prediction_path = work / f"{variant}_predictions.csv.gz"
        metadata_path = work / f"{variant}_metadata.json"
        for path in (*paths.values(), prediction_path, metadata_path):
            if not path.exists():
                raise FileNotFoundError(path)
        metadata = json.loads(metadata_path.read_text())
        if metadata["config_sha256"] != config_sha256:
            raise ValueError(f"Configuration checksum mismatch for {variant}.")
        if variant != "dmpnn" and metadata["feature_metadata_sha256"] != feature_sha256:
            raise ValueError(f"Frozen-feature checksum mismatch for {variant}.")
        variant_metadata_sha256[variant] = sha256_file(metadata_path)
        metric = pd.read_csv(paths["metrics"])
        prediction = pd.read_csv(prediction_path)
        selection = pd.read_csv(paths["selection"])
        audit = pd.read_csv(paths["fit_audit"])
        if len(metric) != 90 or len(prediction) != 2 * 1513 * 3:
            raise ValueError(f"Incomplete 2x5x3 outer contexts for {variant}.")
        if prediction.duplicated(["repeat", "outer_fold", "seed", "source_row"]).any():
            raise ValueError(f"Duplicate outer prediction for {variant}.")
        if not prediction.probability.between(0, 1).all():
            raise ValueError(f"Invalid probability for {variant}.")
        if not (audit[["source_overlap_n", "scaffold_overlap_n", "outer_test_fit_overlap_n"]] == 0).all().all():
            raise ValueError(f"Fit boundary violated for {variant}.")
        rows = []
        for context, frame in prediction.groupby(["repeat", "outer_fold", "seed"]):
            intercept, slope = calibration_line(frame.label.to_numpy(int), frame.probability.to_numpy(float))
            rows.append({"repeat": context[0], "outer_fold": context[1], "seed": context[2],
                         "variant": variant, "stratum": "all", "adaptive_ece": adaptive_ece(
                             frame.label.to_numpy(int), frame.probability.to_numpy(float)),
                         "calibration_intercept": intercept, "calibration_slope": slope})
        metric = metric.merge(pd.DataFrame(rows), on=["repeat", "outer_fold", "seed", "variant", "stratum"], how="left", validate="one_to_one")
        metrics.append(metric)
        predictions.append(prediction)
        searches.append(selection)
        audits.append(audit)
    metric_table = pd.concat(metrics, ignore_index=True)
    prediction_table = pd.concat(predictions, ignore_index=True)
    search_table = pd.concat(searches, ignore_index=True)
    audit_table = pd.concat(audits, ignore_index=True)
    baseline = metric_table[metric_table.variant == "mean_mlp"][
        ["repeat", "outer_fold", "seed", "stratum", "average_precision", "brier_score"]
    ].rename(columns={"average_precision": "mean_mlp_average_precision", "brier_score": "mean_mlp_brier_score"})
    metric_table = metric_table.merge(baseline, on=["repeat", "outer_fold", "seed", "stratum"], validate="many_to_one")
    metric_table["delta_ap_vs_mean_mlp"] = metric_table.average_precision - metric_table.mean_mlp_average_precision
    metric_table["delta_brier_vs_mean_mlp"] = metric_table.brier_score - metric_table.mean_mlp_brier_score
    fold = metric_table.groupby(["variant", "stratum", "repeat", "outer_fold"], as_index=False).agg(
        n_molecules=("n_molecules", "first"), average_precision=("average_precision", "mean"),
        brier_score=("brier_score", "mean"), delta_ap_vs_rf=("delta_ap_vs_rf", "mean"),
        delta_brier_vs_rf=("delta_brier_vs_rf", "mean"),
        delta_ap_vs_mean_mlp=("delta_ap_vs_mean_mlp", "mean"),
        delta_brier_vs_mean_mlp=("delta_brier_vs_mean_mlp", "mean"),
        adaptive_ece=("adaptive_ece", "mean"),
        calibration_intercept=("calibration_intercept", "mean"),
        calibration_slope=("calibration_slope", "mean"),
        approx_inference_ms_per_molecule=("approx_inference_ms_per_molecule", "mean"),
    )
    rule = config["experiment"]["entry_rule"]
    decisions = []
    for variant in variants:
        all_fold = fold[(fold.variant == variant) & (fold.stratum == "all")]
        low_fold = fold[(fold.variant == variant) & (fold.stratum == "low_similarity")]
        wins = int((low_fold.delta_ap_vs_rf > 0).sum())
        overall_wins_vs_mean = int((all_fold.delta_ap_vs_mean_mlp > 0).sum())
        brier_delta = float(all_fold.delta_brier_vs_rf.mean())
        global_delta = float(all_fold.delta_ap_vs_mean_mlp.mean())
        passes = (wins >= int(rule["minimum_low_similarity_win_folds_vs_rf"])
                  and brier_delta <= float(rule["maximum_brier_increase_vs_rf"])
                  and global_delta >= float(rule["minimum_overall_ap_improvement_vs_mean_mlp"]))
        decisions.append({"variant": variant, "low_similarity_win_folds_vs_rf": wins,
                          "overall_win_folds_vs_mean_mlp": overall_wins_vs_mean,
                          "mean_low_similarity_delta_ap_vs_rf": float(low_fold.delta_ap_vs_rf.mean()),
                          "mean_overall_delta_ap_vs_mean_mlp": global_delta,
                          "mean_overall_delta_brier_vs_rf": brier_delta,
                          "mean_overall_ap": float(all_fold.average_precision.mean()),
                          "mean_overall_brier": float(all_fold.brier_score.mean()),
                          "mean_adaptive_ece": float(all_fold.adaptive_ece.mean()),
                          "mean_calibration_intercept": float(all_fold.calibration_intercept.mean()),
                          "mean_calibration_slope": float(all_fold.calibration_slope.mean()),
                          "mean_inference_ms_per_molecule": float(all_fold.approx_inference_ms_per_molecule.mean()),
                          "passes_exploratory_entry_rule": passes})
    output = config["outputs"]
    tables = {
        "predictions_csv_gz": prediction_table.sort_values(["variant", "repeat", "outer_fold", "seed", "source_row"]),
        "ablations_csv": metric_table.sort_values(["variant", "stratum", "repeat", "outer_fold", "seed"]),
        "hyperparameter_csv": search_table.sort_values(["variant", "repeat", "outer_fold", "candidate_index", "inner_fold"]),
        "fit_audit_csv": audit_table,
    }
    output_paths = {key: ROOT / output[key] for key in tables}
    for key, table in tables.items():
        output_paths[key].parent.mkdir(parents=True, exist_ok=True)
        if key == "predictions_csv_gz":
            table.to_csv(output_paths[key], index=False, compression={"method": "gzip", "mtime": 0})
        else:
            table.to_csv(output_paths[key], index=False)
    fold_path = ROOT / "results_v2/phase7_fold_comparisons.csv"
    decision_path = ROOT / "results_v2/phase7_expert_decisions.csv"
    fold.to_csv(fold_path, index=False)
    pd.DataFrame(decisions).to_csv(decision_path, index=False)
    output_paths["fold_comparisons_csv"] = fold_path
    output_paths["expert_decisions_csv"] = decision_path
    metadata_path = ROOT / output["run_metadata_json"]
    metadata_path.write_text(json.dumps({
        "phase": "phase7_expert_ablations", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "post_phase5_exploratory_not_independent_confirmation",
        "config_sha256": config_sha256, "variants": list(variants),
        "variant_metadata_sha256": variant_metadata_sha256,
        "source_sha256": {
            name: sha256_file(ROOT / "src" / f"{name}.py")
            for name in ("phase7_features", "phase7_models", "phase7_ablation_runner", "phase7_dmpnn")
        },
        "output_sha256": {name: sha256_file(path) for name, path in output_paths.items()},
        "input_sha256": {
            "standardized_csv": sha256_file(ROOT / config["inputs"]["standardized_csv"]),
            "calibrated_rf_csv": sha256_file(ROOT / config["inputs"]["calibrated_rf_csv"]),
            "feature_metadata_json": feature_sha256,
            "outer_manifests": {
                str(repeat): sha256_file(ROOT / config["inputs"]["split_directory"] / f"outer_repeat_{repeat}.csv")
                for repeat in config["experiment"]["repeats"]
            },
        },
        "outer_fold_contexts": 10, "seeds_per_fold": 3,
        "prediction_rows": len(prediction_table), "metric_rows": len(metric_table),
        "fit_audit_rows": len(audit_table),
        "primary_selection_policy": "Only outer-development inner folds select hyperparameters and epochs; Platt uses OOF development predictions.",
        "statistical_unit": "Three seeds averaged within each repeat x outer fold; repeats reuse molecules.",
        "script_sha256": sha256_file(Path(__file__)),
    }, indent=2) + "\n")
    output_paths["run_metadata_json"] = metadata_path
    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs_v2/phase7_expert_ablations.yaml")
    parser.add_argument("--variants", default=",".join(EXPECTED_VARIANTS),
                        help="Comma-separated completed variants; mean_mlp is required.")
    args = parser.parse_args()
    variants = tuple(value.strip() for value in args.variants.split(",") if value.strip())
    if "mean_mlp" not in variants:
        raise ValueError("Mean MLP reference is required.")
    for name, path in run(args.config.resolve(), variants).items():
        print(f"{name}: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
