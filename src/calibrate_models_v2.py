"""Leakage-safe Platt calibration for Phase 3 nested-CV probabilities."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
try:
    from audit_panel_alignment import require_matplotlib_panel_alignment
except ImportError:
    require_matplotlib_panel_alignment = None


mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 8, "xtick.labelsize": 6,
    "ytick.labelsize": 6, "legend.fontsize": 6, "axes.spines.right": False,
    "axes.spines.top": False, "axes.linewidth": 0.8, "legend.frameon": False,
    "svg.fonttype": "none", "pdf.fonttype": 42,
})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clipped_logit(probability: np.ndarray, clip: float) -> np.ndarray:
    value = np.clip(np.asarray(probability, dtype=float), clip, 1.0 - clip)
    return np.log(value / (1.0 - value))


def adaptive_bin_summary(y: np.ndarray, probability: np.ndarray, bins: int) -> pd.DataFrame:
    """Equal-frequency reliability bins, with deterministic tie handling."""
    ordered = pd.DataFrame({"label": y, "probability": probability}).sort_values("probability", kind="mergesort").reset_index(drop=True)
    edges = np.linspace(0, len(ordered), min(bins, len(ordered)) + 1, dtype=int)
    rows = []
    for left, right in zip(edges[:-1], edges[1:]):
        subset = ordered.iloc[left:right]
        rows.append({"n": len(subset), "mean_prediction": float(subset.probability.mean()), "observed_fraction": float(subset.label.mean())})
    return pd.DataFrame(rows)


def adaptive_ece(y: np.ndarray, probability: np.ndarray, bins: int) -> float:
    table = adaptive_bin_summary(y, probability, bins)
    return float((table["n"] / table["n"].sum() * (table["mean_prediction"] - table["observed_fraction"]).abs()).sum())


def calibration_slope_intercept(y: np.ndarray, probability: np.ndarray, clip: float) -> tuple[float, float]:
    model = LogisticRegression(C=1_000_000.0, solver="lbfgs", max_iter=1000).fit(clipped_logit(probability, clip).reshape(-1, 1), y)
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def evaluate(y: np.ndarray, probability: np.ndarray, config: dict[str, Any]) -> dict[str, float]:
    clip, bins = float(config["calibration"]["probability_clip"]), int(config["calibration"]["adaptive_ece_bins"])
    slope, intercept = calibration_slope_intercept(y, probability, clip)
    return {
        "brier_score": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, np.clip(probability, clip, 1.0 - clip))),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "adaptive_ece": adaptive_ece(y, probability, bins),
        "pr_auc": float(average_precision_score(y, probability)),
        "roc_auc": float(roc_auc_score(y, probability)),
    }


def fit_platt(y: np.ndarray, probability: np.ndarray, config: dict[str, Any]) -> LogisticRegression:
    spec = config["calibration"]["platt"]
    return LogisticRegression(C=float(spec["C"]), solver=spec["solver"], max_iter=int(spec["max_iter"])).fit(
        clipped_logit(probability, float(config["calibration"]["probability_clip"])).reshape(-1, 1), y
    )


def platt_predict(model: LogisticRegression, probability: np.ndarray, clip: float) -> np.ndarray:
    return model.predict_proba(clipped_logit(probability, clip).reshape(-1, 1))[:, 1]


def make_reliability_figure(data: pd.DataFrame, config: dict[str, Any], stem: Path) -> None:
    """Plot aggregate outer-test calibration; metrics remain context-level in CSV."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), constrained_layout=True)
    colors = {"raw": "#8D99AE", "platt": "#267A9B", "isotonic": "#D66A43"}
    labels = {"raw": "Raw", "platt": "Platt", "isotonic": "Isotonic (sensitivity)"}
    for axis, (model, title) in zip(axes, [("morgan_rf", "Morgan RF"), ("frozen_chemberta_mlp", "Frozen ChemBERTa + MLP")]):
        subset = data[(data.model == model) & (data.split == "outer_test")]
        for method, column in [("raw", "probability_raw"), ("platt", "probability_platt"), ("isotonic", "probability_isotonic")]:
            table = adaptive_bin_summary(subset.label.to_numpy(), subset[column].to_numpy(), int(config["calibration"]["adaptive_ece_bins"]))
            axis.plot(table.mean_prediction, table.observed_fraction, marker="o", markersize=3.2, linewidth=1.2, color=colors[method], label=labels[method])
        axis.plot([0, 1], [0, 1], color="#555555", linewidth=0.8, linestyle="--", zorder=0)
        axis.set(title=title, xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability", ylabel="Observed active fraction")
        axis.legend(loc="upper left")
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    if require_matplotlib_panel_alignment is not None:
        require_matplotlib_panel_alignment(fig, json_out=f"{stem}.alignment-layout.json", overlay_svg=f"{stem}.alignment.svg", tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def run(config_path: Path) -> dict[str, Path]:
    config = yaml.safe_load(config_path.read_text())
    input_path = ROOT / config["inputs"]["base_predictions_csv"]
    predictions = pd.read_csv(input_path)
    required = {"CID", "source_row", "repeat", "outer_fold", "seed", "model", "split", "label", "probability"}
    if missing := required.difference(predictions.columns):
        raise ValueError(f"Base prediction table lacks columns: {sorted(missing)}")
    keys = ["repeat", "outer_fold", "seed", "model"]
    output_rows: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    clip = float(config["calibration"]["probability_clip"])
    for context, frame in predictions.groupby(keys, sort=True):
        repeat, outer_fold, seed, model_name = context
        oof = frame[frame.split == "oof_development"].copy()
        test = frame[frame.split == "outer_test"].copy()
        if len(oof) + len(test) != 1513 or oof.source_row.duplicated().any() or test.source_row.duplicated().any():
            raise ValueError(f"Incomplete Phase 3 context: {context}")
        platt = fit_platt(oof.label.to_numpy(dtype=int), oof.probability.to_numpy(), config)
        isotonic = IsotonicRegression(out_of_bounds="clip").fit(oof.probability.to_numpy(), oof.label.to_numpy(dtype=int))
        complete = pd.concat([oof, test], ignore_index=True)
        complete = complete.rename(columns={"probability": "probability_raw"})
        complete["probability_platt"] = platt_predict(platt, complete.probability_raw.to_numpy(), clip)
        complete["probability_isotonic"] = isotonic.predict(complete.probability_raw.to_numpy())
        complete["calibration_fit_split"] = "oof_development"
        output_rows.append(complete)
        audit_rows.append({
            "repeat": repeat, "outer_fold": outer_fold, "seed": seed, "model": model_name,
            "calibration_fit_split": "oof_development", "fit_rows": len(oof), "outer_test_rows": len(test),
            "fit_positive_fraction": float(oof.label.mean()), "platt_coefficient": float(platt.coef_[0, 0]),
            "platt_intercept": float(platt.intercept_[0]), "isotonic_sensitivity_enabled": True,
        })
        for method, column in [("raw", "probability_raw"), ("platt", "probability_platt"), ("isotonic", "probability_isotonic")]:
            metric_rows.append({"repeat": repeat, "outer_fold": outer_fold, "seed": seed, "model": model_name, "evaluation_split": "outer_test", "calibration_method": method, **evaluate(test.label.to_numpy(dtype=int), complete.loc[complete.split == "outer_test", column].to_numpy(), config)})
    calibrated = pd.concat(output_rows, ignore_index=True).sort_values(keys + ["split", "source_row"])
    expected_rows = len(predictions)
    if len(calibrated) != expected_rows or calibrated[["probability_raw", "probability_platt", "probability_isotonic"]].isna().any().any():
        raise RuntimeError("Calibrated prediction table violates its completeness contract.")
    outputs = config["outputs"]
    prediction_path, audit_path, metric_path = (ROOT / outputs[key] for key in ("calibrated_predictions_csv", "calibration_fit_audit_csv", "calibration_metrics_csv"))
    metadata_path, figure_stem = ROOT / outputs["metadata_json"], ROOT / outputs["reliability_figure_stem"]
    for path in (prediction_path, audit_path, metric_path, metadata_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    calibrated.to_csv(prediction_path, index=False)
    pd.DataFrame(audit_rows).to_csv(audit_path, index=False)
    context_metrics = pd.DataFrame(metric_rows)
    summary = context_metrics.groupby(["model", "calibration_method", "evaluation_split"], as_index=False).agg(
        n_outer_contexts=("brier_score", "size"),
        **{f"{metric}_{stat}": (metric, stat) for metric in ["brier_score", "log_loss", "calibration_slope", "calibration_intercept", "adaptive_ece", "pr_auc", "roc_auc"] for stat in ("mean", "std")},
    )
    pd.concat([context_metrics.assign(row_type="outer_context"), summary.assign(row_type="summary")], ignore_index=True, sort=False).to_csv(metric_path, index=False)
    make_reliability_figure(calibrated, config, figure_stem)
    metadata = {
        "phase": "phase4_probability_calibration", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path.relative_to(ROOT)), "input_sha256": sha256_file(input_path),
        "config": str(config_path.relative_to(ROOT)), "config_sha256": sha256_file(config_path),
        "primary_method": "platt_logit", "isotonic_role": "sensitivity_analysis_only",
        "fit_policy": "Calibrators are fit only on the matching outer context's inner OOF-development predictions; outer-test labels are excluded.",
        "fusion_policy": "Fixed and dynamic fusion must use probability_platt from this table as identical calibrated component inputs.",
        "n_contexts": len(audit_rows), "n_calibrated_rows": len(calibrated),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return {"predictions": prediction_path, "metrics": metric_path, "fit_audit": audit_path, "reliability_figure": figure_stem.with_suffix(".pdf"), "metadata": metadata_path}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs_v2/phase4_probability_calibration.yaml")
    args = parser.parse_args()
    for name, path in run(args.config.resolve()).items():
        print(f"{name}: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
