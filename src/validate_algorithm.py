"""Week 5 locked-test validation of the similarity-aware fusion algorithm."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from baseline import build_model
from chemberta import (
    ActivityMLP,
    encode_with_frozen_encoder,
    load_frozen_encoder,
    probabilities,
    save_embedding_cache,
    train_mlp,
)
from evaluate import classification_metrics
from fingerprints import MorganConfig, morgan_fingerprints, save_fingerprint_cache
from fusion import (
    dynamic_fusion,
    fixed_fusion,
    maximum_tanimoto_similarity,
    optimize_dynamic_weight,
    optimize_fixed_weight,
)


SEEDS = (11, 22, 33, 44, 55)
EXPERIMENTS = {
    "A": "Morgan RF",
    "B": "Frozen ChemBERTa MLP",
    "C": "Fixed-weight fusion",
    "D": "Dynamic similarity fusion",
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def median_runtime(function, repeats: int) -> tuple[float, object]:
    durations: list[float] = []
    output = None
    for _ in range(repeats):
        started = time.perf_counter()
        output = function()
        durations.append(time.perf_counter() - started)
    return float(np.median(durations)), output


def test_metrics_with_validation_threshold(
    y_valid: np.ndarray,
    valid_probabilities: np.ndarray,
    y_test: np.ndarray,
    test_probabilities: np.ndarray,
) -> tuple[dict[str, float], float]:
    threshold = classification_metrics(y_valid, valid_probabilities)["threshold"]
    return classification_metrics(y_test, test_probabilities, threshold=threshold), threshold


def safe_bin_metrics(
    y_true: np.ndarray, probabilities_: np.ndarray, threshold: float
) -> dict[str, float]:
    if len(np.unique(y_true)) < 2:
        predicted = (probabilities_ >= threshold).astype(int)
        return {
            "roc_auc": np.nan,
            "pr_auc": float(average_precision_score(y_true, probabilities_)),
            "brier_score": float(brier_score_loss(y_true, probabilities_)),
            "threshold": threshold,
            "f1": float(
                2 * np.sum((predicted == 1) & (y_true == 1))
                / max(1, 2 * np.sum((predicted == 1) & (y_true == 1))
                      + np.sum((predicted == 1) & (y_true == 0))
                      + np.sum((predicted == 0) & (y_true == 1)))
            ),
            "balanced_accuracy": np.nan,
            "mcc": np.nan,
        }
    return classification_metrics(y_true, probabilities_, threshold=threshold)


def aggregate_seed_metrics(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_columns = ["roc_auc", "pr_auc", "brier_score", "f1", "balanced_accuracy", "mcc"]
    rows: list[dict[str, object]] = []
    for (experiment, model), group in seed_metrics.groupby(["experiment", "model"], sort=False):
        row: dict[str, object] = {
            "experiment": experiment,
            "model": model,
            "n_seeds": int(len(group)),
        }
        for metric in metric_columns:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def save_figure(fig: plt.Figure, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_test_discrimination(aggregate: pd.DataFrame, output_dir: Path) -> None:
    ordered = ["A", "B", "C", "D"]
    table = aggregate.set_index("experiment").loc[ordered]
    y = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(3.50, 3.05), constrained_layout=True)
    ax.errorbar(
        table["roc_auc_mean"], y - 0.10, xerr=table["roc_auc_std"], fmt="o",
        color="#4C78A8", capsize=3, label="ROC-AUC",
    )
    ax.errorbar(
        table["pr_auc_mean"], y + 0.10, xerr=table["pr_auc_std"], fmt="s",
        color="#E07B54", capsize=3, label="PR-AUC",
    )
    ax.set_yticks(y, [f"{idx}: {EXPERIMENTS[idx]}" for idx in ordered])
    lower = max(0.0, float(table[["roc_auc_mean", "pr_auc_mean"]].min().min() - 0.06))
    ax.set_xlim(lower, 1.0)
    ax.set_xlabel("Locked-test score")
    ax.set_title("Five-seed locked-test discrimination")
    ax.invert_yaxis()
    ax.legend(loc="upper right")
    save_figure(fig, output_dir / "week5_test_discrimination")


def plot_similarity_bins(bin_metrics: pd.DataFrame, output_dir: Path) -> None:
    subset = bin_metrics[bin_metrics["experiment"].isin(["C", "D"])]
    summary = (
        subset.groupby(["experiment", "similarity_bin"], observed=True)["pr_auc"]
        .agg(["mean", "std"])
        .reset_index()
    )
    labels = ["Q1\nlowest", "Q2", "Q3", "Q4\nhighest"]
    x = np.arange(4)
    fig, ax = plt.subplots(figsize=(3.50, 2.90), constrained_layout=True)
    for experiment, color, marker in (("C", "#7F7F7F", "o"), ("D", "#6F4E7C", "s")):
        values = summary[summary["experiment"] == experiment].set_index("similarity_bin").loc[
            ["Q1", "Q2", "Q3", "Q4"]
        ]
        ax.errorbar(
            x, values["mean"], yerr=values["std"], marker=marker, color=color,
            linewidth=1.4, capsize=3, label=f"{experiment}: {EXPERIMENTS[experiment]}",
        )
    ax.set_xticks(x, labels)
    ax.set_xlabel("Validation-defined similarity quartile")
    ax.set_ylabel("Test PR-AUC (mean ± SD)")
    ax.set_title("Fusion performance across similarity regions")
    ax.legend(loc="lower left")
    save_figure(fig, output_dir / "week5_similarity_bin_pr_auc")


def plot_inference_time(timing: pd.DataFrame, output_dir: Path) -> None:
    ordered = ["A", "B", "C", "D"]
    table = timing.set_index("experiment").loc[ordered]
    y = np.arange(4)
    fig, ax = plt.subplots(figsize=(3.50, 2.90), constrained_layout=True)
    values = table["milliseconds_per_molecule"].to_numpy()
    ax.barh(y, values, color=["#4C78A8", "#E07B54", "#7F7F7F", "#6F4E7C"])
    ax.set_yticks(y, [f"{idx}: {EXPERIMENTS[idx]}" for idx in ordered])
    ax.set_xlim(0.0, float(values.max() * 1.18))
    for row, value in enumerate(values):
        ax.text(value + values.max() * 0.025, row, f"{value:.2f}", va="center", fontsize=8)
    ax.set_xlabel("Warm inference time per molecule (ms)")
    ax.set_title("End-to-end inference cost")
    ax.invert_yaxis()
    save_figure(fig, output_dir / "week5_inference_time")


def run(project_root: Path) -> dict[str, object]:
    data_dir = project_root / "data/processed"
    features_dir = data_dir / "features"
    results_dir = project_root / "results"
    figures_dir = results_dir / "figures"
    models_dir = project_root / "models"
    cache_dir = models_dir / "huggingface"
    for directory in (features_dir, results_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(data_dir / "train.csv")
    valid = pd.read_csv(data_dir / "valid.csv")
    test = pd.read_csv(data_dir / "test.csv")
    scaffold_sets = [set(frame["scaffold"].fillna("")) for frame in (train, valid, test)]
    if not all(scaffold_sets[i].isdisjoint(scaffold_sets[j]) for i in range(3) for j in range(i + 1, 3)):
        raise AssertionError("Scaffold leakage exists among train, validation, and test.")

    y_train = train["Class"].to_numpy(dtype=np.int8)
    y_valid = valid["Class"].to_numpy(dtype=np.int8)
    y_test = test["Class"].to_numpy(dtype=np.int8)
    morgan_config = MorganConfig(radius=2, n_bits=2048, include_chirality=True)
    with np.load(features_dir / "morgan_r2_2048_train.npz") as cache:
        X_train_fp = cache["X"]
    with np.load(features_dir / "morgan_r2_2048_valid.npz") as cache:
        X_valid_fp = cache["X"]

    rf_featurization_seconds, X_test_fp_object = median_runtime(
        lambda: morgan_fingerprints(test["canonical_smiles"], morgan_config), repeats=5
    )
    X_test_fp = np.asarray(X_test_fp_object, dtype=np.uint8)
    save_fingerprint_cache(
        features_dir / "morgan_r2_2048_test.npz", X_test_fp, y_test,
        test["canonical_smiles"], morgan_config, "test",
    )

    tokenizer, encoder, device = load_frozen_encoder(cache_dir, local_files_only=True)
    encode_with_frozen_encoder(test["canonical_smiles"].iloc[:8], tokenizer, encoder, device)
    encoder_seconds_values: list[float] = []
    X_test_raw = None
    test_encoder_metadata = None
    for _ in range(3):
        started = time.perf_counter()
        X_test_raw, test_encoder_metadata = encode_with_frozen_encoder(
            test["canonical_smiles"], tokenizer, encoder, device
        )
        encoder_seconds_values.append(time.perf_counter() - started)
    encoder_seconds = float(np.median(encoder_seconds_values))
    assert X_test_raw is not None and test_encoder_metadata is not None
    save_embedding_cache(
        features_dir / "chemberta_77m_mlm_test.npz", X_test_raw, y_test,
        test["canonical_smiles"], "test", test_encoder_metadata,
    )
    del encoder

    with np.load(features_dir / "chemberta_77m_mlm_train.npz") as cache:
        X_train_raw = cache["X"].astype(np.float32)
    with np.load(features_dir / "chemberta_77m_mlm_valid.npz") as cache:
        X_valid_raw = cache["X"].astype(np.float32)
    scaler = joblib.load(models_dir / "chemberta_embedding_scaler.joblib")
    X_train_emb = scaler.transform(X_train_raw).astype(np.float32)
    X_valid_emb = scaler.transform(X_valid_raw).astype(np.float32)
    X_test_emb = scaler.transform(X_test_raw).astype(np.float32)

    validation_similarity, _ = maximum_tanimoto_similarity(X_valid_fp, X_train_fp)
    test_similarity, nearest_train_indices = maximum_tanimoto_similarity(X_test_fp, X_train_fp)
    _, bin_edges = pd.qcut(validation_similarity, q=4, retbins=True, duplicates="drop")
    assignment_edges = bin_edges.copy()
    assignment_edges[0] = -np.inf
    assignment_edges[-1] = np.inf
    test_bins = pd.cut(
        test_similarity, bins=assignment_edges, labels=["Q1", "Q2", "Q3", "Q4"],
        include_lowest=True,
    )

    with (results_dir / "rf_baseline_summary.json").open(encoding="utf-8") as handle:
        rf_summary = json.load(handle)
    with (results_dir / "chemberta_baseline_summary.json").open(encoding="utf-8") as handle:
        chem_summary = json.load(handle)
    rf_params = rf_summary["best_params"]
    chem_params = chem_summary["best_params"]

    seed_metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    fusion_parameter_rows: list[dict[str, float | int]] = []
    bin_metric_rows: list[dict[str, object]] = []
    for seed in SEEDS:
        rf_model = build_model(rf_params, seed)
        rf_model.fit(X_train_fp, y_train)
        rf_valid = rf_model.predict_proba(X_valid_fp)[:, 1]
        rf_test = rf_model.predict_proba(X_test_fp)[:, 1]

        chem_model, chem_validation_details, _ = train_mlp(
            X_train_emb, y_train, X_valid_emb, y_valid, chem_params, seed
        )
        chem_valid = probabilities(chem_model, X_valid_emb)
        chem_test = probabilities(chem_model, X_test_emb)

        fixed_alpha, _ = optimize_fixed_weight(y_valid, rf_valid, chem_valid)
        fixed_valid = fixed_fusion(rf_valid, chem_valid, fixed_alpha)
        fixed_test = fixed_fusion(rf_test, chem_test, fixed_alpha)
        a, b, _ = optimize_dynamic_weight(y_valid, rf_valid, chem_valid, validation_similarity)
        dynamic_valid, _ = dynamic_fusion(rf_valid, chem_valid, validation_similarity, a, b)
        dynamic_test, dynamic_alpha = dynamic_fusion(
            rf_test, chem_test, test_similarity, a, b
        )
        fusion_parameter_rows.append(
            {
                "seed": seed,
                "fixed_alpha_rf": fixed_alpha,
                "dynamic_a": a,
                "dynamic_b": b,
                "dynamic_test_alpha_mean": float(dynamic_alpha.mean()),
                "chemberta_best_epoch": int(chem_validation_details["best_epoch"]),
            }
        )

        probability_sets = {
            "A": (rf_valid, rf_test),
            "B": (chem_valid, chem_test),
            "C": (fixed_valid, fixed_test),
            "D": (dynamic_valid, dynamic_test),
        }
        thresholds: dict[str, float] = {}
        for experiment, (valid_probabilities, test_probabilities) in probability_sets.items():
            metrics, threshold = test_metrics_with_validation_threshold(
                y_valid, valid_probabilities, y_test, test_probabilities
            )
            thresholds[experiment] = threshold
            seed_metric_rows.append(
                {
                    "experiment": experiment,
                    "model": EXPERIMENTS[experiment],
                    "split": "test",
                    "seed": seed,
                    **metrics,
                }
            )
            for similarity_bin in ("Q1", "Q2", "Q3", "Q4"):
                mask = np.asarray(test_bins == similarity_bin)
                bin_metric_rows.append(
                    {
                        "experiment": experiment,
                        "model": EXPERIMENTS[experiment],
                        "seed": seed,
                        "similarity_bin": similarity_bin,
                        "n": int(mask.sum()),
                        "n_active": int(y_test[mask].sum()),
                        **safe_bin_metrics(y_test[mask], test_probabilities[mask], threshold),
                    }
                )

        frame = test[["CID", "canonical_smiles", "scaffold", "Class"]].copy()
        frame["seed"] = seed
        frame["max_train_tanimoto"] = test_similarity
        frame["similarity_bin"] = test_bins.astype(str)
        frame["nearest_train_CID"] = train.iloc[nearest_train_indices]["CID"].to_numpy()
        frame["rf_probability"] = rf_test
        frame["chemberta_probability"] = chem_test
        frame["fixed_fusion_probability"] = fixed_test
        frame["dynamic_fusion_probability"] = dynamic_test
        frame["dynamic_alpha_rf"] = dynamic_alpha
        for experiment, threshold in thresholds.items():
            frame[f"threshold_{experiment}"] = threshold
        prediction_frames.append(frame)

    seed_metrics = pd.DataFrame(seed_metric_rows)
    aggregate = aggregate_seed_metrics(seed_metrics)
    bin_metrics = pd.DataFrame(bin_metric_rows)
    parameters = pd.DataFrame(fusion_parameter_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    seed_metrics.to_csv(results_dir / "week5_test_metrics_by_seed.csv", index=False)
    aggregate.to_csv(results_dir / "week5_test_metrics_summary.csv", index=False)
    bin_metrics.to_csv(results_dir / "week5_similarity_bin_metrics.csv", index=False)
    parameters.to_csv(results_dir / "week5_fusion_parameters_by_seed.csv", index=False)
    predictions.to_csv(results_dir / "week5_test_predictions.csv", index=False)

    ensemble = (
        predictions.groupby(["CID", "scaffold", "Class"], as_index=False)
        .agg(
            max_train_tanimoto=("max_train_tanimoto", "first"),
            fixed_probability=("fixed_fusion_probability", "mean"),
            dynamic_probability=("dynamic_fusion_probability", "mean"),
        )
    )
    ensemble["dynamic_brier_advantage_vs_fixed"] = (
        (ensemble["fixed_probability"] - ensemble["Class"]) ** 2
        - (ensemble["dynamic_probability"] - ensemble["Class"]) ** 2
    )
    scaffold_analysis = (
        ensemble.groupby("scaffold", as_index=False)
        .agg(
            n=("CID", "size"),
            active_fraction=("Class", "mean"),
            mean_similarity=("max_train_tanimoto", "mean"),
            dynamic_brier_advantage_vs_fixed=("dynamic_brier_advantage_vs_fixed", "mean"),
        )
        .sort_values("dynamic_brier_advantage_vs_fixed")
    )
    scaffold_analysis.to_csv(results_dir / "week5_scaffold_error_analysis.csv", index=False)

    saved_rf = joblib.load(models_dir / "random_forest_morgan.joblib")
    saved_chem_artifact = torch.load(
        models_dir / "chemberta_mlp.pt", map_location="cpu", weights_only=True
    )
    saved_chem = ActivityMLP(
        saved_chem_artifact["input_dim"],
        saved_chem_artifact["params"]["hidden_dim"],
        saved_chem_artifact["params"]["dropout"],
    )
    saved_chem.load_state_dict(saved_chem_artifact["state_dict"])
    rf_head_seconds, rf_scores_object = median_runtime(
        lambda: saved_rf.predict_proba(X_test_fp)[:, 1], repeats=20
    )
    chem_head_seconds, chem_scores_object = median_runtime(
        lambda: probabilities(saved_chem, X_test_emb), repeats=100
    )
    rf_scores = np.asarray(rf_scores_object)
    chem_scores = np.asarray(chem_scores_object)
    similarity_seconds, similarity_object = median_runtime(
        lambda: maximum_tanimoto_similarity(X_test_fp, X_train_fp)[0], repeats=20
    )
    timed_similarity = np.asarray(similarity_object)
    with (results_dir / "fusion_summary.json").open(encoding="utf-8") as handle:
        week4 = json.load(handle)
    fixed_math_seconds, _ = median_runtime(
        lambda: fixed_fusion(rf_scores, chem_scores, week4["fixed_fusion"]["alpha_rf"]),
        repeats=1000,
    )
    dynamic_math_seconds, _ = median_runtime(
        lambda: dynamic_fusion(
            rf_scores, chem_scores, timed_similarity,
            week4["dynamic_fusion"]["a"], week4["dynamic_fusion"]["b"],
        ),
        repeats=1000,
    )
    total_seconds = {
        "A": rf_featurization_seconds + rf_head_seconds,
        "B": encoder_seconds + chem_head_seconds,
        "C": rf_featurization_seconds + rf_head_seconds + encoder_seconds + chem_head_seconds + fixed_math_seconds,
        "D": rf_featurization_seconds + rf_head_seconds + encoder_seconds + chem_head_seconds + similarity_seconds + dynamic_math_seconds,
    }
    timing = pd.DataFrame(
        [
            {
                "experiment": experiment,
                "model": EXPERIMENTS[experiment],
                "total_seconds_for_227_molecules": seconds,
                "milliseconds_per_molecule": seconds / len(test) * 1000.0,
            }
            for experiment, seconds in total_seconds.items()
        ]
    )
    timing.to_csv(results_dir / "week5_inference_time.csv", index=False)

    summary = {
        "scope": "Week 5 locked-test evaluation after Week 1-4 model selection",
        "test_rows": int(len(test)),
        "seeds": list(SEEDS),
        "test_metrics_mean_std": aggregate.to_dict(orient="records"),
        "fusion_parameters_mean_std": {
            column: {"mean": float(parameters[column].mean()), "std": float(parameters[column].std(ddof=1))}
            for column in ("fixed_alpha_rf", "dynamic_a", "dynamic_b", "dynamic_test_alpha_mean")
        },
        "validation_defined_similarity_edges": [float(value) for value in bin_edges],
        "test_similarity": {
            "minimum": float(test_similarity.min()),
            "mean": float(test_similarity.mean()),
            "median": float(np.median(test_similarity)),
            "maximum": float(test_similarity.max()),
        },
        "inference_time": {
            "definition": "warm loaded models; representation generation and prediction included; sequential CPU pipeline",
            "encoder_device": str(device),
            "components_seconds": {
                "rf_featurization": rf_featurization_seconds,
                "rf_head": rf_head_seconds,
                "chemberta_encoder": encoder_seconds,
                "chemberta_mlp_head": chem_head_seconds,
                "tanimoto_similarity": similarity_seconds,
                "fixed_fusion_math": fixed_math_seconds,
                "dynamic_fusion_math": dynamic_math_seconds,
            },
            "experiments": timing.to_dict(orient="records"),
        },
        "test_encoder": test_encoder_metadata,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
        },
    }
    with (results_dir / "week5_test_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    plot_test_discrimination(aggregate, figures_dir)
    plot_similarity_bins(bin_metrics, figures_dir)
    plot_inference_time(timing, figures_dir)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print(json.dumps(run(arguments.project_root.resolve()), indent=2))
