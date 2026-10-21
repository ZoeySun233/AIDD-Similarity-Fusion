"""Week 2 Morgan-fingerprint Random Forest baseline.

Only the frozen train and validation scaffold splits are read. The test split
remains untouched until the final comparison stage.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import sys
import time
from pathlib import Path

import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rdkit
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, roc_curve

from evaluate import classification_metrics
from fingerprints import MorganConfig, morgan_fingerprints, save_fingerprint_cache


SELECTION_SEED = 42
ROBUSTNESS_SEEDS = (11, 22, 33, 44, 55)

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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parameter_grid() -> list[dict[str, object]]:
    keys = ("max_depth", "min_samples_leaf", "max_features")
    values = ([None, 12, 24], [1, 3, 5], ["sqrt", 0.25])
    return [dict(zip(keys, combination)) for combination in itertools.product(*values)]


def build_model(params: dict[str, object], seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=500,
        criterion="gini",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
        **params,
    )


def save_figure(fig: plt.Figure, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_roc(y_true: np.ndarray, probabilities: np.ndarray, output_dir: Path) -> None:
    metrics = classification_metrics(y_true, probabilities)
    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, probabilities)
    fig, ax = plt.subplots(figsize=(3.50, 2.90), constrained_layout=True)
    ax.plot(
        false_positive_rate,
        true_positive_rate,
        color="#4C78A8",
        linewidth=1.8,
        label=f"Random Forest (AUC={metrics['roc_auc']:.3f})",
    )
    ax.plot([0, 1], [0, 1], color="#9AA0A6", linestyle="--", linewidth=1.0, label="Chance")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="False-positive rate", ylabel="True-positive rate")
    ax.set_title("Scaffold-validation ROC curve")
    ax.legend(loc="lower right")
    save_figure(fig, output_dir / "rf_validation_roc")


def plot_precision_recall(
    y_true: np.ndarray, probabilities: np.ndarray, output_dir: Path
) -> None:
    metrics = classification_metrics(y_true, probabilities)
    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    prevalence = float(np.mean(y_true))
    fig, ax = plt.subplots(figsize=(3.50, 2.90), constrained_layout=True)
    ax.plot(
        recall,
        precision,
        color="#4C78A8",
        linewidth=1.8,
        label=f"Random Forest (AP={metrics['pr_auc']:.3f})",
    )
    ax.axhline(
        prevalence,
        color="#9AA0A6",
        linestyle="--",
        linewidth=1.0,
        label=f"Prevalence ({prevalence:.3f})",
    )
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Recall", ylabel="Precision")
    ax.set_title("Validation precision–recall curve")
    ax.legend(loc="lower left")
    save_figure(fig, output_dir / "rf_validation_precision_recall")


def plot_seed_robustness(seed_metrics: pd.DataFrame, output_dir: Path) -> None:
    metric_columns = ["roc_auc", "pr_auc"]
    display_names = ["ROC-AUC", "PR-AUC"]
    means = seed_metrics[metric_columns].mean().to_numpy()
    standard_deviations = seed_metrics[metric_columns].std(ddof=1).to_numpy()
    x = np.arange(len(metric_columns))
    fig, ax = plt.subplots(figsize=(3.50, 2.90), constrained_layout=True)
    ax.errorbar(
        x,
        means,
        yerr=standard_deviations,
        fmt="o",
        markersize=6,
        color="#4C78A8",
        ecolor="#4C78A8",
        elinewidth=1.2,
        capsize=4,
    )
    ax.set_xticks(x, display_names)
    ax.set_ylabel("Validation score")
    ax.set_ylim(0.93, 0.97)
    ax.set_title("Random Forest is stable across seeds")
    ax.text(
        0.98,
        0.05,
        "Mean ± one SD; n=5 seeds",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color="#4A4A4A",
    )
    save_figure(fig, output_dir / "rf_validation_seed_stability")


def run(project_root: Path) -> dict[str, object]:
    train_path = project_root / "data/processed/train.csv"
    valid_path = project_root / "data/processed/valid.csv"
    results_dir = project_root / "results"
    figures_dir = results_dir / "figures"
    models_dir = project_root / "models"
    features_dir = project_root / "data/processed/features"
    for directory in (results_dir, figures_dir, models_dir, features_dir):
        directory.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(train_path)
    valid = pd.read_csv(valid_path)
    required = {"canonical_smiles", "Class", "scaffold"}
    for name, frame in (("train", train), ("valid", valid)):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")
    train_scaffolds = set(train["scaffold"].fillna(""))
    valid_scaffolds = set(valid["scaffold"].fillna(""))
    if not train_scaffolds.isdisjoint(valid_scaffolds):
        raise AssertionError("Train/validation scaffold leakage detected.")

    config = MorganConfig(radius=2, n_bits=2048, include_chirality=True)
    X_train = morgan_fingerprints(train["canonical_smiles"], config)
    X_valid = morgan_fingerprints(valid["canonical_smiles"], config)
    y_train = train["Class"].to_numpy(dtype=np.int8)
    y_valid = valid["Class"].to_numpy(dtype=np.int8)
    save_fingerprint_cache(
        features_dir / "morgan_r2_2048_train.npz",
        X_train,
        y_train,
        train["canonical_smiles"],
        config,
        "train",
    )
    save_fingerprint_cache(
        features_dir / "morgan_r2_2048_valid.npz",
        X_valid,
        y_valid,
        valid["canonical_smiles"],
        config,
        "valid",
    )

    search_rows: list[dict[str, object]] = []
    for params in parameter_grid():
        started = time.perf_counter()
        model = build_model(params, SELECTION_SEED)
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_valid)[:, 1]
        metrics = classification_metrics(y_valid, probabilities)
        search_rows.append(
            {
                **params,
                **metrics,
                "fit_seconds": float(time.perf_counter() - started),
            }
        )
    search = pd.DataFrame(search_rows).sort_values(
        ["pr_auc", "roc_auc", "brier_score"], ascending=[False, False, True]
    )
    search.insert(0, "rank", np.arange(1, len(search) + 1))
    search.to_csv(results_dir / "rf_hyperparameter_search.csv", index=False)
    best_params = {
        key: search.iloc[0][key] for key in ("max_depth", "min_samples_leaf", "max_features")
    }
    if pd.isna(best_params["max_depth"]):
        best_params["max_depth"] = None
    else:
        best_params["max_depth"] = int(best_params["max_depth"])
    best_params["min_samples_leaf"] = int(best_params["min_samples_leaf"])
    if isinstance(best_params["max_features"], np.floating):
        best_params["max_features"] = float(best_params["max_features"])

    final_model = build_model(best_params, SELECTION_SEED)
    final_model.fit(X_train, y_train)
    valid_probabilities = final_model.predict_proba(X_valid)[:, 1]
    canonical_metrics = classification_metrics(y_valid, valid_probabilities)

    metric_rows: list[dict[str, object]] = [
        {
            "model": "Prevalence baseline",
            "split": "valid",
            "seed": "deterministic",
            **classification_metrics(
                y_valid, np.full(len(y_valid), float(np.mean(y_train))), threshold=0.5
            ),
        },
        {
            "model": "Morgan RF",
            "split": "valid",
            "seed": SELECTION_SEED,
            **canonical_metrics,
        },
    ]
    robustness_rows: list[dict[str, object]] = []
    for seed in ROBUSTNESS_SEEDS:
        model = build_model(best_params, seed)
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_valid)[:, 1]
        row = {
            "model": "Morgan RF",
            "split": "valid",
            "seed": seed,
            **classification_metrics(y_valid, probabilities),
        }
        robustness_rows.append(row)
        metric_rows.append(row)
    pd.DataFrame(metric_rows).to_csv(results_dir / "metrics.csv", index=False)

    predictions = valid[["CID", "canonical_smiles", "scaffold", "Class"]].copy()
    predictions["probability_active"] = valid_probabilities
    predictions.to_csv(results_dir / "rf_validation_predictions.csv", index=False)

    model_path = models_dir / "random_forest_morgan.joblib"
    joblib.dump(final_model, model_path)
    robustness = pd.DataFrame(robustness_rows)
    summary = {
        "scope": "Week 2 train/validation only; test.csv was not read",
        "selection_rule": "maximize validation PR-AUC, then ROC-AUC, then minimize Brier score",
        "selection_seed": SELECTION_SEED,
        "robustness_seeds": list(ROBUSTNESS_SEEDS),
        "best_params": best_params,
        "fingerprint": {
            "radius": config.radius,
            "n_bits": config.n_bits,
            "include_chirality": config.include_chirality,
        },
        "train_rows": int(len(train)),
        "valid_rows": int(len(valid)),
        "train_csv_sha256": file_sha256(train_path),
        "valid_csv_sha256": file_sha256(valid_path),
        "canonical_validation_metrics": canonical_metrics,
        "five_seed_validation_mean": {
            key: float(robustness[key].mean())
            for key in ("roc_auc", "pr_auc", "brier_score", "f1", "balanced_accuracy", "mcc")
        },
        "five_seed_validation_std": {
            key: float(robustness[key].std(ddof=1))
            for key in ("roc_auc", "pr_auc", "brier_score", "f1", "balanced_accuracy", "mcc")
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "rdkit": rdkit.__version__,
        },
        "model_artifact": str(model_path.relative_to(project_root)),
    }
    with (results_dir / "rf_baseline_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    plot_roc(y_valid, valid_probabilities, figures_dir)
    plot_precision_recall(y_valid, valid_probabilities, figures_dir)
    plot_seed_robustness(robustness, figures_dir)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    output = run(arguments.project_root.resolve())
    print(json.dumps(output, indent=2))
