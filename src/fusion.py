"""Week 4 similarity-aware probability fusion.

The experiment combines the frozen Week 2 and Week 3 validation probabilities.
It computes each validation molecule's maximum Morgan/Tanimoto similarity to
the training set, fits both fixed and sigmoid dynamic fusion on validation, and
keeps the test split locked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from evaluate import classification_metrics


SELECTION_SEED = 42

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


def tanimoto_similarity_matrix(
    query_fingerprints: np.ndarray, reference_fingerprints: np.ndarray
) -> np.ndarray:
    """Return all pairwise Tanimoto similarities for binary fingerprints."""

    query = np.asarray(query_fingerprints, dtype=bool)
    reference = np.asarray(reference_fingerprints, dtype=bool)
    if query.ndim != 2 or reference.ndim != 2:
        raise ValueError("Fingerprint arrays must both be two-dimensional.")
    if query.shape[1] != reference.shape[1]:
        raise ValueError("Query and reference fingerprints must have equal widths.")
    intersection = query.astype(np.int32) @ reference.astype(np.int32).T
    union = query.sum(axis=1, dtype=np.int32)[:, None] + reference.sum(
        axis=1, dtype=np.int32
    )[None, :] - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=np.float64),
        where=union > 0,
    )


def maximum_tanimoto_similarity(
    query_fingerprints: np.ndarray, reference_fingerprints: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    matrix = tanimoto_similarity_matrix(query_fingerprints, reference_fingerprints)
    nearest_indices = np.argmax(matrix, axis=1)
    maximum = matrix[np.arange(len(matrix)), nearest_indices]
    return maximum, nearest_indices


def fixed_fusion(
    rf_probabilities: np.ndarray,
    chemberta_probabilities: np.ndarray,
    alpha: float,
) -> np.ndarray:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("Fixed alpha must lie in [0, 1].")
    return alpha * rf_probabilities + (1.0 - alpha) * chemberta_probabilities


def dynamic_weights(similarity: np.ndarray, a: float, b: float) -> np.ndarray:
    return expit(a * np.asarray(similarity, dtype=float) + b)


def dynamic_fusion(
    rf_probabilities: np.ndarray,
    chemberta_probabilities: np.ndarray,
    similarity: np.ndarray,
    a: float,
    b: float,
) -> tuple[np.ndarray, np.ndarray]:
    alpha = dynamic_weights(similarity, a, b)
    fused = alpha * rf_probabilities + (1.0 - alpha) * chemberta_probabilities
    return fused, alpha


def ranking_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    return {
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
    }


def select_best(search: pd.DataFrame) -> pd.Series:
    return search.sort_values(
        ["pr_auc", "roc_auc", "brier_score"], ascending=[False, False, True]
    ).iloc[0]


def optimize_fixed_weight(
    y_true: np.ndarray,
    rf_probabilities: np.ndarray,
    chemberta_probabilities: np.ndarray,
) -> tuple[float, pd.DataFrame]:
    rows: list[dict[str, float]] = []
    for alpha in np.linspace(0.0, 1.0, 1001):
        scores = fixed_fusion(rf_probabilities, chemberta_probabilities, float(alpha))
        rows.append({"alpha": float(alpha), **ranking_metrics(y_true, scores)})
    search = pd.DataFrame(rows)
    best = select_best(search)
    return float(best["alpha"]), search


def dynamic_search_grid(
    y_true: np.ndarray,
    rf_probabilities: np.ndarray,
    chemberta_probabilities: np.ndarray,
    similarity: np.ndarray,
    a_values: np.ndarray,
    b_values: np.ndarray,
    stage: str,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for a in a_values:
        for b in b_values:
            scores, alpha = dynamic_fusion(
                rf_probabilities, chemberta_probabilities, similarity, float(a), float(b)
            )
            rows.append(
                {
                    "stage": stage,
                    "a": float(a),
                    "b": float(b),
                    "mean_alpha": float(np.mean(alpha)),
                    "min_alpha": float(np.min(alpha)),
                    "max_alpha": float(np.max(alpha)),
                    **ranking_metrics(y_true, scores),
                }
            )
    return pd.DataFrame(rows)


def optimize_dynamic_weight(
    y_true: np.ndarray,
    rf_probabilities: np.ndarray,
    chemberta_probabilities: np.ndarray,
    similarity: np.ndarray,
) -> tuple[float, float, pd.DataFrame]:
    coarse = dynamic_search_grid(
        y_true,
        rf_probabilities,
        chemberta_probabilities,
        similarity,
        np.arange(-32.0, 32.0001, 0.5),
        np.arange(-12.0, 12.0001, 0.5),
        "coarse",
    )
    coarse_best = select_best(coarse)
    fine = dynamic_search_grid(
        y_true,
        rf_probabilities,
        chemberta_probabilities,
        similarity,
        np.arange(float(coarse_best["a"]) - 1.0, float(coarse_best["a"]) + 1.0001, 0.05),
        np.arange(float(coarse_best["b"]) - 1.0, float(coarse_best["b"]) + 1.0001, 0.05),
        "fine",
    )
    search = pd.concat([coarse, fine], ignore_index=True)
    best = select_best(search)
    return float(best["a"]), float(best["b"]), search


def validate_prediction_alignment(
    valid: pd.DataFrame, rf_predictions: pd.DataFrame, chemberta_predictions: pd.DataFrame
) -> None:
    expected = valid["CID"].astype(str).to_numpy()
    for name, frame in (("RF", rf_predictions), ("ChemBERTa", chemberta_predictions)):
        observed = frame["CID"].astype(str).to_numpy()
        if not np.array_equal(expected, observed):
            raise ValueError(f"{name} validation predictions are not aligned with valid.csv.")
        if not np.array_equal(valid["Class"].to_numpy(), frame["Class"].to_numpy()):
            raise ValueError(f"{name} validation labels disagree with valid.csv.")


def similarity_bin_analysis(
    y_true: np.ndarray,
    rf_probabilities: np.ndarray,
    chemberta_probabilities: np.ndarray,
    similarity: np.ndarray,
) -> tuple[pd.DataFrame, float, float]:
    rf_advantage = (chemberta_probabilities - y_true) ** 2 - (
        rf_probabilities - y_true
    ) ** 2
    correlation = spearmanr(similarity, rf_advantage)
    frame = pd.DataFrame(
        {"similarity": similarity, "rf_brier_advantage": rf_advantage}
    )
    frame["similarity_quartile"] = pd.qcut(
        frame["similarity"], q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop"
    )
    grouped = (
        frame.groupby("similarity_quartile", observed=True)
        .agg(
            n=("rf_brier_advantage", "size"),
            mean_similarity=("similarity", "mean"),
            min_similarity=("similarity", "min"),
            max_similarity=("similarity", "max"),
            mean_rf_brier_advantage=("rf_brier_advantage", "mean"),
            sd_rf_brier_advantage=("rf_brier_advantage", "std"),
        )
        .reset_index()
    )
    grouped["se_rf_brier_advantage"] = grouped["sd_rf_brier_advantage"] / np.sqrt(grouped["n"])
    return grouped, float(correlation.statistic), float(correlation.pvalue)


def save_figure(fig: plt.Figure, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(metrics: pd.DataFrame, output_dir: Path) -> None:
    display = {
        "Morgan RF": "Morgan RF",
        "Frozen ChemBERTa MLP": "ChemBERTa MLP",
        "Fixed-weight fusion": "Fixed fusion",
        "Dynamic similarity fusion": "Dynamic fusion",
    }
    ordered = list(display)
    subset = metrics.set_index("model").loc[ordered]
    y = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(3.50, 3.05), constrained_layout=True)
    ax.scatter(subset["roc_auc"], y - 0.10, color="#4C78A8", s=28, label="ROC-AUC")
    ax.scatter(subset["pr_auc"], y + 0.10, color="#E07B54", marker="s", s=25, label="PR-AUC")
    ax.set_yticks(y, [display[name] for name in ordered])
    lower = max(0.0, float(subset[["roc_auc", "pr_auc"]].min().min() - 0.04))
    ax.set_xlim(lower, 1.0)
    ax.set_xlabel("Scaffold-validation score")
    ax.set_title("Fusion compared with both component models")
    ax.invert_yaxis()
    ax.legend(loc="upper left")
    save_figure(fig, output_dir / "fusion_validation_model_comparison")


def plot_weight_function(
    similarity: np.ndarray, a: float, b: float, output_dir: Path
) -> None:
    grid = np.linspace(0.0, 1.0, 501)
    fig, ax = plt.subplots(figsize=(3.50, 2.90), constrained_layout=True)
    ax.plot(grid, dynamic_weights(grid, a, b), color="#6F4E7C", linewidth=1.8)
    ax.plot(similarity, np.full_like(similarity, 0.015), "|", color="#A9A9A9", alpha=0.45,
            markersize=5, markeredgewidth=0.6, label="Validation similarities")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Maximum train-set Tanimoto similarity",
           ylabel="Dynamic RF weight, α(x)")
    ax.set_title("Learned similarity-to-weight function")
    ax.legend(loc="best")
    save_figure(fig, output_dir / "fusion_dynamic_weight_function")


def plot_similarity_hypothesis(
    binned: pd.DataFrame, rho: float, output_dir: Path
) -> None:
    fig, ax = plt.subplots(figsize=(3.50, 2.90), constrained_layout=True)
    ax.axhline(0.0, color="#9AA0A6", linestyle="--", linewidth=1.0)
    ax.errorbar(
        binned["mean_similarity"],
        binned["mean_rf_brier_advantage"],
        yerr=binned["se_rf_brier_advantage"],
        color="#4C78A8",
        marker="o",
        linewidth=1.4,
        capsize=3,
    )
    ax.set(xlabel="Mean maximum Tanimoto similarity",
           ylabel="RF Brier advantage (mean ± SE)")
    ax.set_title(f"Similarity hypothesis diagnostic (Spearman ρ={rho:.2f})")
    save_figure(fig, output_dir / "fusion_similarity_hypothesis")


def run(project_root: Path) -> dict[str, object]:
    train_path = project_root / "data/processed/train.csv"
    valid_path = project_root / "data/processed/valid.csv"
    train_fp_path = project_root / "data/processed/features/morgan_r2_2048_train.npz"
    valid_fp_path = project_root / "data/processed/features/morgan_r2_2048_valid.npz"
    rf_path = project_root / "results/rf_validation_predictions.csv"
    chemberta_path = project_root / "results/chemberta_validation_predictions.csv"
    results_dir = project_root / "results"
    figures_dir = results_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(train_path)
    valid = pd.read_csv(valid_path)
    rf_predictions = pd.read_csv(rf_path)
    chemberta_predictions = pd.read_csv(chemberta_path)
    validate_prediction_alignment(valid, rf_predictions, chemberta_predictions)
    if not set(train["scaffold"].fillna("")).isdisjoint(set(valid["scaffold"].fillna(""))):
        raise AssertionError("Train/validation scaffold leakage detected.")

    with np.load(train_fp_path) as cache:
        train_fingerprints = cache["X"]
        if not np.array_equal(cache["smiles"], train["canonical_smiles"].astype(str).to_numpy(dtype=str)):
            raise ValueError("Train fingerprint cache is not aligned with train.csv.")
    with np.load(valid_fp_path) as cache:
        valid_fingerprints = cache["X"]
        if not np.array_equal(cache["smiles"], valid["canonical_smiles"].astype(str).to_numpy(dtype=str)):
            raise ValueError("Validation fingerprint cache is not aligned with valid.csv.")

    similarity, nearest_indices = maximum_tanimoto_similarity(
        valid_fingerprints, train_fingerprints
    )
    y_valid = valid["Class"].to_numpy(dtype=np.int8)
    rf_probabilities = rf_predictions["probability_active"].to_numpy(dtype=float)
    chemberta_probabilities = chemberta_predictions["probability_active"].to_numpy(dtype=float)

    fixed_alpha, fixed_search = optimize_fixed_weight(
        y_valid, rf_probabilities, chemberta_probabilities
    )
    fixed_probabilities = fixed_fusion(
        rf_probabilities, chemberta_probabilities, fixed_alpha
    )
    a, b, dynamic_search = optimize_dynamic_weight(
        y_valid, rf_probabilities, chemberta_probabilities, similarity
    )
    dynamic_probabilities, alpha = dynamic_fusion(
        rf_probabilities, chemberta_probabilities, similarity, a, b
    )

    fixed_search.to_csv(results_dir / "fixed_weight_fusion_search.csv", index=False)
    dynamic_search.to_csv(results_dir / "dynamic_fusion_parameter_search.csv", index=False)

    model_probabilities = {
        "Morgan RF": rf_probabilities,
        "Frozen ChemBERTa MLP": chemberta_probabilities,
        "Fixed-weight fusion": fixed_probabilities,
        "Dynamic similarity fusion": dynamic_probabilities,
    }
    metric_rows = [
        {
            "model": model,
            "split": "valid",
            "seed": SELECTION_SEED,
            **classification_metrics(y_valid, probabilities),
        }
        for model, probabilities in model_probabilities.items()
    ]
    comparison = pd.DataFrame(metric_rows)
    comparison.to_csv(results_dir / "fusion_validation_metrics.csv", index=False)

    binned, rho, rho_pvalue = similarity_bin_analysis(
        y_valid, rf_probabilities, chemberta_probabilities, similarity
    )
    binned.to_csv(results_dir / "similarity_bin_analysis.csv", index=False)

    predictions = valid[["CID", "canonical_smiles", "scaffold", "Class"]].copy()
    predictions["max_train_tanimoto"] = similarity
    predictions["nearest_train_CID"] = train.iloc[nearest_indices]["CID"].to_numpy()
    predictions["rf_probability"] = rf_probabilities
    predictions["chemberta_probability"] = chemberta_probabilities
    predictions["fixed_alpha_rf"] = fixed_alpha
    predictions["fixed_fusion_probability"] = fixed_probabilities
    predictions["dynamic_alpha_rf"] = alpha
    predictions["dynamic_fusion_probability"] = dynamic_probabilities
    predictions.to_csv(results_dir / "fusion_validation_predictions.csv", index=False)

    metrics_path = results_dir / "metrics.csv"
    all_metrics = pd.read_csv(metrics_path)
    all_metrics = all_metrics[
        ~all_metrics["model"].isin(["Fixed-weight fusion", "Dynamic similarity fusion"])
    ]
    pd.concat(
        [
            all_metrics,
            comparison[comparison["model"].isin(["Fixed-weight fusion", "Dynamic similarity fusion"])],
        ],
        ignore_index=True,
    ).to_csv(metrics_path, index=False)

    dynamic_best = classification_metrics(y_valid, dynamic_probabilities)
    fixed_best = classification_metrics(y_valid, fixed_probabilities)
    summary = {
        "scope": "Week 4 validation optimization only; test.csv was not read",
        "selection_rule": "maximize validation PR-AUC, then ROC-AUC, then minimize Brier score",
        "hypothesis": "Higher train-set similarity may favor Morgan RF over frozen ChemBERTa",
        "fixed_fusion": {"alpha_rf": fixed_alpha, "metrics": fixed_best},
        "dynamic_fusion": {
            "a": a,
            "b": b,
            "alpha_min": float(alpha.min()),
            "alpha_mean": float(alpha.mean()),
            "alpha_max": float(alpha.max()),
            "metrics": dynamic_best,
        },
        "component_metrics": {
            model: classification_metrics(y_valid, probabilities)
            for model, probabilities in model_probabilities.items()
            if model in ("Morgan RF", "Frozen ChemBERTa MLP")
        },
        "similarity": {
            "minimum": float(similarity.min()),
            "mean": float(similarity.mean()),
            "median": float(np.median(similarity)),
            "maximum": float(similarity.max()),
            "spearman_similarity_vs_rf_brier_advantage": rho,
            "spearman_pvalue": rho_pvalue,
        },
        "hypothesis_direction_supported": bool(a > 0 and rho > 0),
        "train_rows": int(len(train)),
        "valid_rows": int(len(valid)),
        "input_sha256": {
            "train_csv": file_sha256(train_path),
            "valid_csv": file_sha256(valid_path),
            "rf_predictions": file_sha256(rf_path),
            "chemberta_predictions": file_sha256(chemberta_path),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    with (results_dir / "fusion_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    plot_model_comparison(comparison, figures_dir)
    plot_weight_function(similarity, a, b, figures_dir)
    plot_similarity_hypothesis(binned, rho, figures_dir)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    output = run(arguments.project_root.resolve())
    print(json.dumps(output, indent=2))
