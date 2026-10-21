"""Nested-CV fusion ablations using only Phase 4 Platt-calibrated inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from scipy.optimize import minimize, minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
METHODS = ["C0_equal_weight", "C1_fixed_weight", "D0_original_similarity_gate", "D1_shrunk_similarity", "D2_similarity_entropy", "D3_similarity_entropy_topk", "D4_similarity_entropy_topk_variance", "E0_logistic_stacking", "N0_permuted_similarity", "N1_random_matched_gate", "N2_confidence_only"]
SIMILARITY_CANDIDATES = {
    "D1_shrunk_similarity": ["max_similarity"],
    "D2_similarity_entropy": ["max_similarity", "entropy_difference"],
    "D3_similarity_entropy_topk": ["max_similarity", "entropy_difference", "topk_mean_similarity", "topk_active_fraction"],
    "D4_similarity_entropy_topk_variance": ["max_similarity", "entropy_difference", "topk_mean_similarity", "topk_active_fraction", "variance_difference"],
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sigmoid(value: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -35, 35)))


def entropy(probability: np.ndarray) -> np.ndarray:
    value = np.clip(probability, 1e-8, 1 - 1e-8)
    return -(value * np.log(value) + (1 - value) * np.log(1 - value))


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {"brier_score": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))), "pr_auc": float(average_precision_score(y, p)), "roc_auc": float(roc_auc_score(y, p))}


def standardize_fit(train: pd.DataFrame, target: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, np.ndarray, dict[str, list[float]]]:
    mean = train[columns].mean().to_numpy(float, copy=True)
    scale = train[columns].std(ddof=0).to_numpy(float, copy=True)
    scale[scale < 1e-8] = 1.0
    return (train[columns].to_numpy(float) - mean) / scale, (target[columns].to_numpy(float) - mean) / scale, {"mean": mean.tolist(), "scale": scale.tolist()}


def fixed_weight(y: np.ndarray, rf: np.ndarray, chem: np.ndarray) -> float:
    result = minimize_scalar(lambda alpha: brier_score_loss(y, alpha * rf + (1 - alpha) * chem), bounds=(0, 1), method="bounded")
    return float(result.x)


def fit_shrunk_gate(y: np.ndarray, rf: np.ndarray, chem: np.ndarray, X: np.ndarray, fixed_alpha: float, beta_l2: float, lambda_l2: float, maxiter: int) -> np.ndarray:
    def objective(theta: np.ndarray) -> float:
        dynamic = sigmoid(X @ theta[:-1])
        lam = float(sigmoid(theta[-1]))
        alpha = (1 - lam) * fixed_alpha + lam * dynamic
        probability = alpha * rf + (1 - alpha) * chem
        return float(brier_score_loss(y, probability) + beta_l2 * np.square(theta[:-1]).sum() + lambda_l2 * lam * lam)
    initial = np.zeros(X.shape[1] + 1)
    initial[-1] = -2.0  # regularized preference for the fixed-fusion nesting point
    result = minimize(objective, initial, method="L-BFGS-B", options={"maxiter": maxiter})
    if not result.success:
        raise RuntimeError(f"Gate optimization failed: {result.message}")
    return result.x


def shrunk_predict(rf: np.ndarray, chem: np.ndarray, X: np.ndarray, fixed_alpha: float, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dynamic = sigmoid(X @ theta[:-1])
    lam = float(sigmoid(theta[-1]))
    alpha = (1 - lam) * fixed_alpha + lam * dynamic
    return alpha * rf + (1 - alpha) * chem, alpha


def fit_original_gate(y: np.ndarray, rf: np.ndarray, chem: np.ndarray, X: np.ndarray, maxiter: int) -> np.ndarray:
    def objective(beta: np.ndarray) -> float:
        alpha = sigmoid(X @ beta)
        return float(brier_score_loss(y, alpha * rf + (1 - alpha) * chem))
    result = minimize(objective, np.zeros(X.shape[1]), method="L-BFGS-B", options={"maxiter": maxiter})
    if not result.success:
        raise RuntimeError(f"Original gate optimization failed: {result.message}")
    return result.x


def similarity_features(query_positions: np.ndarray, train_positions: np.ndarray, fingerprints: list[Any], train_labels: np.ndarray, top_k: int) -> np.ndarray:
    rows = []
    for position in query_positions:
        similarity = np.asarray(DataStructs.BulkTanimotoSimilarity(fingerprints[position], [fingerprints[index] for index in train_positions]), dtype=float)
        rank = np.argsort(similarity)[-min(top_k, len(similarity)):]
        rows.append((float(similarity.max()), float(similarity[rank].mean()), float(train_labels[rank].mean())))
    return np.asarray(rows, dtype=float)


def paired_context_predictions(predictions: pd.DataFrame, repeat: int, outer_fold: int, seed: int) -> pd.DataFrame:
    subset = predictions[(predictions.repeat == repeat) & (predictions.outer_fold == outer_fold) & (predictions.seed == seed)]
    rf = subset[subset.model == "morgan_rf"][["source_row", "CID", "split", "label", "probability_platt"]].rename(columns={"probability_platt": "rf_probability"})
    chem = subset[subset.model == "frozen_chemberta_mlp"][["source_row", "probability_platt"]].rename(columns={"probability_platt": "chem_probability"})
    return rf.merge(chem, on="source_row", validate="one_to_one")


def add_seed_variance(predictions: pd.DataFrame) -> pd.DataFrame:
    variance = predictions.groupby(["repeat", "outer_fold", "split", "source_row", "model"], as_index=False).probability_platt.var(ddof=0).rename(columns={"probability_platt": "probability_variance"})
    wide = variance.pivot(index=["repeat", "outer_fold", "split", "source_row"], columns="model", values="probability_variance").reset_index().rename(columns={"morgan_rf": "rf_variance", "frozen_chemberta_mlp": "chem_variance"})
    wide["variance_difference"] = wide.chem_variance - wide.rf_variance
    return wide


def crossfit_losses(dev: pd.DataFrame, feature_columns: list[str], config: dict[str, Any]) -> tuple[list[float], list[dict[str, Any]]]:
    """Gate-selection losses with each inner fold held out from gate fitting."""
    losses, details = [], []
    opt = config["optimization"]
    for inner_fold in (1, 2, 3):
        train, valid = dev[dev.inner_fold != inner_fold], dev[dev.inner_fold == inner_fold]
        alpha_fixed = fixed_weight(train.label.to_numpy(), train.rf_probability.to_numpy(), train.chem_probability.to_numpy())
        X_train, X_valid, scale = standardize_fit(train, valid, feature_columns)
        X_train, X_valid = np.column_stack([np.ones(len(train)), X_train]), np.column_stack([np.ones(len(valid)), X_valid])
        theta = fit_shrunk_gate(train.label.to_numpy(), train.rf_probability.to_numpy(), train.chem_probability.to_numpy(), X_train, alpha_fixed, float(opt["beta_l2"]), float(opt["lambda_l2"]), int(opt["optimizer_maxiter"]))
        probability, alpha = shrunk_predict(valid.rf_probability.to_numpy(), valid.chem_probability.to_numpy(), X_valid, alpha_fixed, theta)
        losses.append(float(brier_score_loss(valid.label.to_numpy(), probability)))
        details.append({"inner_fold": inner_fold, "fixed_alpha": alpha_fixed, "lambda": float(sigmoid(theta[-1])), "beta": theta[:-1].tolist(), "standardization": scale})
    return losses, details


def select_gate(dev: pd.DataFrame, config: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    records = []
    for complexity, (method, columns) in enumerate(SIMILARITY_CANDIDATES.items(), start=1):
        losses, _ = crossfit_losses(dev, columns, config)
        records.append({"candidate": method, "features": json.dumps(columns), "complexity_rank": complexity, "mean_brier": float(np.mean(losses)), "standard_error_brier": float(np.std(losses, ddof=1) / np.sqrt(len(losses))), "fold_brier": json.dumps(losses)})
    best = min(records, key=lambda row: row["mean_brier"])
    threshold = best["mean_brier"] + best["standard_error_brier"]
    eligible = [row for row in records if row["mean_brier"] <= threshold]
    chosen = min(eligible, key=lambda row: row["complexity_rank"])["candidate"]
    for row in records:
        row["one_se_threshold"] = threshold
        row["selected_by_one_se"] = row["candidate"] == chosen
    return chosen, records


def fit_full_candidate(train: pd.DataFrame, test: pd.DataFrame, method: str, config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    columns = SIMILARITY_CANDIDATES[method]
    alpha_fixed = fixed_weight(train.label.to_numpy(), train.rf_probability.to_numpy(), train.chem_probability.to_numpy())
    X_train, X_test, scale = standardize_fit(train, test, columns)
    X_train, X_test = np.column_stack([np.ones(len(train)), X_train]), np.column_stack([np.ones(len(test)), X_test])
    opt = config["optimization"]
    theta = fit_shrunk_gate(train.label.to_numpy(), train.rf_probability.to_numpy(), train.chem_probability.to_numpy(), X_train, alpha_fixed, float(opt["beta_l2"]), float(opt["lambda_l2"]), int(opt["optimizer_maxiter"]))
    probability, alpha = shrunk_predict(test.rf_probability.to_numpy(), test.chem_probability.to_numpy(), X_test, alpha_fixed, theta)
    return probability, alpha, {"features": columns, "fixed_alpha": alpha_fixed, "lambda": float(sigmoid(theta[-1])), "beta": theta[:-1].tolist(), "standardization": scale}


def record_prediction_rows(sink: list[dict[str, Any]], frame: pd.DataFrame, probability: np.ndarray, alpha: np.ndarray, method: str, repeat: int, outer_fold: int, seed: int, selected: bool) -> None:
    for (_, row), value, weight in zip(frame.iterrows(), probability, alpha):
        sink.append({"CID": row.CID, "source_row": int(row.source_row), "repeat": repeat, "outer_fold": outer_fold, "seed": seed, "method": method, "split": "outer_test", "label": int(row.label), "probability": float(value), "alpha_rf": float(weight), "selected_dynamic_gate": selected})


def run(config_path: Path) -> dict[str, Path]:
    config = yaml.safe_load(config_path.read_text())
    inputs, outputs = config["inputs"], config["outputs"]
    data_path, pred_path = ROOT / inputs["standardized_csv"], ROOT / inputs["calibrated_predictions_csv"]
    data = pd.read_csv(data_path, keep_default_na=False).sort_values("source_row").reset_index(drop=True)
    predictions = pd.read_csv(pred_path)
    if set(predictions.model) != {"morgan_rf", "frozen_chemberta_mlp"} or set(predictions.split) != {"oof_development", "outer_test"}:
        raise ValueError("Phase 4 calibrated prediction contract is incomplete.")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=int(config["similarity"]["morgan_radius"]), fpSize=int(config["similarity"]["fingerprint_size"]), includeChirality=True)
    fingerprints = [generator.GetFingerprint(Chem.MolFromSmiles(smiles)) for smiles in data.standardized_smiles]
    position = {int(source): index for index, source in enumerate(data.source_row)}
    variance = add_seed_variance(predictions)
    output_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    feature_audit: list[dict[str, Any]] = []
    seeds = sorted(predictions.seed.unique())
    for repeat in (1, 2):
        outer = pd.read_csv(ROOT / inputs["split_directory"] / f"outer_repeat_{repeat}.csv")[["source_row", "outer_fold"]]
        for outer_fold in range(1, 6):
            inner = pd.read_csv(ROOT / inputs["split_directory"] / f"inner_repeat_{repeat}_outer_fold_{outer_fold}.csv")[["source_row", "inner_fold"]]
            outer_test_sources = outer.loc[outer.outer_fold == outer_fold, "source_row"].to_numpy(int)
            dev_sources = outer.loc[outer.outer_fold != outer_fold, "source_row"].to_numpy(int)
            dev_labels = data.set_index("source_row").loc[dev_sources, "Class"].to_numpy(int)
            # Features are computed once per outer context using only the appropriate training subset.
            dev_feature_parts = []
            for inner_fold in (1, 2, 3):
                query_sources = inner.loc[inner.inner_fold == inner_fold, "source_row"].to_numpy(int)
                train_sources = inner.loc[inner.inner_fold != inner_fold, "source_row"].to_numpy(int)
                query_positions = np.asarray([position[value] for value in query_sources])
                train_positions = np.asarray([position[value] for value in train_sources])
                train_labels = data.set_index("source_row").loc[train_sources, "Class"].to_numpy(int)
                values = similarity_features(query_positions, train_positions, fingerprints, train_labels, int(config["similarity"]["top_k"]))
                dev_feature_parts.append(pd.DataFrame({"source_row": query_sources, "max_similarity": values[:, 0], "topk_mean_similarity": values[:, 1], "topk_active_fraction": values[:, 2]}))
            dev_features = pd.concat(dev_feature_parts, ignore_index=True)
            test_positions, train_positions = np.asarray([position[value] for value in outer_test_sources]), np.asarray([position[value] for value in dev_sources])
            test_values = similarity_features(test_positions, train_positions, fingerprints, dev_labels, int(config["similarity"]["top_k"]))
            test_features = pd.DataFrame({"source_row": outer_test_sources, "max_similarity": test_values[:, 0], "topk_mean_similarity": test_values[:, 1], "topk_active_fraction": test_values[:, 2]})
            for seed in seeds:
                paired = paired_context_predictions(predictions, repeat, outer_fold, int(seed))
                dev = paired[paired.split == "oof_development"].merge(inner, on="source_row", validate="one_to_one").merge(dev_features, on="source_row", validate="one_to_one")
                test = paired[paired.split == "outer_test"].merge(test_features, on="source_row", validate="one_to_one")
                dev = dev.merge(variance[(variance.repeat == repeat) & (variance.outer_fold == outer_fold) & (variance.split == "oof_development")][["source_row", "variance_difference"]], on="source_row", validate="one_to_one")
                test = test.merge(variance[(variance.repeat == repeat) & (variance.outer_fold == outer_fold) & (variance.split == "outer_test")][["source_row", "variance_difference"]], on="source_row", validate="one_to_one")
                for frame in (dev, test):
                    frame["entropy_difference"] = entropy(frame.chem_probability.to_numpy()) - entropy(frame.rf_probability.to_numpy())
                chosen, selection = select_gate(dev, config)
                for record in selection:
                    record.update({"repeat": repeat, "outer_fold": outer_fold, "seed": int(seed), "selection_population": "cross-fitted inner OOF predictions"})
                    selection_rows.append(record)
                fixed_alpha = fixed_weight(dev.label.to_numpy(), dev.rf_probability.to_numpy(), dev.chem_probability.to_numpy())
                y_test, rf_test, chem_test = test.label.to_numpy(), test.rf_probability.to_numpy(), test.chem_probability.to_numpy()
                # Essential controls C0 / C1.
                record_prediction_rows(output_rows, test, 0.5 * rf_test + 0.5 * chem_test, np.full(len(test), 0.5), "C0_equal_weight", repeat, outer_fold, int(seed), False)
                record_prediction_rows(output_rows, test, fixed_alpha * rf_test + (1 - fixed_alpha) * chem_test, np.full(len(test), fixed_alpha), "C1_fixed_weight", repeat, outer_fold, int(seed), False)
                # D0 is deliberately the unshrunk original maximum-similarity sigmoid gate.
                x_train, x_test, _ = standardize_fit(dev, test, ["max_similarity"])
                beta0 = fit_original_gate(dev.label.to_numpy(), dev.rf_probability.to_numpy(), dev.chem_probability.to_numpy(), np.column_stack([np.ones(len(dev)), x_train]), int(config["optimization"]["optimizer_maxiter"]))
                alpha0 = sigmoid(np.column_stack([np.ones(len(test)), x_test]) @ beta0)
                record_prediction_rows(output_rows, test, alpha0 * rf_test + (1 - alpha0) * chem_test, alpha0, "D0_original_similarity_gate", repeat, outer_fold, int(seed), False)
                # Full selected gate and all prespecified ablations are reported, never selected on outer test.
                for method in SIMILARITY_CANDIDATES:
                    probability, alpha, details = fit_full_candidate(dev, test, method, config)
                    record_prediction_rows(output_rows, test, probability, alpha, method, repeat, outer_fold, int(seed), method == chosen)
                    feature_audit.append({"repeat": repeat, "outer_fold": outer_fold, "seed": int(seed), "method": method, **{key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in details.items()}})
                # E0: a non-gating calibrated logistic stacking control.
                stacker = LogisticRegression(C=1.0, solver="lbfgs").fit(dev[["rf_probability", "chem_probability"]], dev.label)
                stack_prob = stacker.predict_proba(test[["rf_probability", "chem_probability"]])[:, 1]
                record_prediction_rows(output_rows, test, stack_prob, np.full(len(test), np.nan), "E0_logistic_stacking", repeat, outer_fold, int(seed), False)
                # Negative controls: a deterministic similarity permutation, distribution-matched random weights, confidence only.
                rng = np.random.default_rng(10_000 * repeat + 100 * outer_fold + int(seed))
                perm_dev, perm_test = dev.copy(), test.copy()
                perm_dev["max_similarity"] = rng.permutation(dev.max_similarity.to_numpy())
                perm_test["max_similarity"] = rng.permutation(test.max_similarity.to_numpy())
                neg_probability, neg_alpha, _ = fit_full_candidate(perm_dev, perm_test, "D1_shrunk_similarity", config)
                record_prediction_rows(output_rows, test, neg_probability, neg_alpha, "N0_permuted_similarity", repeat, outer_fold, int(seed), False)
                d1_probability, d1_alpha, _ = fit_full_candidate(dev, test, "D1_shrunk_similarity", config)
                random_alpha = rng.beta(max(d1_alpha.mean() * 20, 0.1), max((1 - d1_alpha.mean()) * 20, 0.1), len(test))
                record_prediction_rows(output_rows, test, random_alpha * rf_test + (1 - random_alpha) * chem_test, random_alpha, "N1_random_matched_gate", repeat, outer_fold, int(seed), False)
                conf_probability, conf_alpha, _ = fit_full_candidate(dev, test, "D2_similarity_entropy", config) if False else _confidence_only(dev, test, config)
                record_prediction_rows(output_rows, test, conf_probability, conf_alpha, "N2_confidence_only", repeat, outer_fold, int(seed), False)
                feature_audit.append({"repeat": repeat, "outer_fold": outer_fold, "seed": int(seed), "method": "feature_ranges", "oof_max_similarity_mean": float(dev.max_similarity.mean()), "test_max_similarity_mean": float(test.max_similarity.mean()), "oof_entropy_difference_mean": float(dev.entropy_difference.mean()), "test_entropy_difference_mean": float(test.entropy_difference.mean())})
    table = pd.DataFrame(output_rows)
    expected = 2 * 5 * len(seeds) * len(METHODS)
    contexts = table.groupby(["repeat", "outer_fold", "seed", "method"]).source_row.nunique()
    if len(contexts) != expected or not (contexts == table.groupby(["repeat", "outer_fold", "seed", "method"]).size()).all():
        raise RuntimeError("Fusion output does not contain exactly one outer-test prediction per row/context/method.")
    metric_rows = []
    for keys, frame in table.groupby(["repeat", "outer_fold", "seed", "method"]):
        metric_rows.append({"repeat": keys[0], "outer_fold": keys[1], "seed": keys[2], "method": keys[3], **metrics(frame.label.to_numpy(), frame.probability.to_numpy())})
    context_metrics = pd.DataFrame(metric_rows)
    summary = context_metrics.groupby("method", as_index=False).agg(**{f"{metric}_{stat}": (metric, stat) for metric in ["brier_score", "log_loss", "pr_auc", "roc_auc"] for stat in ("mean", "std")}, n_contexts=("brier_score", "size"))
    prediction_path, metric_path, selection_path, feature_path, metadata_path = (ROOT / outputs[key] for key in ("fusion_predictions_csv", "fusion_metrics_csv", "gate_selection_csv", "feature_audit_csv", "metadata_json"))
    for path in (prediction_path, metric_path, selection_path, feature_path, metadata_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(prediction_path, index=False)
    pd.concat([context_metrics.assign(row_type="outer_context"), summary.assign(row_type="summary")], ignore_index=True, sort=False).to_csv(metric_path, index=False)
    pd.DataFrame(selection_rows).to_csv(selection_path, index=False)
    pd.DataFrame(feature_audit).to_csv(feature_path, index=False)
    metadata = {"phase": "phase5_v2_fusion", "generated_at_utc": datetime.now(timezone.utc).isoformat(), "inputs": {"calibrated_predictions": str(pred_path.relative_to(ROOT)), "sha256": sha256_file(pred_path)}, "config_sha256": sha256_file(config_path), "component_input": "probability_platt for both models", "gate_fit_policy": "Gate selection and fitting use only OOF-development predictions; outer test is only predicted after fitting.", "one_se_policy": "Choose the least-complex D1-D4 gate whose mean cross-fitted inner-fold Brier loss is within one standard error of the best.", "methods": METHODS, "prediction_rows": int(len(table))}
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return {"predictions": prediction_path, "metrics": metric_path, "selection": selection_path, "feature_audit": feature_path, "metadata": metadata_path}


def _confidence_only(dev: pd.DataFrame, test: pd.DataFrame, config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Negative control: entropy difference without structural similarity."""
    fixed_alpha = fixed_weight(dev.label.to_numpy(), dev.rf_probability.to_numpy(), dev.chem_probability.to_numpy())
    X_train, X_test, scale = standardize_fit(dev, test, ["entropy_difference"])
    X_train, X_test = np.column_stack([np.ones(len(dev)), X_train]), np.column_stack([np.ones(len(test)), X_test])
    opt = config["optimization"]
    theta = fit_shrunk_gate(dev.label.to_numpy(), dev.rf_probability.to_numpy(), dev.chem_probability.to_numpy(), X_train, fixed_alpha, float(opt["beta_l2"]), float(opt["lambda_l2"]), int(opt["optimizer_maxiter"]))
    probability, alpha = shrunk_predict(test.rf_probability.to_numpy(), test.chem_probability.to_numpy(), X_test, fixed_alpha, theta)
    return probability, alpha, {"features": ["entropy_difference"], "fixed_alpha": fixed_alpha, "lambda": float(sigmoid(theta[-1])), "beta": theta[:-1].tolist(), "standardization": scale}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs_v2/phase5_fusion.yaml")
    args = parser.parse_args()
    for name, path in run(args.config.resolve()).items():
        print(f"{name}: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
