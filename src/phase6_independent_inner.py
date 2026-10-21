"""Post-Phase-5, label-independent inner validation for gate selection.

For each held-out inner fold, base-model OOF predictions are regenerated using
only the other two scaffold folds. Calibration and fusion are then fitted on
those two folds. The held-out fold has no influence on any fitted stage.
This is a dated sensitivity analysis with fixed, non-tuned hyperparameters;
the frozen Phase 5 outer-test predictions are preserved separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import torch
import yaml
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import StandardScaler

from fingerprints import MorganConfig, morgan_fingerprints
from fusion_v2 import (
    SIMILARITY_CANDIDATES,
    entropy,
    fit_full_candidate,
    fit_original_gate,
    fixed_weight,
    sigmoid,
    similarity_features,
    standardize_fit,
)
from train_base_models_v2 import fit_mlp, predict_mlp, sha256_file


ROOT = Path(__file__).resolve().parents[1]
INNER_METHODS = [
    "A1_calibrated_rf", "B1_calibrated_chemberta", "C0_equal_weight",
    "C1_fixed_weight", "D0_original_similarity_gate", *SIMILARITY_CANDIDATES,
    "E0_logistic_stacking",
]


def identifier_hash(values: np.ndarray) -> str:
    ordered = sorted(int(value) for value in values)
    return hashlib.sha256("\n".join(map(str, ordered)).encode()).hexdigest()


def fit_platt_on_oof(y: np.ndarray, raw: np.ndarray, spec: dict[str, Any]) -> LogisticRegression:
    clip = float(spec["probability_clip"])
    clipped = np.clip(raw, clip, 1 - clip)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    return LogisticRegression(C=float(spec["C"]), solver=str(spec["solver"]), max_iter=int(spec["max_iter"])).fit(logits, y)


def calibrated_probability(calibrator: LogisticRegression, raw: np.ndarray, clip: float) -> np.ndarray:
    clipped = np.clip(raw, clip, 1 - clip)
    return calibrator.predict_proba(np.log(clipped / (1 - clipped)).reshape(-1, 1))[:, 1]


def fit_experts(
    train_positions: np.ndarray, query_positions: np.ndarray, y: np.ndarray,
    X_rf: np.ndarray, X_chem: np.ndarray, seed: int, spec: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    rf_spec, chem_spec = spec["rf"], spec["chemberta_head"]
    rf = RandomForestClassifier(
        n_estimators=int(rf_spec["n_estimators"]), class_weight=str(rf_spec["class_weight"]),
        max_depth=int(rf_spec["max_depth"]) if rf_spec["max_depth"] is not None else None,
        min_samples_leaf=int(rf_spec["min_samples_leaf"]), max_features=rf_spec["max_features"],
        random_state=seed, n_jobs=int(rf_spec["n_jobs"]),
    ).fit(X_rf[train_positions], y[train_positions])
    rf_probability = rf.predict_proba(X_rf[query_positions])[:, 1]
    scaler = StandardScaler().fit(X_chem[train_positions])
    params = {key: chem_spec[key] for key in ("hidden_dim", "dropout", "learning_rate", "weight_decay")}
    chem_model, _ = fit_mlp(
        scaler.transform(X_chem[train_positions]).astype(np.float32), y[train_positions],
        params, seed, int(chem_spec["epochs"]),
    )
    chem_probability = predict_mlp(chem_model, scaler.transform(X_chem[query_positions]).astype(np.float32))
    return rf_probability, chem_probability


def add_features(
    frame: pd.DataFrame, queries: np.ndarray, references: np.ndarray,
    y: np.ndarray, fingerprints: list[Any], top_k: int,
) -> pd.DataFrame:
    values = similarity_features(queries, references, fingerprints, y[references], top_k)
    frame = frame.copy()
    frame["max_similarity"] = values[:, 0]
    frame["topk_mean_similarity"] = values[:, 1]
    frame["topk_active_fraction"] = values[:, 2]
    return frame


def fit_trace(
    repeat: int, outer_fold: int, seed: int, held_fold: int, stage: str,
    fit_positions: np.ndarray, validation_positions: np.ndarray,
    outer_positions: np.ndarray, data: pd.DataFrame,
) -> dict[str, Any]:
    fit_sources = data.iloc[fit_positions].source_row.to_numpy(int)
    validation_sources = data.iloc[validation_positions].source_row.to_numpy(int)
    outer_sources = data.iloc[outer_positions].source_row.to_numpy(int)
    fit_scaffolds = set(data.iloc[fit_positions].scaffold)
    validation_scaffolds = set(data.iloc[validation_positions].scaffold)
    overlap = set(fit_sources) & set(validation_sources)
    outer_overlap = set(fit_sources) & set(outer_sources)
    scaffold_overlap = fit_scaffolds & validation_scaffolds
    if overlap or outer_overlap or scaffold_overlap:
        raise RuntimeError(f"Fit partition leaks into held-out fold: {(repeat, outer_fold, seed, held_fold, stage)}")
    return {
        "repeat": repeat, "outer_fold": outer_fold, "seed": seed, "inner_validation_fold": held_fold,
        "stage": stage, "fit_n": len(fit_sources), "validation_n": len(validation_sources),
        "fit_source_sha256": identifier_hash(fit_sources),
        "validation_source_sha256": identifier_hash(validation_sources),
        "source_overlap_n": len(overlap), "scaffold_overlap_n": len(scaffold_overlap),
        "outer_test_overlap_n": len(outer_overlap),
    }


def method_predictions(dev: pd.DataFrame, valid: pd.DataFrame, gate_config: dict[str, Any]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rf, chem = valid.rf_probability.to_numpy(), valid.chem_probability.to_numpy()
    y_dev, rf_dev, chem_dev = dev.label.to_numpy(int), dev.rf_probability.to_numpy(), dev.chem_probability.to_numpy()
    alpha_fixed = fixed_weight(y_dev, rf_dev, chem_dev)
    answer: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "A1_calibrated_rf": (rf, np.full(len(valid), np.nan)),
        "B1_calibrated_chemberta": (chem, np.full(len(valid), np.nan)),
        "C0_equal_weight": (0.5 * rf + 0.5 * chem, np.full(len(valid), 0.5)),
        "C1_fixed_weight": (alpha_fixed * rf + (1 - alpha_fixed) * chem, np.full(len(valid), alpha_fixed)),
    }
    x_dev, x_valid, _ = standardize_fit(dev, valid, ["max_similarity"])
    beta = fit_original_gate(y_dev, rf_dev, chem_dev, np.column_stack((np.ones(len(dev)), x_dev)), int(gate_config["optimizer_maxiter"]))
    alpha0 = sigmoid(np.column_stack((np.ones(len(valid)), x_valid)) @ beta)
    answer["D0_original_similarity_gate"] = (alpha0 * rf + (1 - alpha0) * chem, alpha0)
    fusion_config = {"optimization": gate_config}
    for method in SIMILARITY_CANDIDATES:
        probability, alpha, _ = fit_full_candidate(dev, valid, method, fusion_config)
        answer[method] = (probability, alpha)
    stacker = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000).fit(dev[["rf_probability", "chem_probability"]], y_dev)
    answer["E0_logistic_stacking"] = (stacker.predict_proba(valid[["rf_probability", "chem_probability"]])[:, 1], np.full(len(valid), np.nan))
    return answer


def run(config_path: Path) -> dict[str, Path]:
    config = yaml.safe_load(config_path.read_text())
    inputs, outputs = config["inputs"], config["outputs"]
    spec = config["independent_inner_sensitivity"]
    torch.set_num_threads(int(spec["chemberta_head"]["torch_threads"]))
    data_path = ROOT / inputs["standardized_csv"]
    data = pd.read_csv(data_path, keep_default_na=False).sort_values("source_row").reset_index(drop=True)
    if data.source_row.duplicated().any() or len(data) != 1513:
        raise ValueError("Unexpected standardized-data identity.")
    source_to_position = {int(row): position for position, row in enumerate(data.source_row)}
    y = data.Class.to_numpy(int)
    rf_spec = spec["rf"]
    X_rf = morgan_fingerprints(
        data.standardized_smiles,
        MorganConfig(radius=int(rf_spec["radius"]), n_bits=int(rf_spec["fingerprint_size"]), include_chirality=bool(rf_spec["include_chirality"])),
    )
    embedding_path = ROOT / inputs["embedding_cache"]
    with np.load(embedding_path) as cache:
        if not np.array_equal(cache["source_row"], data.source_row.to_numpy()):
            raise ValueError("ChemBERTa embedding cache does not match the standardized rows.")
        X_chem = cache["X"].astype(np.float32)
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=int(rf_spec["radius"]), fpSize=int(rf_spec["fingerprint_size"]),
        includeChirality=bool(rf_spec["include_chirality"]),
    )
    fingerprints = [generator.GetFingerprint(Chem.MolFromSmiles(value)) for value in data.standardized_smiles]
    prediction_parts: list[pd.DataFrame] = []
    fit_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    seeds = [int(value) for value in spec["training_seeds"]]
    top_k = int(spec["similarity"]["top_k"])
    for repeat in (1, 2):
        outer = pd.read_csv(ROOT / inputs["split_directory"] / f"outer_repeat_{repeat}.csv")
        for outer_fold in range(1, 6):
            outer_sources = outer.loc[outer.outer_fold == outer_fold, "source_row"].to_numpy(int)
            outer_positions = np.asarray([source_to_position[value] for value in outer_sources])
            inner = pd.read_csv(ROOT / inputs["split_directory"] / f"inner_repeat_{repeat}_outer_fold_{outer_fold}.csv")
            if set(inner.source_row) & set(outer_sources):
                raise ValueError("Inner manifest contains outer-test rows.")
            for held_fold in (1, 2, 3):
                meta_sources = inner.loc[inner.inner_fold != held_fold, "source_row"].to_numpy(int)
                valid_sources = inner.loc[inner.inner_fold == held_fold, "source_row"].to_numpy(int)
                meta_positions = np.asarray([source_to_position[value] for value in meta_sources])
                valid_positions = np.asarray([source_to_position[value] for value in valid_sources])
                meta_inner = inner.set_index("source_row").loc[meta_sources, "inner_fold"].to_numpy(int)
                meta_features = pd.DataFrame({"source_row": meta_sources, "label": y[meta_positions]})
                for meta_fold in sorted(set(meta_inner)):
                    query_mask = meta_inner == meta_fold
                    query_pos, train_pos = meta_positions[query_mask], meta_positions[~query_mask]
                    values = similarity_features(query_pos, train_pos, fingerprints, y[train_pos], top_k)
                    meta_features.loc[query_mask, ["max_similarity", "topk_mean_similarity", "topk_active_fraction"]] = values
                valid_features = add_features(
                    pd.DataFrame({"source_row": valid_sources, "label": y[valid_positions]}),
                    valid_positions, meta_positions, y, fingerprints, top_k,
                )
                seed_frames: list[tuple[int, pd.DataFrame, pd.DataFrame]] = []
                for seed in seeds:
                    oof_rf, oof_chem = np.empty(len(meta_positions)), np.empty(len(meta_positions))
                    for meta_fold in sorted(set(meta_inner)):
                        query_mask = meta_inner == meta_fold
                        train_pos, query_pos = meta_positions[~query_mask], meta_positions[query_mask]
                        fit_rows.append(fit_trace(repeat, outer_fold, seed, held_fold, f"base_oof_train_for_fold_{meta_fold}", train_pos, valid_positions, outer_positions, data))
                        oof_rf[query_mask], oof_chem[query_mask] = fit_experts(train_pos, query_pos, y, X_rf, X_chem, seed, spec)
                    fit_rows.append(fit_trace(repeat, outer_fold, seed, held_fold, "base_full_meta_train", meta_positions, valid_positions, outer_positions, data))
                    valid_rf, valid_chem = fit_experts(meta_positions, valid_positions, y, X_rf, X_chem, seed, spec)
                    fit_rows.append(fit_trace(repeat, outer_fold, seed, held_fold, "platt_and_gate_fit", meta_positions, valid_positions, outer_positions, data))
                    clip = float(spec["calibration"]["probability_clip"])
                    rf_cal = fit_platt_on_oof(y[meta_positions], oof_rf, spec["calibration"])
                    chem_cal = fit_platt_on_oof(y[meta_positions], oof_chem, spec["calibration"])
                    meta = meta_features.copy()
                    valid = valid_features.copy()
                    meta["rf_probability"] = calibrated_probability(rf_cal, oof_rf, clip)
                    meta["chem_probability"] = calibrated_probability(chem_cal, oof_chem, clip)
                    valid["rf_probability"] = calibrated_probability(rf_cal, valid_rf, clip)
                    valid["chem_probability"] = calibrated_probability(chem_cal, valid_chem, clip)
                    seed_frames.append((seed, meta, valid))
                # D4's seed variance is computed only from fitted predictions.
                for column in ("rf_probability", "chem_probability"):
                    meta_stack = np.stack([frame[column].to_numpy() for _, frame, _ in seed_frames])
                    valid_stack = np.stack([frame[column].to_numpy() for _, _, frame in seed_frames])
                    for _, meta, valid in seed_frames:
                        meta[f"{column}_variance"] = meta_stack.var(axis=0)
                        valid[f"{column}_variance"] = valid_stack.var(axis=0)
                for seed, meta, valid in seed_frames:
                    for frame in (meta, valid):
                        frame["variance_difference"] = frame.chem_probability_variance - frame.rf_probability_variance
                        frame["entropy_difference"] = entropy(frame.chem_probability.to_numpy()) - entropy(frame.rf_probability.to_numpy())
                    results = method_predictions(meta, valid, spec["gate"])
                    for method, (probability, alpha) in results.items():
                        prediction_parts.append(pd.DataFrame({
                            "source_row": valid.source_row.to_numpy(int), "repeat": repeat,
                            "outer_fold": outer_fold, "seed": seed,
                            "inner_validation_fold": held_fold, "method": method,
                            "label": valid.label.to_numpy(int), "probability": probability,
                            "alpha_rf": alpha,
                        }))
                    for method in SIMILARITY_CANDIDATES:
                        probability = results[method][0]
                        gate_rows.append({
                            "repeat": repeat, "outer_fold": outer_fold, "seed": seed,
                            "inner_validation_fold": held_fold, "candidate": method,
                            "brier_score": float(brier_score_loss(valid.label.to_numpy(int), probability)),
                        })
            print(f"independent inner complete: repeat={repeat} outer_fold={outer_fold}", flush=True)
    predictions = pd.concat(prediction_parts, ignore_index=True).sort_values(
        ["repeat", "outer_fold", "seed", "method", "source_row"]
    )
    if not predictions.probability.between(0, 1).all() or predictions.duplicated(["repeat", "outer_fold", "seed", "method", "source_row"]).any():
        raise RuntimeError("Independent validation predictions violate range or uniqueness.")
    summary_rows: list[dict[str, Any]] = []
    for (repeat, outer_fold, seed), group in pd.DataFrame(gate_rows).groupby(["repeat", "outer_fold", "seed"]):
        candidate_stats = []
        for rank, method in enumerate(SIMILARITY_CANDIDATES, start=1):
            scores = group.loc[group.candidate == method].sort_values("inner_validation_fold").brier_score.to_numpy()
            candidate_stats.append({"repeat": repeat, "outer_fold": outer_fold, "seed": seed, "candidate": method,
                                    "complexity_rank": rank, "mean_brier": float(scores.mean()),
                                    "standard_error_brier": float(scores.std(ddof=1) / np.sqrt(len(scores))),
                                    "fold_brier": json.dumps(scores.tolist())})
        best = min(candidate_stats, key=lambda row: row["mean_brier"])
        threshold = best["mean_brier"] + best["standard_error_brier"]
        chosen = min((row for row in candidate_stats if row["mean_brier"] <= threshold), key=lambda row: row["complexity_rank"])["candidate"]
        for row in candidate_stats:
            row["one_se_threshold"] = threshold
            row["selected_by_one_se"] = row["candidate"] == chosen
            summary_rows.append(row)
    output_paths = {key: ROOT / outputs[key] for key in (
        "independent_inner_predictions_csv_gz", "independent_inner_fit_audit_csv",
        "independent_gate_selection_csv", "independent_inner_metadata_json",
    )}
    for path in output_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_paths["independent_inner_predictions_csv_gz"], index=False, compression={"method": "gzip", "mtime": 0})
    pd.DataFrame(fit_rows).to_csv(output_paths["independent_inner_fit_audit_csv"], index=False)
    pd.DataFrame(summary_rows).to_csv(output_paths["independent_gate_selection_csv"], index=False)
    metadata = {
        "phase": "phase6_independent_inner_sensitivity", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "post_phase5_exploratory_sensitivity", "config_sha256": sha256_file(config_path),
        "standardized_data_sha256": sha256_file(data_path), "embedding_cache_sha256": sha256_file(embedding_path),
        "fit_policy": "Each held inner fold is absent from base-model, Platt, fixed-weight, and gate fits.",
        "hyperparameter_policy": "Fixed phase6 sensitivity values; not selected on held inner labels.",
        "training_seeds": seeds, "prediction_rows": len(predictions), "methods": INNER_METHODS,
        "dependency_versions": {
            "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
            "torch": torch.__version__, "rdkit": Chem.rdBase.rdkitVersion,
        },
        "script_sha256": sha256_file(Path(__file__)),
    }
    output_paths["independent_inner_metadata_json"].write_text(json.dumps(metadata, indent=2) + "\n")
    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs_v2/phase6_statistical_evaluation.yaml")
    args = parser.parse_args()
    for name, path in run(args.config.resolve()).items():
        print(f"{name}: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
