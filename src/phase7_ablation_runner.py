"""Run one Phase 7 expert variant on the frozen 2x5 nested scaffold folds.

This is exploratory BACE reuse after Phase 5. Hyperparameters and epochs are
chosen on each outer-development set; outer labels are read only for scoring.
Each variant writes separate resumable-style artifacts for later consolidation.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from transformers import AutoTokenizer

from phase7_models import FeatureBundle, fit_predict
from train_base_models_v2 import sha256_file


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = (
    "mean_mlp", "cls_mlp", "attention_mlp", "mean_logistic", "mean_mlp_aug2",
    "mean_mlp_ft_last0", "mean_mlp_ft_last1", "mean_mlp_ft_last2",
)


def identifier_hash(values: np.ndarray) -> str:
    return hashlib.sha256("\n".join(map(str, sorted(int(value) for value in values))).encode()).hexdigest()


def parameter_grid(variant: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    if variant.startswith("mean_mlp_ft_"):
        return [{"learning_rate": float(config["finetuning"]["learning_rate"])}]
    spec = config["frozen_head"]["logistic_grid" if variant == "mean_logistic" else "mlp_grid"]
    return [dict(zip(spec, values)) for values in itertools.product(*(spec[key] for key in spec))]


def maximum_similarity(query: np.ndarray, references: np.ndarray, fingerprints: list[Any]) -> np.ndarray:
    if len(references) == 0:
        raise ValueError("Similarity reference set is empty.")
    reference_fingerprints = [fingerprints[index] for index in references]
    return np.asarray([max(DataStructs.BulkTanimotoSimilarity(fingerprints[index], reference_fingerprints))
                       for index in query], dtype=float)


def similarity_labels(
    inner_folds: np.ndarray, development: np.ndarray, outer_test: np.ndarray,
    fingerprints: list[Any], quantile: float,
) -> tuple[np.ndarray, float]:
    oof = []
    for fold in (1, 2, 3):
        query = development[inner_folds == fold]
        reference = development[inner_folds != fold]
        oof.extend(maximum_similarity(query, reference, fingerprints))
    cutoff = float(np.quantile(oof, quantile))
    return maximum_similarity(outer_test, development, fingerprints), cutoff


def score(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "average_precision": float(average_precision_score(y, p)) if y.sum() else np.nan,
        "brier_score": float(brier_score_loss(y, p)),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
    }


def fit_platt(y: np.ndarray, p: np.ndarray, config: dict[str, Any]) -> LogisticRegression:
    clip = float(config["probability_clip"])
    clipped = np.clip(p, clip, 1 - clip)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    return LogisticRegression(C=float(config["C"]), max_iter=int(config["max_iter"]), solver="lbfgs").fit(logits, y)


def apply_platt(model: LogisticRegression, p: np.ndarray, clip: float) -> np.ndarray:
    clipped = np.clip(p, clip, 1 - clip)
    return model.predict_proba(np.log(clipped / (1 - clipped)).reshape(-1, 1))[:, 1]


def load_features(config: dict[str, Any], data: pd.DataFrame) -> FeatureBundle:
    outputs, spec = config["outputs"], config["encoder"]
    with np.load(ROOT / outputs["token_cache"]) as tokens:
        if not np.array_equal(tokens["source_row"], data.source_row.to_numpy()):
            raise ValueError("Token-cache source rows differ from standardized data.")
        mean, cls, hidden, mask = (tokens[key] for key in ("mean", "cls", "hidden", "mask"))
    with np.load(ROOT / outputs["augmented_cache"]) as augmented:
        if not np.array_equal(augmented["source_row"], data.source_row.to_numpy()):
            raise ValueError("Augmented-cache source rows differ from standardized data.")
        augmented_mean = augmented["X"].astype(np.float32)
    tokenizer = AutoTokenizer.from_pretrained(spec["checkpoint"], cache_dir=ROOT / spec["local_cache"],
                                               local_files_only=bool(spec["local_files_only"]))
    encoded = tokenizer(data.standardized_smiles.astype(str).tolist(), padding="max_length",
                        truncation=True, max_length=mask.shape[1], return_tensors="np")
    if not np.array_equal(encoded["attention_mask"], mask):
        raise ValueError("Tokenized canonical inputs differ from frozen-cache masks.")
    return FeatureBundle(mean=mean.astype(np.float32), cls=cls.astype(np.float32), hidden=hidden,
                         mask=mask, augmented_mean=augmented_mean,
                         input_ids=encoded["input_ids"], input_mask=encoded["attention_mask"],
                         labels=data.Class.to_numpy(int))


def assert_fit_boundary(
    train: np.ndarray, query: np.ndarray, outer_test: np.ndarray,
    data: pd.DataFrame, stage: str, context: dict[str, Any],
) -> dict[str, Any]:
    train_sources = data.iloc[train].source_row.to_numpy(int)
    query_sources = data.iloc[query].source_row.to_numpy(int)
    outer_sources = data.iloc[outer_test].source_row.to_numpy(int)
    source_overlap = len(set(train_sources) & set(query_sources))
    scaffold_overlap = len(set(data.iloc[train].scaffold) & set(data.iloc[query].scaffold))
    outer_overlap = len(set(train_sources) & set(outer_sources))
    if source_overlap or scaffold_overlap or outer_overlap:
        raise ValueError(f"Fit boundary failed: {stage} {context}")
    return {**context, "stage": stage, "fit_n": len(train), "query_n": len(query),
            "fit_source_sha256": identifier_hash(train_sources),
            "query_source_sha256": identifier_hash(query_sources),
            "source_overlap_n": source_overlap, "scaffold_overlap_n": scaffold_overlap,
            "outer_test_fit_overlap_n": outer_overlap}


def run(config_path: Path, variant: str, only_repeat: int | None, only_outer_fold: int | None) -> dict[str, Path]:
    if variant not in VARIANTS:
        raise ValueError(f"Unsupported variant: {variant}")
    config = yaml.safe_load(config_path.read_text())
    torch.set_num_threads(int(config["encoder"]["torch_threads"]))
    data = pd.read_csv(ROOT / config["inputs"]["standardized_csv"], keep_default_na=False).sort_values("source_row").reset_index(drop=True)
    if len(data) != 1513 or data.source_row.duplicated().any():
        raise ValueError("Unexpected BACE standardized-data identity.")
    features = load_features(config, data)
    source_to_position = {int(source): index for index, source in enumerate(data.source_row)}
    fingerprint_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    fingerprints = [fingerprint_generator.GetFingerprint(Chem.MolFromSmiles(value)) for value in data.standardized_smiles]
    rf = pd.read_csv(ROOT / config["inputs"]["calibrated_rf_csv"])
    rf = rf[(rf.model == "morgan_rf") & (rf.split == "outer_test")]
    prediction_rows, metric_rows, search_rows, audit_rows = [], [], [], []
    experiment = config["experiment"]
    encoder_config = dict(config["encoder"])
    encoder_config["local_cache"] = str(ROOT / encoder_config["local_cache"])
    feature_metadata = json.loads((ROOT / config["outputs"]["feature_metadata_json"]).read_text())
    encoder_seconds_per_molecule = float(feature_metadata["canonical_encoder_seconds"]) / len(data)
    for repeat in experiment["repeats"]:
        if only_repeat is not None and int(repeat) != only_repeat:
            continue
        outer_manifest = pd.read_csv(ROOT / config["inputs"]["split_directory"] / f"outer_repeat_{repeat}.csv")
        for outer_fold in experiment["outer_folds"]:
            if only_outer_fold is not None and int(outer_fold) != only_outer_fold:
                continue
            inner_manifest = pd.read_csv(ROOT / config["inputs"]["split_directory"] /
                                         f"inner_repeat_{repeat}_outer_fold_{outer_fold}.csv")
            development = np.asarray([source_to_position[int(value)] for value in inner_manifest.source_row], dtype=int)
            inner_folds = inner_manifest.inner_fold.to_numpy(int)
            outer_sources = outer_manifest.loc[outer_manifest.outer_fold == outer_fold, "source_row"].to_numpy(int)
            outer_test = np.asarray([source_to_position[value] for value in outer_sources], dtype=int)
            if set(development) & set(outer_test) or len(development) + len(outer_test) != len(data):
                raise ValueError("Inner development and outer test do not partition all molecules.")
            test_similarity, cutoff = similarity_labels(inner_folds, development, outer_test, fingerprints,
                                                        float(experiment["low_similarity_development_quantile"]))
            low = test_similarity <= cutoff
            candidates = parameter_grid(variant, config)
            tuning = []
            for candidate_index, params in enumerate(candidates, start=1):
                scores, epochs = [], []
                for inner_fold in experiment["inner_folds"]:
                    train = development[inner_folds != inner_fold]
                    valid = development[inner_folds == inner_fold]
                    context = {"variant": variant, "repeat": repeat, "outer_fold": outer_fold,
                               "seed": int(experiment["selection_seed"]), "inner_fold": inner_fold,
                               "candidate_index": candidate_index}
                    audit_rows.append(assert_fit_boundary(train, valid, outer_test, data, "inner_parameter_search", context))
                    max_epochs = int(config["finetuning"]["max_epochs"] if "ft_last" in variant else config["frozen_head"]["max_epochs"])
                    patience = int(config["finetuning"]["patience"] if "ft_last" in variant else config["frozen_head"]["patience"])
                    probability, epoch, fit_seconds, infer_seconds = fit_predict(
                        variant, train, valid, features, params, int(experiment["selection_seed"]),
                        max_epochs, None if variant == "mean_logistic" else patience,
                        encoder_config, config["finetuning"],
                    )
                    value = score(features.labels[valid], probability)
                    scores.append(value["average_precision"])
                    epochs.append(epoch)
                    search_rows.append({**context, "params_json": json.dumps(params, sort_keys=True),
                                        "selected_epoch": epoch, "fit_seconds": fit_seconds,
                                        "inference_seconds": infer_seconds, **value})
                tuning.append((float(np.mean(scores)), int(np.median(epochs)), candidate_index, params))
            best_score, selected_epochs, best_index, best_params = max(tuning, key=lambda row: (row[0], -row[2]))
            for row in search_rows:
                if row["variant"] == variant and row["repeat"] == repeat and row["outer_fold"] == outer_fold:
                    row["selected"] = row["candidate_index"] == best_index
                    row["selected_outer_epochs"] = selected_epochs
            for seed in experiment["training_seeds"]:
                seed = int(seed)
                oof = np.full(len(development), np.nan)
                fit_seconds_total = 0.0
                for inner_fold in experiment["inner_folds"]:
                    train_mask, valid_mask = inner_folds != inner_fold, inner_folds == inner_fold
                    train, valid = development[train_mask], development[valid_mask]
                    context = {"variant": variant, "repeat": repeat, "outer_fold": outer_fold,
                               "seed": seed, "inner_fold": inner_fold, "candidate_index": best_index}
                    audit_rows.append(assert_fit_boundary(train, valid, outer_test, data, "calibration_oof_fit", context))
                    probability, _, fit_seconds, _ = fit_predict(
                        variant, train, valid, features, best_params, seed,
                        selected_epochs if selected_epochs else 1, None,
                        encoder_config, config["finetuning"],
                    )
                    oof[valid_mask] = probability
                    fit_seconds_total += fit_seconds
                if not np.isfinite(oof).all():
                    raise ValueError("Incomplete Phase 7 OOF predictions.")
                platt = fit_platt(features.labels[development], oof, config["calibration"])
                context = {"variant": variant, "repeat": repeat, "outer_fold": outer_fold,
                           "seed": seed, "inner_fold": 0, "candidate_index": best_index}
                audit_rows.append(assert_fit_boundary(development, outer_test, outer_test, data, "outer_full_development_fit", context))
                raw, _, fit_seconds, infer_seconds = fit_predict(
                    variant, development, outer_test, features, best_params, seed,
                    selected_epochs if selected_epochs else 1, None,
                    encoder_config, config["finetuning"],
                )
                fit_seconds_total += fit_seconds
                calibrated = apply_platt(platt, raw, float(config["calibration"]["probability_clip"]))
                if not np.isfinite(calibrated).all() or not np.all((calibrated >= 0) & (calibrated <= 1)):
                    raise ValueError("Invalid calibrated Phase 7 probabilities.")
                rf_context = rf[(rf.repeat == repeat) & (rf.outer_fold == outer_fold) & (rf.seed == seed)]
                rf_context = rf_context.set_index("source_row").loc[outer_sources]
                if not np.array_equal(rf_context.label.to_numpy(int), features.labels[outer_test]):
                    raise ValueError("RF reference labels are not paired with Phase 7 rows.")
                rf_probability = rf_context.probability_platt.to_numpy(float)
                inference_ms = 1000 * (infer_seconds / len(outer_test) +
                                       (0 if "ft_last" in variant else encoder_seconds_per_molecule))
                for position, raw_value, calibrated_value, similarity, low_flag in zip(
                    outer_test, raw, calibrated, test_similarity, low,
                ):
                    row = data.iloc[position]
                    prediction_rows.append({
                        "CID": row.CID, "source_row": int(row.source_row), "repeat": repeat,
                        "outer_fold": outer_fold, "seed": seed, "variant": variant,
                        "split": "outer_test", "label": int(row.Class),
                        "raw_probability": float(raw_value), "probability": float(calibrated_value),
                        "max_similarity": float(similarity), "low_similarity_cutoff": cutoff,
                        "low_similarity": bool(low_flag), "scaffold": row.scaffold,
                    })
                for stratum, selected in (("all", np.ones(len(outer_test), dtype=bool)),
                                          ("low_similarity", low), ("high_similarity", ~low)):
                    y = features.labels[outer_test][selected]
                    result = score(y, calibrated[selected])
                    rf_result = score(y, rf_probability[selected])
                    metric_rows.append({"variant": variant, "repeat": repeat, "outer_fold": outer_fold,
                                        "seed": seed, "stratum": stratum, "n_molecules": len(y),
                                        "n_active": int(y.sum()), "active_fraction": float(y.mean()),
                                        **result, "rf_average_precision": rf_result["average_precision"],
                                        "rf_brier_score": rf_result["brier_score"],
                                        "delta_ap_vs_rf": result["average_precision"] - rf_result["average_precision"],
                                        "delta_brier_vs_rf": result["brier_score"] - rf_result["brier_score"],
                                        "selected_params_json": json.dumps(best_params, sort_keys=True),
                                        "selected_epochs": selected_epochs,
                                        "inner_selection_ap": best_score,
                                        "fit_seconds_oof_plus_final": fit_seconds_total,
                                        "approx_inference_ms_per_molecule": inference_ms})
            print(f"{variant}: repeat={repeat} outer_fold={outer_fold} complete", flush=True)
    work = ROOT / "results_v2/phase7_work"
    work.mkdir(parents=True, exist_ok=True)
    paths = {
        "predictions": work / f"{variant}_predictions.csv.gz",
        "metrics": work / f"{variant}_metrics.csv",
        "selection": work / f"{variant}_selection.csv",
        "fit_audit": work / f"{variant}_fit_audit.csv",
        "metadata": work / f"{variant}_metadata.json",
    }
    pd.DataFrame(prediction_rows).sort_values(["repeat", "outer_fold", "seed", "source_row"]).to_csv(
        paths["predictions"], index=False, compression={"method": "gzip", "mtime": 0})
    pd.DataFrame(metric_rows).sort_values(["repeat", "outer_fold", "seed", "stratum"]).to_csv(paths["metrics"], index=False)
    pd.DataFrame(search_rows).sort_values(["repeat", "outer_fold", "candidate_index", "inner_fold"]).to_csv(paths["selection"], index=False)
    pd.DataFrame(audit_rows).to_csv(paths["fit_audit"], index=False)
    paths["metadata"].write_text(json.dumps({
        "phase": "phase7_expert_variant", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "post_phase5_exploratory", "variant": variant,
        "config_sha256": sha256_file(config_path),
        "feature_metadata_sha256": sha256_file(ROOT / config["outputs"]["feature_metadata_json"]),
        "prediction_rows": len(prediction_rows), "metric_rows": len(metric_rows),
        "fit_audit_rows": len(audit_rows), "outer_contexts": len(set((row["repeat"], row["outer_fold"]) for row in metric_rows)),
        "script_sha256": sha256_file(Path(__file__)),
    }, indent=2) + "\n")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs_v2/phase7_expert_ablations.yaml")
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--repeat", type=int)
    parser.add_argument("--outer-fold", type=int)
    args = parser.parse_args()
    for key, path in run(args.config.resolve(), args.variant, args.repeat, args.outer_fold).items():
        print(f"{key}: {path.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
