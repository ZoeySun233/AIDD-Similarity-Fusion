"""Nested-CV OOF and outer-test predictions for the two frozen v2 experts.

The only label-aware operations in this module occur inside an outer
development set.  ChemBERTa embeddings are label-free frozen representations;
every probability used later for calibration or fusion is produced by a model
that excluded that molecule from its fitting data.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn

import chemberta
from fingerprints import MorganConfig, morgan_fingerprints


ROOT = Path(__file__).resolve().parents[1]
PREDICTION_COLUMNS = ["CID", "source_row", "repeat", "outer_fold", "seed", "model", "split", "label", "probability"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metric_values(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    return {
        "pr_auc": float(average_precision_score(y, probability)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "brier_score": float(brier_score_loss(y, probability)),
    }


def param_grid(search: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(search)
    return [dict(zip(keys, values)) for values in itertools.product(*(search[key] for key in keys))]


def choose_params(rows: list[dict[str, Any]]) -> dict[str, Any]:
    table = pd.DataFrame(rows)
    grouped = table.groupby("param_json", as_index=False).agg(
        mean_pr_auc=("pr_auc", "mean"), mean_roc_auc=("roc_auc", "mean"), mean_brier_score=("brier_score", "mean")
    )
    best = grouped.sort_values(["mean_pr_auc", "mean_roc_auc", "mean_brier_score"], ascending=[False, False, True]).iloc[0]
    return json.loads(best["param_json"])


def build_rf(params: dict[str, Any], seed: int, config: dict[str, Any]) -> RandomForestClassifier:
    normalized = dict(params)
    if normalized["max_depth"] is not None:
        normalized["max_depth"] = int(normalized["max_depth"])
    normalized["min_samples_leaf"] = int(normalized["min_samples_leaf"])
    return RandomForestClassifier(
        n_estimators=int(config["n_estimators"]), class_weight=config["class_weight"],
        random_state=seed, n_jobs=-1, **normalized,
    )


def fit_mlp(
    X_train: np.ndarray, y_train: np.ndarray, params: dict[str, Any], seed: int,
    max_epochs: int, X_valid: np.ndarray | None = None, y_valid: np.ndarray | None = None,
    patience: int | None = None,
) -> tuple[nn.Module, int]:
    """Fit a frozen-embedding classifier; validation is only an inner-fold operation."""
    chemberta.set_seed(seed)
    model = chemberta.ActivityMLP(X_train.shape[1], int(params["hidden_dim"]), float(params["dropout"]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(params["learning_rate"]), weight_decay=float(params["weight_decay"]))
    pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    features = torch.from_numpy(X_train).float()
    labels = torch.from_numpy(y_train).float()
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch, best_score, stale = 1, -np.inf, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(features), labels)
        loss.backward()
        optimizer.step()
        if X_valid is None:
            continue
        probability = predict_mlp(model, X_valid)
        score = average_precision_score(y_valid, probability)
        if score > best_score + 1e-7:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if patience is not None and stale >= patience:
            break
    if X_valid is not None and best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch if X_valid is not None else max_epochs


def predict_mlp(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        return torch.sigmoid(model(torch.from_numpy(X).float())).numpy()


def append_predictions(
    sink: list[dict[str, Any]], frame: pd.DataFrame, probability: np.ndarray, *, repeat: int,
    outer_fold: int, seed: int, model: str, split: str,
) -> None:
    for (_, row), value in zip(frame.iterrows(), probability):
        sink.append({
            "CID": row["CID"], "source_row": int(row["source_row"]), "repeat": repeat,
            "outer_fold": outer_fold, "seed": seed, "model": model, "split": split,
            "label": int(row["Class"]), "probability": float(value),
        })


def embedding_matrix(data: pd.DataFrame, config: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    output = config["outputs"]
    cache_path = ROOT / output["embedding_cache"]
    metadata_path = ROOT / output["embedding_metadata_json"]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    smiles = data[config["data"]["smiles_column"]].astype(str).to_numpy(dtype=str)
    if cache_path.exists():
        cached = np.load(cache_path)
        if np.array_equal(cached["source_row"], data["source_row"].to_numpy()) and np.array_equal(cached["smiles"], smiles):
            metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
            return cached["X"].astype(np.float32), metadata
    encoder_config = config["frozen_chemberta_mlp"]
    frame = pd.DataFrame({"canonical_smiles": smiles})
    matrix, metadata = chemberta.extract_embeddings(
        frame["canonical_smiles"], ROOT / "models/huggingface",
        batch_size=int(encoder_config["embedding_batch_size"]), max_length=int(encoder_config["max_length"]),
    )
    np.savez_compressed(cache_path, X=matrix.astype(np.float32), source_row=data["source_row"].to_numpy(), smiles=smiles)
    metadata.update({"input": "standardized_smiles", "rows": int(len(data)), "source_rows_sha256": hashlib.sha256("\n".join(map(str, data["source_row"])).encode()).hexdigest()})
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return matrix.astype(np.float32), metadata


def _manifest(data: pd.DataFrame, repeat: int) -> pd.DataFrame:
    manifest = pd.read_csv(ROOT / f"data/splits_v2/outer_repeat_{repeat}.csv")
    required = {"source_row", "outer_fold"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"Outer manifest {repeat} lacks {required}")
    merged = data.merge(manifest[["source_row", "outer_fold"]], on="source_row", validate="one_to_one")
    return merged.sort_values("source_row").reset_index(drop=True)


def _inner_fold_lookup(repeat: int, outer_fold: int) -> pd.DataFrame:
    return pd.read_csv(ROOT / f"data/splits_v2/inner_repeat_{repeat}_outer_fold_{outer_fold}.csv")[["source_row", "inner_fold"]]


def run(config_path: Path) -> dict[str, Path]:
    config = yaml.safe_load(config_path.read_text())
    data_path = ROOT / config["data"]["standardized_csv"]
    data = pd.read_csv(data_path, keep_default_na=False).sort_values("source_row").reset_index(drop=True)
    required = {"source_row", "CID", "Class", config["data"]["smiles_column"]}
    if missing := required.difference(data.columns):
        raise ValueError(f"Standardized data missing required columns: {sorted(missing)}")
    if len(data) != 1513 or data.source_row.duplicated().any():
        raise ValueError("Unexpected v2 source-row identity; rerun Phase 1 before training.")
    morgan_cfg = config["morgan_rf"]
    X_rf = morgan_fingerprints(data[config["data"]["smiles_column"]], MorganConfig(radius=int(morgan_cfg["radius"]), n_bits=int(morgan_cfg["fingerprint_size"]), include_chirality=bool(morgan_cfg["include_chirality"])))
    X_chem, embedding_metadata = embedding_matrix(data, config)
    source_to_position = {int(row): index for index, row in enumerate(data.source_row)}
    predictions: list[dict[str, Any]] = []
    search_rows: list[dict[str, Any]] = []
    seeds = [int(value) for value in config["training_seeds"]]
    selection_seed = int(config["selection"]["seed"])
    chem_cfg = config["frozen_chemberta_mlp"]

    for repeat in (1, 2):
        outer_manifest = _manifest(data, repeat)
        for outer_fold in range(1, 6):
            development = outer_manifest[outer_manifest.outer_fold != outer_fold].copy()
            test = outer_manifest[outer_manifest.outer_fold == outer_fold].copy()
            inner = _inner_fold_lookup(repeat, outer_fold)
            development = development.merge(inner, on="source_row", validate="one_to_one")
            dev_positions = np.array([source_to_position[int(row)] for row in development.source_row])
            test_positions = np.array([source_to_position[int(row)] for row in test.source_row])
            y_dev, y_test = development.Class.to_numpy(dtype=int), test.Class.to_numpy(dtype=int)

            rf_search: list[dict[str, Any]] = []
            chem_search: list[dict[str, Any]] = []
            for inner_fold in range(1, 4):
                train = development[development.inner_fold != inner_fold]
                valid = development[development.inner_fold == inner_fold]
                train_pos = np.array([source_to_position[int(row)] for row in train.source_row])
                valid_pos = np.array([source_to_position[int(row)] for row in valid.source_row])
                for params in param_grid(morgan_cfg["search"]):
                    model = build_rf(params, selection_seed, morgan_cfg).fit(X_rf[train_pos], train.Class.to_numpy(dtype=int))
                    rf_search.append({"model": "morgan_rf", "repeat": repeat, "outer_fold": outer_fold, "inner_fold": inner_fold, "seed": selection_seed, "param_json": json.dumps(params, sort_keys=True), **metric_values(valid.Class.to_numpy(dtype=int), model.predict_proba(X_rf[valid_pos])[:, 1])})
                scaler = StandardScaler().fit(X_chem[train_pos])
                X_train, X_valid = scaler.transform(X_chem[train_pos]).astype(np.float32), scaler.transform(X_chem[valid_pos]).astype(np.float32)
                for params in param_grid(chem_cfg["search"]):
                    model, best_epoch = fit_mlp(X_train, train.Class.to_numpy(dtype=int), params, selection_seed, int(chem_cfg["max_epochs"]), X_valid, valid.Class.to_numpy(dtype=int), int(chem_cfg["patience"]))
                    chem_search.append({"model": "frozen_chemberta_mlp", "repeat": repeat, "outer_fold": outer_fold, "inner_fold": inner_fold, "seed": selection_seed, "param_json": json.dumps(params, sort_keys=True), "best_epoch": best_epoch, **metric_values(valid.Class.to_numpy(dtype=int), predict_mlp(model, X_valid))})
            rf_params, chem_params = choose_params(rf_search), choose_params(chem_search)
            selected_epoch = int(np.median([row["best_epoch"] for row in chem_search if json.loads(row["param_json"]) == chem_params]))
            for row in rf_search:
                row["selected"] = json.loads(row["param_json"]) == rf_params
                search_rows.append(row)
            for row in chem_search:
                row["selected"] = json.loads(row["param_json"]) == chem_params
                search_rows.append(row)

            # Cross-fitted development predictions: each row is held out by its inner fold.
            for seed in seeds:
                oof_rf = np.empty(len(development), dtype=float)
                oof_chem = np.empty(len(development), dtype=float)
                for inner_fold in range(1, 4):
                    train_mask, valid_mask = development.inner_fold.to_numpy() != inner_fold, development.inner_fold.to_numpy() == inner_fold
                    train_pos, valid_pos = dev_positions[train_mask], dev_positions[valid_mask]
                    rf_model = build_rf(rf_params, seed, morgan_cfg).fit(X_rf[train_pos], y_dev[train_mask])
                    oof_rf[valid_mask] = rf_model.predict_proba(X_rf[valid_pos])[:, 1]
                    scaler = StandardScaler().fit(X_chem[train_pos])
                    chem_model, _ = fit_mlp(scaler.transform(X_chem[train_pos]).astype(np.float32), y_dev[train_mask], chem_params, seed, selected_epoch)
                    oof_chem[valid_mask] = predict_mlp(chem_model, scaler.transform(X_chem[valid_pos]).astype(np.float32))
                append_predictions(predictions, development, oof_rf, repeat=repeat, outer_fold=outer_fold, seed=seed, model="morgan_rf", split="oof_development")
                append_predictions(predictions, development, oof_chem, repeat=repeat, outer_fold=outer_fold, seed=seed, model="frozen_chemberta_mlp", split="oof_development")
                # The final outer model sees all development labels, never outer-test labels.
                rf_final = build_rf(rf_params, seed, morgan_cfg).fit(X_rf[dev_positions], y_dev)
                append_predictions(predictions, test, rf_final.predict_proba(X_rf[test_positions])[:, 1], repeat=repeat, outer_fold=outer_fold, seed=seed, model="morgan_rf", split="outer_test")
                scaler = StandardScaler().fit(X_chem[dev_positions])
                chem_final, _ = fit_mlp(scaler.transform(X_chem[dev_positions]).astype(np.float32), y_dev, chem_params, seed, selected_epoch)
                append_predictions(predictions, test, predict_mlp(chem_final, scaler.transform(X_chem[test_positions]).astype(np.float32)), repeat=repeat, outer_fold=outer_fold, seed=seed, model="frozen_chemberta_mlp", split="outer_test")

    prediction_table = pd.DataFrame(predictions, columns=PREDICTION_COLUMNS).sort_values(["repeat", "outer_fold", "seed", "model", "split", "source_row"])
    expected = 2 * 5 * len(seeds) * 2 * len(data)  # each outer setting partitions all records into OOF dev and outer test
    if len(prediction_table) != expected or prediction_table.probability.isna().any() or not prediction_table.probability.between(0, 1).all():
        raise RuntimeError("Prediction table failed its completeness or probability-range contract.")
    prediction_path = ROOT / config["outputs"]["prediction_csv"]
    hyper_path = ROOT / config["outputs"]["hyperparameter_csv"]
    metadata_path = ROOT / config["outputs"]["run_metadata_json"]
    for path in (prediction_path, hyper_path, metadata_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    prediction_table.to_csv(prediction_path, index=False)
    pd.DataFrame(search_rows).to_csv(hyper_path, index=False)
    metadata = {
        "phase": "phase3_base_models", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "standardized_input": str(data_path.relative_to(ROOT)), "standardized_input_sha256": sha256_file(data_path),
        "config": str(config_path.relative_to(ROOT)), "config_sha256": sha256_file(config_path),
        "training_seeds": seeds, "selection_seed": selection_seed, "embedding": embedding_metadata,
        "predictions": str(prediction_path.relative_to(ROOT)), "prediction_rows": int(len(prediction_table)),
        "oof_policy": "Each OOF probability is generated by a model fit without that sample; gate training must consume only split=oof_development.",
        "outer_test_policy": "Outer test labels are excluded from hyperparameter selection, scaling, and model fitting.",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return {"predictions": prediction_path, "hyperparameters": hyper_path, "metadata": metadata_path}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs_v2/phase3_base_models.yaml")
    args = parser.parse_args()
    for name, path in run(args.config.resolve()).items():
        print(f"{name}: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
