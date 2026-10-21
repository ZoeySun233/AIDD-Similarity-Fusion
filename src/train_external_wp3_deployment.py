"""Train and freeze a new BACE-only deployment model; never read external labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from fingerprints import MorganConfig, morgan_fingerprints
from fusion_v2 import entropy, fit_shrunk_gate, fixed_weight, sigmoid
from train_base_models_v2 import fit_mlp


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def platt_fit(y: np.ndarray, raw: np.ndarray, spec: dict) -> LogisticRegression:
    clip = float(spec["probability_clip"])
    value = np.clip(raw, clip, 1 - clip)
    logit = np.log(value / (1 - value))
    return LogisticRegression(
        C=float(spec["C"]), solver=spec["solver"], max_iter=int(spec["max_iter"])
    ).fit(logit.reshape(-1, 1), y)


def platt_apply(model: LogisticRegression, raw: np.ndarray, clip: float) -> np.ndarray:
    value = np.clip(raw, clip, 1 - clip)
    return model.predict_proba(np.log(value / (1 - value)).reshape(-1, 1))[:, 1]


def scaffold_oof_inputs(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    predictions = pd.read_csv(ROOT / config["historical_oof_input"])
    subset = predictions[(predictions.repeat == 1) & (predictions.split == "outer_test")]
    seeds = set(map(int, config["seeds"]))
    if set(subset.seed) != seeds or set(subset.model) != {"morgan_rf", "frozen_chemberta_mlp"}:
        raise ValueError("Historical repeat-1 outer-test expert predictions are incomplete.")
    counts = subset.groupby(["source_row", "model"]).size()
    if len(counts) != 2 * len(data) or not counts.eq(len(seeds)).all():
        raise ValueError("Expected three held-out seed probabilities per molecule and expert.")
    wide = subset.groupby(["source_row", "model"], as_index=False).probability.mean().pivot(
        index="source_row", columns="model", values="probability"
    ).rename(columns={"morgan_rf": "rf_raw", "frozen_chemberta_mlp": "chem_raw"}).reset_index()
    outer = pd.read_csv(ROOT / config["outer_manifest"], usecols=["source_row", "outer_fold"])
    result = data[["source_row", "CID", "Class", "standardized_smiles", "scaffold"]].merge(
        outer, on="source_row", validate="one_to_one"
    ).merge(wide, on="source_row", validate="one_to_one").sort_values("source_row").reset_index(drop=True)
    if len(result) != len(data) or result[["rf_raw", "chem_raw"]].isna().any().any():
        raise ValueError("OOF join is incomplete.")
    return result


def gate_training_inputs(frame: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    y = frame.Class.to_numpy(int)
    clip = float(config["calibration"]["probability_clip"])
    cross = {"rf": np.empty(len(frame)), "chem": np.empty(len(frame))}
    for fold in sorted(frame.outer_fold.unique()):
        held = (frame.outer_fold.to_numpy() == fold)
        for expert, raw_col in (("rf", "rf_raw"), ("chem", "chem_raw")):
            calibrator = platt_fit(y[~held], frame.loc[~held, raw_col].to_numpy(float), config["calibration"])
            cross[expert][held] = platt_apply(calibrator, frame.loc[held, raw_col].to_numpy(float), clip)
    frame = frame.copy()
    frame["rf_calibrated_crossfit"] = cross["rf"]
    frame["chem_calibrated_crossfit"] = cross["chem"]
    frame["entropy_difference"] = entropy(cross["chem"]) - entropy(cross["rf"])
    rf_config = config["rf"]
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=int(rf_config["radius"]), fpSize=int(rf_config["bits"]),
        includeChirality=bool(rf_config["include_chirality"]),
    )
    fingerprints = [generator.GetFingerprint(Chem.MolFromSmiles(s)) for s in frame.standardized_smiles]
    similarity = np.empty(len(frame), dtype=float)
    for fold in sorted(frame.outer_fold.unique()):
        held = np.flatnonzero(frame.outer_fold.to_numpy() == fold)
        train = np.flatnonzero(frame.outer_fold.to_numpy() != fold)
        train_fps = [fingerprints[index] for index in train]
        for index in held:
            similarity[index] = max(DataStructs.BulkTanimotoSimilarity(fingerprints[index], train_fps))
    frame["max_similarity"] = similarity
    calibrators = {
        expert: platt_fit(y, frame[raw_col].to_numpy(float), config["calibration"])
        for expert, raw_col in (("rf", "rf_raw"), ("chem", "chem_raw"))
    }
    return frame, calibrators


def fit_fusion(frame: pd.DataFrame, config: dict) -> dict:
    rf = frame.rf_calibrated_crossfit.to_numpy(float)
    chem = frame.chem_calibrated_crossfit.to_numpy(float)
    alpha = fixed_weight(frame.Class.to_numpy(int), rf, chem)
    columns = list(config["fusion"]["features"])
    mean = frame[columns].mean().to_numpy(dtype=float, copy=True)
    scale = frame[columns].std(ddof=0).to_numpy(dtype=float, copy=True)
    scale[scale < 1e-8] = 1
    X = np.column_stack([np.ones(len(frame)), (frame[columns].to_numpy(float) - mean) / scale])
    cfg = config["fusion"]
    theta = fit_shrunk_gate(
        frame.Class.to_numpy(int), rf, chem, X, alpha,
        float(cfg["beta_l2"]), float(cfg["lambda_l2"]), int(cfg["optimizer_maxiter"]),
    )
    return {
        "c1_weight_rf": float(alpha),
        "d2_beta": theta[:-1].tolist(),
        "d2_lambda": float(sigmoid(theta[-1])),
        "feature_columns": columns,
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
    }


def run(config_path: Path) -> Path:
    config = yaml.safe_load(config_path.read_text())
    bundle = ROOT / config["bundle_directory"]
    manifest_path = bundle / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError("Frozen deployment manifest already exists; refusing to overwrite it.")
    # No external CSV is parsed here. Only its byte hash enters the later lock.
    cohort_lock = json.loads((ROOT / config["cohort_lock"]).read_text())
    cohort_path = ROOT / config["cohort_input"]
    if cohort_lock["files"][config["cohort_input"]] != sha256(cohort_path):
        raise ValueError("WP2 cohort changed after its eligibility lock.")
    data = pd.read_csv(ROOT / config["training_input"], keep_default_na=False).sort_values("source_row")
    if len(data) != 1513 or data.source_row.duplicated().any() or set(data.Class) != {0, 1}:
        raise ValueError("Original BACE training table failed its identity check.")
    oof = scaffold_oof_inputs(data, config)
    gate_frame, calibrators = gate_training_inputs(oof, config)
    fusion = fit_fusion(gate_frame, config)
    bundle.mkdir(parents=True, exist_ok=True)
    gate_path = bundle / "gate_training_oof.csv"
    gate_frame.to_csv(gate_path, index=False)
    fit_path = bundle / "fusion.json"
    fit_path.write_text(json.dumps(fusion, indent=2) + "\n")
    for expert, model in calibrators.items():
        joblib.dump(model, bundle / f"{expert}_platt.joblib")

    rf_cfg = config["rf"]
    X_rf = morgan_fingerprints(
        data.standardized_smiles, MorganConfig(
            radius=int(rf_cfg["radius"]), n_bits=int(rf_cfg["bits"]),
            include_chirality=bool(rf_cfg["include_chirality"]),
        ),
    )
    cache = np.load(ROOT / "data/processed_v2/features/chemberta_standardized_embeddings.npz")
    if not np.array_equal(cache["source_row"], data.source_row.to_numpy()) or not np.array_equal(
        cache["smiles"], data.standardized_smiles.astype(str).to_numpy()
    ):
        raise ValueError("Frozen ChemBERTa training embedding cache does not match training input.")
    X_chem = cache["X"].astype(np.float32)
    y = data.Class.to_numpy(int)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    for seed in map(int, config["seeds"]):
        rf = RandomForestClassifier(
            n_estimators=int(rf_cfg["n_estimators"]), class_weight=rf_cfg["class_weight"],
            max_depth=int(rf_cfg["max_depth"]), min_samples_leaf=int(rf_cfg["min_samples_leaf"]),
            max_features=rf_cfg["max_features"], random_state=seed, n_jobs=4,
        ).fit(X_rf, y)
        joblib.dump(rf, bundle / f"rf_seed_{seed}.joblib")
        scaler = StandardScaler().fit(X_chem)
        joblib.dump(scaler, bundle / f"chem_scaler_seed_{seed}.joblib")
        spec = config["chemberta"]
        model, _ = fit_mlp(
            scaler.transform(X_chem).astype(np.float32), y,
            {key: spec[key] for key in ("hidden_dim", "dropout", "learning_rate", "weight_decay")},
            seed, int(spec["epochs"]),
        )
        torch.save(model.state_dict(), bundle / f"chem_head_seed_{seed}.pt")

    tracked_inputs = [
        config["training_input"], config["historical_oof_input"], config["outer_manifest"],
        config["cohort_input"], config["cohort_lock"],
        "data/processed_v2/features/chemberta_standardized_embeddings.npz",
        str(config_path.relative_to(ROOT)),
        "src/train_external_wp3_deployment.py", "src/evaluate_external_wp3_deployment.py",
        "src/chemberta.py", "src/fingerprints.py", "src/fusion_v2.py", "src/train_base_models_v2.py",
        "report_v2/external_wp3_amended_protocol.md",
    ]
    artifacts = sorted(path for path in bundle.iterdir() if path.is_file() and path.name != "manifest.json")
    hashes = {relative: sha256(ROOT / relative) for relative in tracked_inputs}
    hashes.update({str(path.relative_to(ROOT)): sha256(path) for path in artifacts})
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest = {
        "status": "FROZEN_BEFORE_EXTERNAL_LABEL_SCORING",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": revision,
        "scope": "New full-original-BACE deployment model; not recovered historical fold checkpoints",
        "external_labels_loaded": False,
        "external_predictions_generated": False,
        "config": str(config_path.relative_to(ROOT)),
        "training_rows": len(data),
        "training_seeds": list(map(int, config["seeds"])),
        "primary_assay": config["primary_assay"],
        "secondary_assay": config["secondary_assay"],
        "encoder_revision": config["chemberta"]["revision"],
        "fusion": fusion,
        "artifact_sha256": hashes,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs_v2/external_wp3_deployment.yaml")
    args = parser.parse_args()
    path = run(args.config.resolve())
    print(f"Frozen deployment bundle: {path}")


if __name__ == "__main__":
    main()
