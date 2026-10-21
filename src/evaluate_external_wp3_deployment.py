"""Verify the frozen bundle, then score the external cohort exactly once."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

import chemberta
from fingerprints import MorganConfig, morgan_fingerprints
from fusion_v2 import entropy, sigmoid
from train_base_models_v2 import predict_mlp
from train_external_wp3_deployment import platt_apply


ROOT = Path(__file__).resolve().parents[1]
METHODS = {
    "RF": "rf_probability", "ChemBERTa": "chem_probability",
    "C1": "c1_probability", "D2": "d2_probability",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_freeze(config_path: Path, output: Path) -> tuple[dict, dict]:
    config = yaml.safe_load(config_path.read_text())
    manifest_path = ROOT / config["bundle_directory"] / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Frozen deployment manifest is missing.")
    if output.exists():
        raise FileExistsError("External WP3 output already exists; one-time scoring refuses to overwrite it.")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "FROZEN_BEFORE_EXTERNAL_LABEL_SCORING":
        raise ValueError("Deployment manifest has not been frozen before scoring.")
    if manifest.get("config") != str(config_path.relative_to(ROOT)):
        raise ValueError("Scoring config differs from frozen deployment config.")
    for relative, expected in manifest["artifact_sha256"].items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT.resolve()) or not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Frozen artifact changed or unavailable: {relative}")
    return config, manifest


def predict_components(cohort: pd.DataFrame, config: dict, manifest: dict) -> pd.DataFrame:
    bundle = ROOT / config["bundle_directory"]
    training = pd.read_csv(ROOT / config["training_input"], usecols=["standardized_smiles"])
    rf_cfg = config["rf"]
    smiles = cohort.standardized_smiles.astype(str)
    X_rf = morgan_fingerprints(
        smiles, MorganConfig(
            radius=int(rf_cfg["radius"]), n_bits=int(rf_cfg["bits"]),
            include_chirality=bool(rf_cfg["include_chirality"]),
        ),
    )
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=int(rf_cfg["radius"]), fpSize=int(rf_cfg["bits"]),
        includeChirality=bool(rf_cfg["include_chirality"]),
    )
    train_fps = [generator.GetFingerprint(Chem.MolFromSmiles(value)) for value in training.standardized_smiles]
    similarity = np.array([
        max(DataStructs.BulkTanimotoSimilarity(generator.GetFingerprint(Chem.MolFromSmiles(value)), train_fps))
        for value in smiles
    ], dtype=float)
    rf_raw = np.mean([
        joblib.load(bundle / f"rf_seed_{seed}.joblib").predict_proba(X_rf)[:, 1]
        for seed in config["seeds"]
    ], axis=0)
    # The local, revision-locked encoder is loaded only after all hashes pass.
    tokenizer, encoder, device = chemberta.load_frozen_encoder(ROOT / "models/huggingface", local_files_only=True)
    if getattr(encoder.config, "_commit_hash", None) != manifest["encoder_revision"]:
        raise ValueError("Frozen encoder revision differs from the deployment manifest.")
    X_chem, metadata = chemberta.encode_with_frozen_encoder(
        smiles, tokenizer, encoder, device, batch_size=int(config["chemberta"]["batch_size"]),
        max_length=int(config["chemberta"]["max_length"]),
    )
    if metadata["truncated_molecules"]:
        raise ValueError("External ChemBERTa inputs would be truncated; stop before scoring.")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    raw_heads = []
    for seed in config["seeds"]:
        scaler = joblib.load(bundle / f"chem_scaler_seed_{seed}.joblib")
        model = chemberta.ActivityMLP(X_chem.shape[1], int(config["chemberta"]["hidden_dim"]), float(config["chemberta"]["dropout"]))
        model.load_state_dict(torch.load(bundle / f"chem_head_seed_{seed}.pt", map_location="cpu", weights_only=True))
        raw_heads.append(predict_mlp(model, scaler.transform(X_chem).astype(np.float32)))
    chem_raw = np.mean(raw_heads, axis=0)
    clip = float(config["calibration"]["probability_clip"])
    rf = platt_apply(joblib.load(bundle / "rf_platt.joblib"), rf_raw, clip)
    chem = platt_apply(joblib.load(bundle / "chem_platt.joblib"), chem_raw, clip)
    fusion = manifest["fusion"]
    C1 = fusion["c1_weight_rf"] * rf + (1 - fusion["c1_weight_rf"]) * chem
    feature_map = {"max_similarity": similarity, "entropy_difference": entropy(chem) - entropy(rf)}
    features = np.column_stack([feature_map[name] for name in fusion["feature_columns"]])
    scaled = (features - np.asarray(fusion["feature_mean"])) / np.asarray(fusion["feature_scale"])
    dynamic = sigmoid(np.column_stack([np.ones(len(cohort)), scaled]) @ np.asarray(fusion["d2_beta"]))
    alpha = (1 - fusion["d2_lambda"]) * fusion["c1_weight_rf"] + fusion["d2_lambda"] * dynamic
    D2 = alpha * rf + (1 - alpha) * chem
    scored = cohort.copy()
    scored["max_similarity"] = similarity
    scored["rf_probability"] = rf
    scored["chem_probability"] = chem
    scored["c1_probability"] = C1
    scored["d2_probability"] = D2
    scored["d2_rf_weight"] = alpha
    if scored[list(METHODS.values())].isna().any().any() or not scored[list(METHODS.values())].apply(lambda col: col.between(0, 1).all()).all():
        raise ValueError("A frozen model produced an invalid probability.")
    return scored


def metric_row(frame: pd.DataFrame, probability: np.ndarray, config: dict) -> dict:
    y = frame.label.to_numpy(int)
    n, positives = len(y), int(y.sum())
    result = {"n": n, "positives": positives, "prevalence": float(positives / n)}
    if n == 0:
        return result
    result["brier"] = float(brier_score_loss(y, probability))
    result["log_loss"] = float(log_loss(y, np.clip(probability, float(config["evaluation"]["log_loss_clip"]), 1 - float(config["evaluation"]["log_loss_clip"])), labels=[0, 1]))
    result["ap"] = float(average_precision_score(y, probability)) if 0 < positives < n else None
    result["roc_auc"] = float(roc_auc_score(y, probability)) if 0 < positives < n else None
    rank_id = frame.assay_chembl_id.astype(str) + "|" + frame.standardized_inchikey.astype(str)
    order = np.lexsort((rank_id.to_numpy(), -probability))
    for fraction in config["evaluation"]["top_fractions"]:
        fraction = float(fraction)
        k = max(1, math.ceil(n * fraction))
        precision = float(y[order[:k]].mean())
        suffix = f"top_{int(round(100 * fraction))}pct"
        result[f"k_{suffix}"] = k
        result[f"precision_{suffix}"] = precision
        result[f"ef_{suffix}"] = float(precision / result["prevalence"]) if positives else None
    return result


def paired_bootstrap(frame: pd.DataFrame, config: dict) -> dict:
    specification = config["evaluation"]
    y = frame.label.to_numpy(int)
    c1 = frame.c1_probability.to_numpy(float)
    d2 = frame.d2_probability.to_numpy(float)
    groups = [np.flatnonzero(frame.scaffold.to_numpy() == scaffold) for scaffold in sorted(frame.scaffold.unique())]
    rng = np.random.default_rng(int(specification["bootstrap_seed"]))
    target, limit = int(specification["bootstrap_replicates"]), int(specification["bootstrap_max_attempts"])
    deltas, rejected, attempts = [], 0, 0
    while len(deltas) < target and attempts < limit:
        attempts += 1
        sampled = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([groups[index] for index in sampled])
        if len(np.unique(y[indices])) < 2:
            rejected += 1
            continue
        deltas.append(float(average_precision_score(y[indices], d2[indices]) - average_precision_score(y[indices], c1[indices])))
    if len(deltas) < int(specification["bootstrap_min_valid"]):
        return {"valid": len(deltas), "rejected_single_class": rejected, "attempts": attempts, "n_scaffolds": len(groups), "ci_lower": None, "ci_upper": None, "informative": False}
    low, high = np.quantile(deltas, [0.025, 0.975])
    informative = len(groups) >= int(specification["insufficient_cluster_threshold"]) and high - low <= float(specification["wide_ci_threshold"])
    return {"valid": len(deltas), "rejected_single_class": rejected, "attempts": attempts, "n_scaffolds": len(groups), "ci_lower": float(low), "ci_upper": float(high), "ci_width": float(high - low), "informative": bool(informative)}


def reliability_rows(frame: pd.DataFrame) -> list[dict]:
    rows = []
    for method, column in METHODS.items():
        p = frame[column].to_numpy(float)
        bins = np.minimum(np.floor(p * 10).astype(int), 9)
        for bin_index in range(10):
            selected = frame.iloc[np.flatnonzero(bins == bin_index)]
            rows.append({
                "assay": frame.assay_chembl_id.iloc[0], "method": method, "bin": bin_index,
                "n": len(selected), "positives": int(selected.label.sum()),
                "mean_probability": float(selected[column].mean()) if len(selected) else None,
                "observed_fraction": float(selected.label.mean()) if len(selected) else None,
            })
    return rows


def plot_reliability(table: pd.DataFrame, path: Path) -> None:
    assays = list(table.assay.unique())
    fig, axes = plt.subplots(1, len(assays), figsize=(5 * len(assays), 4), squeeze=False)
    for ax, assay in zip(axes[0], assays):
        subset = table[(table.assay == assay) & (table.n > 0)]
        for method in METHODS:
            item = subset[subset.method == method]
            ax.plot(item.mean_probability, item.observed_fraction, marker="o", label=method)
        ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability", ylabel="Observed active fraction", title=assay)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def evaluate(scored: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    primary_assay = config["primary_assay"]
    metric_rows, strata_rows, reliability = [], [], []
    for assay in (primary_assay, config["secondary_assay"]):
        frame = scored[scored.assay_chembl_id == assay].copy().reset_index(drop=True)
        for method, column in METHODS.items():
            metric_rows.append({"assay": assay, "role": "primary" if assay == primary_assay else "secondary_sensitivity", "method": method, **metric_row(frame, frame[column].to_numpy(float), config)})
        reliability.extend(reliability_rows(frame))
        edges = config["evaluation"]["similarity_edges"]
        for low, high in zip(edges[:-1], edges[1:]):
            part = frame[(frame.max_similarity >= low) & (frame.max_similarity < high)]
            if len(part) == 0:
                continue
            for method, column in METHODS.items():
                strata_rows.append({"assay": assay, "stratum": f"[{low},{high})", "analysis_role": "exploratory", "method": method, **metric_row(part, part[column].to_numpy(float), config)})
    primary = scored[scored.assay_chembl_id == primary_assay].copy().reset_index(drop=True)
    secondary = scored[scored.assay_chembl_id == config["secondary_assay"]].copy().reset_index(drop=True)
    bootstrap = paired_bootstrap(primary, config)
    def difference(frame: pd.DataFrame) -> float:
        return float(average_precision_score(frame.label, frame.d2_probability) - average_precision_score(frame.label, frame.c1_probability))
    delta = difference(primary)
    second_delta = difference(secondary)
    loo = []
    for scaffold in sorted(primary.scaffold.unique()):
        remaining = primary[primary.scaffold != scaffold]
        loo.append({"scaffold": scaffold, "removed_n": int((primary.scaffold == scaffold).sum()), "delta_ap": difference(remaining)})
    worst_loo = min(row["delta_ap"] for row in loo)
    brier_delta = float(brier_score_loss(primary.label, primary.d2_probability) - brier_score_loss(primary.label, primary.c1_probability))
    spec = config["evaluation"]
    source_robust = second_delta > 0 and second_delta >= float(spec["min_scaffold_retained_gain_fraction"]) * delta
    scaffold_robust = worst_loo > 0 and worst_loo >= float(spec["min_scaffold_retained_gain_fraction"]) * delta
    if not bootstrap["informative"]:
        decision = "inconclusive_interval"
    elif (delta >= float(spec["success_delta_ap"]) and bootstrap["ci_lower"] > 0 and
          brier_delta <= float(spec["success_max_delta_brier"]) and scaffold_robust and source_robust):
        decision = "supportive_with_single_primary_source_caveat"
    else:
        decision = "gate_unsupported_retain_C1"
    summary = {
        "decision": decision,
        "scope": "New frozen full-original-BACE deployment model, not recovered historical checkpoints",
        "primary_assay": primary_assay,
        "secondary_assay": config["secondary_assay"],
        "primary_delta_ap_d2_minus_c1": delta,
        "primary_delta_brier_d2_minus_c1": brier_delta,
        "secondary_delta_ap_d2_minus_c1": second_delta,
        "bootstrap": bootstrap,
        "primary_leave_one_scaffold_out_min_delta_ap": worst_loo,
        "primary_leave_one_scaffold_out_n": len(loo),
        "scaffold_robust": scaffold_robust,
        "secondary_source_concordant": source_robust,
        "primary_source_count": 1,
        "source_sensitivity_note": "Primary is one patent/assay source. Secondary older-source cohort is analyzed separately and is not pooled or used for model selection.",
        "leave_one_scaffold_out": loo,
    }
    return pd.DataFrame(metric_rows), pd.DataFrame(strata_rows), pd.DataFrame(reliability), summary


def run(config_path: Path) -> Path:
    preliminary = yaml.safe_load(config_path.read_text())
    output = ROOT / preliminary["output_directory"]
    config, manifest = verify_freeze(config_path, output)
    # This is the first point at which external labels enter evaluation code.
    cohort = pd.read_csv(ROOT / config["cohort_input"])
    if set(cohort.assay_chembl_id) != {config["primary_assay"], config["secondary_assay"]}:
        raise ValueError("External assay identities changed.")
    if set(cohort.label) != {0, 1} or cohort.duplicated(["assay_chembl_id", "standardized_inchikey"]).any():
        raise ValueError("External cohort does not meet the locked eligibility contract.")
    scored = predict_components(cohort, config, manifest)
    metrics, strata, calibration, summary = evaluate(scored, config)
    output.mkdir(parents=True, exist_ok=False)
    scored.to_csv(output / "paired_predictions.csv", index=False)
    metrics.to_csv(output / "metrics.csv", index=False)
    strata.to_csv(output / "similarity_strata_exploratory.csv", index=False)
    calibration.to_csv(output / "reliability_bins.csv", index=False)
    plot_reliability(calibration, output / "reliability.svg")
    summary["frozen_manifest_sha256"] = sha256(ROOT / config["bundle_directory"] / "manifest.json")
    summary["external_cohort_sha256"] = sha256(ROOT / config["cohort_input"])
    (output / "evaluation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs_v2/external_wp3_deployment.yaml")
    args = parser.parse_args()
    path = run(args.config.resolve())
    print(f"One-time external evaluation written to {path}")


if __name__ == "__main__":
    main()
