"""Build model-independent nested scaffold CV manifests for protocol v2."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator
from rdkit.ML.Cluster import Butina


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_source(data: pd.DataFrame, config: dict[str, Any]) -> None:
    columns = config["dataset"]
    required = [columns[key] for key in ("source_row_column", "label_column", "scaffold_column", "structure_group_column")]
    missing = set(required).difference(data.columns)
    if missing:
        raise ValueError(f"Standardized data is missing: {sorted(missing)}")
    if data[columns["source_row_column"]].duplicated().any():
        raise ValueError("source_row must be unique.")
    if data[columns["scaffold_column"]].isna().any():
        raise ValueError("Every v2 record must have a scaffold before splitting.")
    if data[columns["structure_group_column"]].isna().any():
        raise ValueError("Every v2 record must have a structure-group identifier.")
    if not set(data[columns["label_column"]].unique()).issubset({0, 1}):
        raise ValueError("This protocol requires binary Class labels.")


def _scaffold_groups(data: pd.DataFrame, scaffold_column: str, label_column: str) -> list[dict[str, Any]]:
    groups = []
    for scaffold, frame in data.groupby(scaffold_column, sort=True, dropna=False):
        groups.append({
            "key": str(scaffold), "indices": frame.index.to_numpy(), "size": len(frame),
            "positive": int(frame[label_column].sum()),
        })
    return groups


def balanced_group_folds(groups: list[dict[str, Any]], n_folds: int, seed: int) -> np.ndarray:
    """Assign whole groups to folds while balancing size and active-class count."""
    rng = np.random.default_rng(seed)
    original_groups = list(groups)
    permutation = {group["key"]: float(rng.random()) for group in original_groups}
    allocation_order = sorted(original_groups, key=lambda group: (-group["size"], -abs(group["positive"] / group["size"] - 0.5), permutation[group["key"]], group["key"]))
    total_rows = sum(group["size"] for group in original_groups)
    total_positive = sum(group["positive"] for group in original_groups)
    target_rows, target_positive = total_rows / n_folds, total_positive / n_folds
    rows = np.zeros(n_folds, dtype=float)
    positive = np.zeros(n_folds, dtype=float)
    group_counts = np.zeros(n_folds, dtype=int)
    assignments: dict[str, int] = {}
    for group in allocation_order:
        scores: list[float] = []
        for fold in range(n_folds):
            cand_rows, cand_positive = rows.copy(), positive.copy()
            cand_rows[fold] += group["size"]
            cand_positive[fold] += group["positive"]
            row_loss = np.square((cand_rows - target_rows) / max(target_rows, 1)).sum()
            positive_loss = np.square((cand_positive - target_positive) / max(target_positive, 1)).sum()
            empty_penalty = 0.05 if group_counts[fold] == 0 else 0.0
            scores.append(row_loss + 2.0 * positive_loss - empty_penalty)
        chosen = int(np.argmin(scores))
        assignments[group["key"]] = chosen + 1
        rows[chosen] += group["size"]
        positive[chosen] += group["positive"]
        group_counts[chosen] += 1
    return np.array([assignments[group["key"]] for group in original_groups])


def assign_scaffold_folds(data: pd.DataFrame, label_column: str, scaffold_column: str, folds: int, seed: int) -> pd.Series:
    groups = _scaffold_groups(data, scaffold_column, label_column)
    assignments = balanced_group_folds(groups, folds, seed)
    lookup = {group["key"]: int(fold) for group, fold in zip(groups, assignments)}
    return data[scaffold_column].astype(str).map(lookup).astype(int)


def scaffold_overlap_count(data: pd.DataFrame, fold_column: str, scaffold_column: str) -> int:
    membership = data.groupby(scaffold_column)[fold_column].nunique()
    return int((membership > 1).sum())


def structure_group_overlap_count(data: pd.DataFrame, fold_column: str, group_column: str) -> int:
    membership = data.groupby(group_column)[fold_column].nunique()
    return int((membership > 1).sum())


def fold_audit_rows(
    data: pd.DataFrame, *, family: str, repeat: int | None, outer_fold: int | None,
    fold_column: str, fold_count: int, label_column: str, scaffold_column: str,
    group_column: str, model_selection_visibility: str, role: str,
) -> list[dict[str, Any]]:
    rows = []
    scaffold_overlap = scaffold_overlap_count(data, fold_column, scaffold_column)
    structure_overlap = structure_group_overlap_count(data, fold_column, group_column)
    for fold in range(1, fold_count + 1):
        subset = data[data[fold_column] == fold]
        scaffold_sizes = subset.groupby(scaffold_column).size()
        rows.append({
            "split_family": family, "repeat": repeat, "outer_test_fold": outer_fold,
            "fold": fold, "fold_role": role, "n_compounds": len(subset),
            "positive_count": int(subset[label_column].sum()),
            "positive_fraction": float(subset[label_column].mean()),
            "n_scaffolds": int(subset[scaffold_column].nunique()),
            "n_structure_groups": int(subset[group_column].nunique()),
            "max_scaffold_count": int(scaffold_sizes.max()),
            "max_scaffold_fraction": float(scaffold_sizes.max() / len(subset)),
            "scaffold_groups_crossing_folds": scaffold_overlap,
            "structure_groups_crossing_folds": structure_overlap,
            "model_selection_visibility": model_selection_visibility,
        })
    return rows


def butina_cluster_ids(data: pd.DataFrame, radius: int, fp_size: int, distance_threshold: float) -> pd.Series:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size, includeChirality=True)
    fingerprints = [generator.GetFingerprint(Chem.MolFromSmiles(smiles)) for smiles in data["standardized_smiles"]]
    distances: list[float] = []
    for index in range(1, len(fingerprints)):
        distances.extend(1.0 - similarity for similarity in DataStructs.BulkTanimotoSimilarity(fingerprints[index], fingerprints[:index]))
    clusters = Butina.ClusterData(distances, len(fingerprints), distance_threshold, isDistData=True, reordering=True)
    cluster_ids = np.empty(len(data), dtype=int)
    for cluster_id, members in enumerate(clusters, start=1):
        cluster_ids[list(members)] = cluster_id
    return pd.Series(cluster_ids, index=data.index, name="butina_cluster_id")


def run(config_path: Path) -> dict[str, Path]:
    with config_path.open() as handle:
        config = yaml.safe_load(handle)
    source_path = ROOT / config["dataset"]["standardized_csv"]
    data = pd.read_csv(source_path, keep_default_na=False)
    _validate_source(data, config)
    columns = config["dataset"]
    label, scaffold, group = columns["label_column"], columns["scaffold_column"], columns["structure_group_column"]
    output_dir = ROOT / config["outputs"]["split_directory"]
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_rows: list[dict[str, Any]] = []
    outer_paths: dict[str, Path] = {}
    inner_paths: list[Path] = []
    outer = config["outer_cv"]
    for repeat, outer_seed in enumerate(outer["seeds"], start=1):
        manifest = data[["source_row", "CID", label, scaffold, group]].copy()
        manifest["outer_repeat"] = repeat
        manifest["outer_fold"] = assign_scaffold_folds(data, label, scaffold, int(outer["folds"]), int(outer_seed))
        manifest["outer_role_if_selected"] = "outer_test_when_outer_fold_matches"
        manifest["training_seeds"] = ",".join(map(str, config["training_seeds"]))
        path = output_dir / f"outer_repeat_{repeat}.csv"
        manifest.to_csv(path, index=False)
        outer_paths[f"outer_repeat_{repeat}"] = path
        audit_rows.extend(fold_audit_rows(manifest, family="primary_outer_scaffold", repeat=repeat, outer_fold=None, fold_column="outer_fold", fold_count=int(outer["folds"]), label_column=label, scaffold_column=scaffold, group_column=group, model_selection_visibility="outer test fold is hidden during model selection", role="outer_test"))
        for outer_fold in range(1, int(outer["folds"]) + 1):
            development = manifest[manifest["outer_fold"] != outer_fold].copy()
            development["inner_fold"] = assign_scaffold_folds(development, label, scaffold, int(config["inner_cv"]["folds"]), int(outer_seed) * 100 + outer_fold)
            development["outer_repeat"] = repeat
            development["outer_test_fold"] = outer_fold
            development["inner_role_if_selected"] = "inner_validation_when_inner_fold_matches"
            inner_path = output_dir / f"inner_repeat_{repeat}_outer_fold_{outer_fold}.csv"
            development.to_csv(inner_path, index=False)
            inner_paths.append(inner_path)
            audit_rows.extend(fold_audit_rows(development, family="primary_inner_scaffold", repeat=repeat, outer_fold=outer_fold, fold_column="inner_fold", fold_count=int(config["inner_cv"]["folds"]), label_column=label, scaffold_column=scaffold, group_column=group, model_selection_visibility="inner validation only; corresponding outer test remains hidden", role="inner_validation"))

    sensitivity = config["sensitivity_analyses"]
    if sensitivity["canonical_deepchem_scaffold_cv"]["enabled"]:
        spec = sensitivity["canonical_deepchem_scaffold_cv"]
        manifest = data[["source_row", "CID", label, scaffold, group]].copy()
        manifest["fold"] = assign_scaffold_folds(data, label, scaffold, int(spec["folds"]), int(spec["seed"]))
        path = output_dir / "sensitivity_canonical_deepchem_scaffold.csv"
        manifest.to_csv(path, index=False)
        outer_paths["sensitivity_canonical_deepchem_scaffold"] = path
        audit_rows.extend(fold_audit_rows(manifest, family="sensitivity_canonical_deepchem_scaffold", repeat=None, outer_fold=None, fold_column="fold", fold_count=int(spec["folds"]), label_column=label, scaffold_column=scaffold, group_column=group, model_selection_visibility="sensitivity analysis only", role="sensitivity_test"))
    if sensitivity["butina_cluster_cv"]["enabled"]:
        spec = sensitivity["butina_cluster_cv"]
        manifest = data[["source_row", "CID", label, scaffold, group, "standardized_smiles"]].copy()
        manifest["butina_cluster_id"] = butina_cluster_ids(data, int(spec["morgan_radius"]), int(spec["fingerprint_size"]), float(spec["distance_threshold"]))
        cluster_groups = []
        for cluster_id, frame in manifest.groupby("butina_cluster_id", sort=True):
            cluster_groups.append({"key": str(cluster_id), "indices": frame.index.to_numpy(), "size": len(frame), "positive": int(frame[label].sum())})
        assignments = balanced_group_folds(cluster_groups, int(spec["folds"]), int(spec["seed"]))
        lookup = {group_info["key"]: int(fold) for group_info, fold in zip(cluster_groups, assignments)}
        manifest["fold"] = manifest["butina_cluster_id"].astype(str).map(lookup).astype(int)
        manifest = manifest.drop(columns="standardized_smiles")
        path = output_dir / "sensitivity_butina_cluster.csv"
        manifest.to_csv(path, index=False)
        outer_paths["sensitivity_butina_cluster"] = path
        audit_rows.extend(fold_audit_rows(manifest, family="sensitivity_butina_cluster", repeat=None, outer_fold=None, fold_column="fold", fold_count=int(spec["folds"]), label_column=label, scaffold_column=scaffold, group_column=group, model_selection_visibility="sensitivity analysis only; not scaffold-disjoint by design", role="sensitivity_test"))

    audit_path = ROOT / config["outputs"]["split_audit_csv"]
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(audit_path, index=False)
    metadata_path = ROOT / config["outputs"]["metadata_json"]
    metadata = {
        "phase": "phase2_nested_scaffold_cv", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(source_path.relative_to(ROOT)), "input_sha256": sha256_file(source_path),
        "config": str(config_path.relative_to(ROOT)), "config_sha256": sha256_file(config_path),
        "rdkit_version": rdBase.rdkitVersion, "training_seeds": config["training_seeds"],
        "outer_repeats": outer["repeats"], "outer_folds": outer["folds"], "inner_folds": config["inner_cv"]["folds"],
        "outer_manifests": {key: str(path.relative_to(ROOT)) for key, path in outer_paths.items()},
        "inner_manifests": [str(path.relative_to(ROOT)) for path in inner_paths],
        "split_audit": str(audit_path.relative_to(ROOT)),
        "model_specific_assignments": False,
        "historical_test_used_for_selection": False,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return {**outer_paths, "split_audit": audit_path, "metadata": metadata_path}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs_v2/phase2_nested_scaffold_cv.yaml")
    args = parser.parse_args()
    for name, path in run(args.config.resolve()).items():
        print(f"{name}: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
