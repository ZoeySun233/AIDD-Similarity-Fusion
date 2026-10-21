"""Traceable version-2 molecular standardization and data-quality audit.

This module is deliberately audit-first: it preserves every raw BACE record,
does not alter labels, and records possible exclusions or duplicates as review
flags.  It does not make or tune a version-2 model split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    """Return a content hash without changing the source file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_smiles(mol: Chem.Mol, *, isomeric: bool = True) -> str:
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric)


def safe_inchi_key(mol: Chem.Mol) -> str | None:
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:
        return None


def _largest_fragment(
    mol: Chem.Mol, chooser: rdMolStandardize.LargestFragmentChooser
) -> tuple[Chem.Mol, int, str, str]:
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    component_smiles = sorted(canonical_smiles(fragment) for fragment in fragments)
    largest = chooser.choose(mol)
    largest_smiles = canonical_smiles(largest)
    remaining = list(component_smiles)
    if largest_smiles in remaining:
        remaining.remove(largest_smiles)
    return largest, len(fragments), ".".join(component_smiles), ".".join(remaining)


def standardize_smiles(smiles: str, config: dict[str, Any]) -> dict[str, Any]:
    """Standardize one record and retain each intermediate representation."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "is_valid_smiles": False,
            "standardization_status": "invalid_smiles",
            "standardization_action_trace": "parse_failed",
        }

    std = config["standardization"]
    chooser = rdMolStandardize.LargestFragmentChooser(
        preferOrganic=bool(std["largest_fragment"]["prefer_organic"])
    )
    normalizer = rdMolStandardize.Normalizer()
    reionizer = rdMolStandardize.Reionizer()
    uncharger = rdMolStandardize.Uncharger()
    tautomer_enumerator = rdMolStandardize.TautomerEnumerator()

    raw_canonical = canonical_smiles(mol)
    largest, component_count, components, discarded = _largest_fragment(mol, chooser)
    largest_smiles = canonical_smiles(largest)
    actions: list[str] = []
    if component_count > 1:
        actions.append("largest_fragment_selected")

    current = largest
    if std["normalize_functional_groups"]:
        next_mol = normalizer.normalize(current)
        if canonical_smiles(next_mol) != canonical_smiles(current):
            actions.append("functional_groups_normalized")
        current = next_mol
    normalized_smiles = canonical_smiles(current)

    if std["reionize"]:
        next_mol = reionizer.reionize(current)
        if canonical_smiles(next_mol) != canonical_smiles(current):
            actions.append("charge_reionized")
        current = next_mol
    reionized_smiles = canonical_smiles(current)

    if std["uncharge"]:
        next_mol = uncharger.uncharge(current)
        if canonical_smiles(next_mol) != canonical_smiles(current):
            actions.append("charge_neutralized")
        current = next_mol
    uncharged_smiles = canonical_smiles(current)
    pre_tautomer_smiles = uncharged_smiles

    if std["canonicalize_tautomer"]:
        next_mol = tautomer_enumerator.Canonicalize(current)
        if canonical_smiles(next_mol) != canonical_smiles(current):
            actions.append("tautomer_canonicalized")
        current = next_mol

    standardized = canonical_smiles(
        current, isomeric=bool(std["canonical_smiles_isomeric"])
    )
    achiral = Chem.Mol(current)
    Chem.RemoveStereochemistry(achiral)
    standardized_without_stereo = canonical_smiles(achiral, isomeric=False)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=current, includeChirality=bool(std["scaffold_include_chirality"])
    )
    return {
        "is_valid_smiles": True,
        "standardization_status": "standardized",
        "raw_canonical_smiles": raw_canonical,
        "component_count": component_count,
        "component_smiles": components,
        "largest_fragment_smiles": largest_smiles,
        "discarded_component_smiles": discarded,
        "normalized_smiles": normalized_smiles,
        "reionized_smiles": reionized_smiles,
        "uncharged_smiles": uncharged_smiles,
        "pre_tautomer_smiles": pre_tautomer_smiles,
        "standardized_smiles": standardized,
        "standardized_smiles_no_stereo": standardized_without_stereo,
        "standardized_inchikey": safe_inchi_key(current),
        "scaffold": scaffold,
        "standardization_action_trace": ";".join(actions) if actions else "no_structure_change",
    }


def _duplicate_group_ids(values: pd.Series, prefix: str) -> tuple[pd.Series, dict[str, int]]:
    counts = values.value_counts(dropna=True)
    duplicates = sorted(counts[counts > 1].index.tolist())
    lookup = {value: f"{prefix}_{index:04d}" for index, value in enumerate(duplicates, 1)}
    return values.map(lookup).astype("string"), {lookup[value]: int(counts[value]) for value in duplicates}


def annotate_review_flags(data: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Add non-destructive duplicate and label-review flags to source rows."""
    result = data.copy()
    valid = result["is_valid_smiles"].fillna(False)
    # This is the indivisible grouping key for the future v2 splitter.  Its
    # presence means that standardized duplicates can be kept in the same v2
    # fold rather than discarded or allowed to leak across folds.
    result["v2_structure_group_id"] = pd.Series(pd.NA, index=result.index, dtype="string")
    result.loc[valid, "v2_structure_group_id"] = result.loc[valid, "standardized_smiles"].map(
        lambda value: "v2_structure_" + hashlib.sha256(value.encode()).hexdigest()[:16]
    )
    result["structure_duplicate_group"] = pd.Series(pd.NA, index=result.index, dtype="string")
    result["stereo_duplicate_group"] = pd.Series(pd.NA, index=result.index, dtype="string")
    result["tautomer_duplicate_group"] = pd.Series(pd.NA, index=result.index, dtype="string")

    structure_ids, _ = _duplicate_group_ids(result.loc[valid, "standardized_smiles"], "structure")
    result.loc[valid, "structure_duplicate_group"] = structure_ids
    group_class_counts = result.loc[valid].groupby("standardized_smiles")["Class"].nunique()
    result["standardized_structure_conflict"] = valid & result["standardized_smiles"].map(group_class_counts).fillna(0).gt(1)

    legacy_nunique = result.loc[valid].groupby("standardized_smiles")["legacy_v1_split"].nunique()
    result["cross_legacy_split_duplicate"] = valid & result["standardized_smiles"].map(legacy_nunique).fillna(0).gt(1)

    stereo_groups = result.loc[valid].groupby("standardized_smiles_no_stereo")["standardized_smiles"].nunique()
    stereo_candidate = valid & result["standardized_smiles_no_stereo"].map(stereo_groups).fillna(0).gt(1)
    stereo_ids, _ = _duplicate_group_ids(result.loc[stereo_candidate, "standardized_smiles_no_stereo"], "stereo")
    result.loc[stereo_candidate, "stereo_duplicate_group"] = stereo_ids
    result["potential_stereo_duplicate"] = stereo_candidate

    tautomer_groups = result.loc[valid].groupby("standardized_smiles")["pre_tautomer_smiles"].nunique()
    tautomer_candidate = valid & result["standardized_smiles"].map(tautomer_groups).fillna(0).gt(1)
    tautomer_ids, _ = _duplicate_group_ids(result.loc[tautomer_candidate, "standardized_smiles"], "tautomer")
    result.loc[tautomer_candidate, "tautomer_duplicate_group"] = tautomer_ids
    result["potential_tautomer_duplicate"] = tautomer_candidate

    threshold = float(config["audit"]["pic50_activity_threshold"])
    pic50 = pd.to_numeric(result["pIC50"], errors="coerce")
    result["class_pic50_consistency_review"] = valid & (
        ((result["Class"] == 1) & (pic50 < threshold))
        | ((result["Class"] == 0) & (pic50 >= threshold))
    )
    result["class_pic50_review_reason"] = pd.Series(pd.NA, index=result.index, dtype="string")
    result.loc[(result["Class"] == 1) & (pic50 < threshold), "class_pic50_review_reason"] = (
        f"active_label_with_pIC50_below_{threshold:g}"
    )
    result.loc[(result["Class"] == 0) & (pic50 >= threshold), "class_pic50_review_reason"] = (
        f"inactive_label_with_pIC50_at_or_above_{threshold:g}"
    )

    flag_columns = [
        "standardized_structure_conflict",
        "cross_legacy_split_duplicate",
        "potential_stereo_duplicate",
        "potential_tautomer_duplicate",
        "class_pic50_consistency_review",
    ]
    result["review_required"] = result[flag_columns].any(axis=1)
    result["deduplication_recommendation"] = "retain_without_change"
    result.loc[result["review_required"], "deduplication_recommendation"] = "retain_pending_manual_review"
    return result


def build_label_consistency_audit(data: pd.DataFrame) -> pd.DataFrame:
    """Emit one reviewable issue per source record; no labels are modified."""
    issues = [
        ("standardized_structure_conflict", "structure_duplicate_group", "same standardized structure has multiple Class labels"),
        ("cross_legacy_split_duplicate", "structure_duplicate_group", "standardized structure appears in more than one frozen v1 split"),
        ("potential_stereo_duplicate", "stereo_duplicate_group", "distinct stereochemical structures share a stereo-stripped representation"),
        ("potential_tautomer_duplicate", "tautomer_duplicate_group", "different pre-tautomer representations converge after tautomer canonicalization"),
        ("class_pic50_consistency_review", "structure_duplicate_group", "Class conflicts with the prespecified pIC50 diagnostic threshold"),
    ]
    rows: list[dict[str, Any]] = []
    for flag, group_column, detail in issues:
        for _, record in data.loc[data[flag]].iterrows():
            issue_detail = record.get("class_pic50_review_reason") if flag == "class_pic50_consistency_review" else detail
            group_value = record.get(group_column)
            group_size = 1
            if pd.notna(group_value):
                group_size = int((data[group_column] == group_value).sum())
            rows.append(
                {
                    "source_row": int(record["source_row"]),
                    "CID": record.get("CID"),
                    "legacy_v1_split": record.get("legacy_v1_split"),
                    "Class": int(record["Class"]),
                    "pIC50": record.get("pIC50"),
                    "standardized_smiles": record.get("standardized_smiles"),
                    "issue_type": flag,
                    "issue_detail": issue_detail,
                    "audit_group": group_value,
                    "audit_group_size": group_size,
                    "recommended_action": "retain_label_and_review_manually",
                }
            )
    columns = [
        "source_row", "CID", "legacy_v1_split", "Class", "pIC50", "standardized_smiles",
        "issue_type", "issue_detail", "audit_group", "audit_group_size", "recommended_action",
    ]
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["issue_type", "source_row"], ignore_index=True
    )


def compute_activity_cliffs(data: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Find opposite-label, high-similarity pairs from standardized structures."""
    rules = config["audit"]["activity_cliff"]
    radius, fp_size = int(rules["morgan_radius"]), int(rules["fingerprint_size"])
    minimum = float(rules["minimum_tanimoto"])
    potency_threshold = float(rules["potency_difference_log10_threshold"])
    valid = data[data["is_valid_smiles"]].reset_index(drop=True)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size, includeChirality=True)
    mols = [Chem.MolFromSmiles(smiles) for smiles in valid["standardized_smiles"]]
    fingerprints = [generator.GetFingerprint(mol) for mol in mols]
    rows: list[dict[str, Any]] = []
    for left_index, left_fp in enumerate(fingerprints):
        similarities = DataStructs.BulkTanimotoSimilarity(left_fp, fingerprints[left_index + 1 :])
        for offset, similarity in enumerate(similarities, start=left_index + 1):
            if similarity < minimum:
                continue
            left, right = valid.iloc[left_index], valid.iloc[offset]
            if int(left["Class"]) == int(right["Class"]):
                continue
            pdelta = abs(float(left["pIC50"]) - float(right["pIC50"]))
            rows.append(
                {
                    "source_row_a": int(left["source_row"]), "source_row_b": int(right["source_row"]),
                    "CID_a": left.get("CID"), "CID_b": right.get("CID"),
                    "legacy_v1_split_a": left.get("legacy_v1_split"), "legacy_v1_split_b": right.get("legacy_v1_split"),
                    "Class_a": int(left["Class"]), "Class_b": int(right["Class"]),
                    "pIC50_a": float(left["pIC50"]), "pIC50_b": float(right["pIC50"]),
                    "standardized_smiles_a": left["standardized_smiles"], "standardized_smiles_b": right["standardized_smiles"],
                    "tanimoto_similarity": float(similarity), "pIC50_absolute_difference": pdelta,
                    "potency_cliff_ge_1_log10": bool(pdelta >= potency_threshold),
                    "same_standardized_scaffold": bool(left["scaffold"] == right["scaffold"]),
                    "is_exact_standardized_structure_pair": bool(left["standardized_smiles"] == right["standardized_smiles"]),
                    "activity_cliff_rule": f"opposite_Class_and_Morgan_r{radius}_Tanimoto_ge_{minimum:g}",
                }
            )
    columns = [
        "source_row_a", "source_row_b", "CID_a", "CID_b", "legacy_v1_split_a", "legacy_v1_split_b",
        "Class_a", "Class_b", "pIC50_a", "pIC50_b", "standardized_smiles_a", "standardized_smiles_b",
        "tanimoto_similarity", "pIC50_absolute_difference", "potency_cliff_ge_1_log10",
        "same_standardized_scaffold", "is_exact_standardized_structure_pair", "activity_cliff_rule",
    ]
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["tanimoto_similarity", "pIC50_absolute_difference"], ascending=[False, False], ignore_index=True
    )


def build_data_audit(
    data: pd.DataFrame, cliffs: pd.DataFrame, raw_path: Path, legacy_path: Path, config_path: Path
) -> pd.DataFrame:
    valid = data["is_valid_smiles"].fillna(False)
    def count(column: str) -> int:
        return int(data[column].fillna(False).sum())

    rows = [
        ("provenance", "raw_csv_sha256", sha256_file(raw_path), "SHA-256 of immutable raw input"),
        ("provenance", "legacy_split_csv_sha256", sha256_file(legacy_path), "SHA-256 of frozen v1 split reference"),
        ("provenance", "configuration_sha256", sha256_file(config_path), "SHA-256 of consumed v2 configuration"),
        ("provenance", "rdkit_version", rdBase.rdkitVersion, "RDKit build used for standardization"),
        ("records", "raw_source_rows", int(len(data)), "Rows read from raw CSV; all are retained in bace_standardized.csv"),
        ("records", "valid_smiles", int(valid.sum()), "Rows whose raw SMILES parsed in RDKit"),
        ("records", "invalid_smiles", int((~valid).sum()), "Rows retained but not standardizable"),
        ("standardization", "multicomponent_rows", int((data.loc[valid, "component_count"] > 1).sum()), "Raw structures containing more than one dot-separated component"),
        ("standardization", "largest_fragment_selected_rows", int(data["standardization_action_trace"].str.contains("largest_fragment_selected", na=False).sum()), "Rows altered by largest-fragment selection"),
        ("standardization", "functional_group_normalized_rows", int(data["standardization_action_trace"].str.contains("functional_groups_normalized", na=False).sum()), "Rows changed by RDKit normalization"),
        ("standardization", "charge_reionized_rows", int(data["standardization_action_trace"].str.contains("charge_reionized", na=False).sum()), "Rows changed by RDKit reionization"),
        ("standardization", "charge_neutralized_rows", int(data["standardization_action_trace"].str.contains("charge_neutralized", na=False).sum()), "Rows changed by RDKit uncharging"),
        ("standardization", "tautomer_canonicalized_rows", int(data["standardization_action_trace"].str.contains("tautomer_canonicalized", na=False).sum()), "Rows changed by tautomer canonicalization"),
        ("duplicates", "exact_standardized_duplicate_rows", int(data.loc[valid, "standardized_smiles"].duplicated(keep=False).sum()), "Rows in a repeated standardized structure group"),
        ("duplicates", "conflicting_standardized_label_rows", count("standardized_structure_conflict"), "Rows in a standardized structure group with multiple Class values"),
        ("duplicates", "potential_stereo_duplicate_rows", count("potential_stereo_duplicate"), "Rows sharing a stereo-stripped but not isomeric representation"),
        ("duplicates", "potential_tautomer_duplicate_rows", count("potential_tautomer_duplicate"), "Rows that converge from distinct pre-tautomer structures"),
        ("duplicates", "cross_legacy_split_duplicate_rows", count("cross_legacy_split_duplicate"), "Rows in an exact standardized structure group across frozen v1 splits"),
        ("duplicates", "zero_cross_legacy_split_duplicate_requirement_met", bool(count("cross_legacy_split_duplicate") == 0), "Frozen v1 only: false requires a new v2 group-aware scaffold split; historical assignments remain unchanged"),
        ("labels", "class_pic50_consistency_review_rows", count("class_pic50_consistency_review"), "Diagnostic review only; Class labels were not changed"),
        ("labels", "active_class_rows", int((data["Class"] == 1).sum()), "Raw Class = 1"),
        ("labels", "inactive_class_rows", int((data["Class"] == 0).sum()), "Raw Class = 0"),
        ("scaffolds", "unique_standardized_scaffolds", int(data.loc[valid, "scaffold"].nunique(dropna=False)), "Bemis-Murcko scaffolds after standardization"),
        ("scaffolds", "acyclic_empty_scaffold_rows", int((data.loc[valid, "scaffold"] == "").sum()), "Valid standardized structures without a ring scaffold"),
        ("activity_cliffs", "opposite_label_high_similarity_pairs", int(len(cliffs)), "Pairs meeting the prespecified activity-cliff rule"),
        ("activity_cliffs", "potency_cliff_ge_1_log10_pairs", int(cliffs["potency_cliff_ge_1_log10"].sum()) if not cliffs.empty else 0, "Flagged pairs with |ΔpIC50| ≥ 1.0"),
    ]
    return pd.DataFrame(rows, columns=["audit_section", "metric", "value", "definition"])


def run(config_path: Path) -> dict[str, Path]:
    """Execute the Phase 1 audit and return written paths."""
    with config_path.open() as handle:
        config = yaml.safe_load(handle)
    dataset = config["dataset"]
    outputs = config["outputs"]
    raw_path = ROOT / dataset["raw_csv"]
    legacy_path = ROOT / dataset["legacy_split_csv"]
    raw = pd.read_csv(raw_path)
    legacy = pd.read_csv(legacy_path)
    required = {dataset["smiles_column"], dataset["identifier_column"], dataset["classification_target"], dataset["regression_target"]}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Raw CSV is missing required columns: {sorted(missing)}")
    if len(raw) != len(legacy) or not (legacy["source_row"].to_numpy() == range(len(raw))).all():
        raise ValueError("Frozen v1 split cannot be aligned to raw source rows.")

    base = raw[[dataset["identifier_column"], dataset["smiles_column"], dataset["classification_target"], dataset["regression_target"]]].copy()
    base.insert(0, "source_row", range(len(base)))
    base = base.rename(columns={dataset["smiles_column"]: "mol", dataset["classification_target"]: "Class", dataset["regression_target"]: "pIC50"})
    base["legacy_v1_split"] = legacy["split"].to_numpy()
    annotations = [standardize_smiles(str(smiles), config) if pd.notna(smiles) else {"is_valid_smiles": False, "standardization_status": "missing_smiles", "standardization_action_trace": "parse_skipped_missing_smiles"} for smiles in base["mol"]]
    standardized = annotate_review_flags(pd.concat([base, pd.DataFrame(annotations)], axis=1), config)
    label_audit = build_label_consistency_audit(standardized)
    cliffs = compute_activity_cliffs(standardized, config)
    audit = build_data_audit(standardized, cliffs, raw_path, legacy_path, config_path)

    paths = {key: ROOT / value for key, value in outputs.items()}
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    standardized.to_csv(paths["standardized_csv"], index=False)
    label_audit.to_csv(paths["label_consistency_audit_csv"], index=False)
    cliffs.to_csv(paths["activity_cliffs_csv"], index=False)
    audit.to_csv(paths["data_audit_csv"], index=False)
    metadata = {
        "phase": "phase1_data_standardization_audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_csv": str(raw_path.relative_to(ROOT)),
        "raw_csv_sha256": sha256_file(raw_path),
        "legacy_v1_split_csv": str(legacy_path.relative_to(ROOT)),
        "legacy_v1_split_sha256": sha256_file(legacy_path),
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256_file(config_path),
        "rdkit_version": rdBase.rdkitVersion,
        "output_sha256": {key: sha256_file(path) for key, path in paths.items() if key != "run_metadata_json"},
        "labels_modified": False,
        "rows_removed": 0,
        "v2_folds_created": False,
        "future_split_group_column": "v2_structure_group_id",
    }
    paths["run_metadata_json"].write_text(json.dumps(metadata, indent=2) + "\n")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs_v2/phase1_data_standardization.yaml")
    args = parser.parse_args()
    paths = run(args.config.resolve())
    for label, path in paths.items():
        print(f"{label}: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
