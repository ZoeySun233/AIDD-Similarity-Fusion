"""Snapshot and audit two ChEMBL BACE-1 assay cohorts without model scoring.

Usage: python src/curate_external_wp2.py --fetch
       python src/curate_external_wp2.py --audit
The snapshot is immutable: existing API responses are read, never replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import requests
import yaml
from rdkit import rdBase
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from preprocessing_v2 import standardize_smiles


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs_v2/external_wp2_chembl.yaml"
STANDARDIZATION = ROOT / "configs_v2/phase1_data_standardization.yaml"
ORIGINAL = ROOT / "data/processed_v2/bace_standardized.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def session() -> requests.Session:
    client = requests.Session()
    retry = Retry(
        total=4,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    client.mount("https://", HTTPAdapter(max_retries=retry))
    client.headers["User-Agent"] = "AIDD-Similarity-Fusion/2.0 external-data-feasibility"
    return client


def immutable_get(client: requests.Session, url: str, path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    response = client.get(url, timeout=(15, 90))
    response.raise_for_status()
    parsed = response.json()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(response.content)
    return parsed


def fetch() -> None:
    config = load_config()
    api = config["source"]["api_base"].rstrip("/")
    out = ROOT / config["source"]["snapshot_dir"]
    client = session()
    status = immutable_get(client, f"{api}/status.json", out / "status.json")
    if status.get("chembl_db_version") != "ChEMBL_37":
        raise ValueError(f"Unexpected release: {status.get('chembl_db_version')}")
    target = config["source"]["target_chembl_id"]
    immutable_get(client, f"{api}/target/{target}.json", out / "target.json")
    manifest = {
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "chembl_release": status["chembl_db_version"],
        "chembl_release_date": status.get("chembl_release_date"),
        "api_base": api,
        "target_chembl_id": target,
        "files": [],
        "assays": {},
    }
    for candidate in config["source"]["assays"]:
        assay_id = candidate["assay_id"]
        assay = immutable_get(
            client, f"{api}/assay/{assay_id}.json", out / f"assay_{assay_id}.json"
        )
        doc = assay["document_chembl_id"]
        immutable_get(client, f"{api}/document/{doc}.json", out / f"document_{doc}.json")
        offset = 0
        rows = 0
        seen_ids: set[int] = set()
        while True:
            params = urlencode({"assay_chembl_id": assay_id, "limit": 1000, "offset": offset})
            url = f"{api}/activity.json?{params}"
            page_path = out / "activities" / assay_id / f"page_{offset:06d}.json"
            page = immutable_get(client, url, page_path)
            activities = page["activities"]
            rows += len(activities)
            ids = [int(row["activity_id"]) for row in activities]
            if len(ids) != len(set(ids)) or seen_ids.intersection(ids):
                raise ValueError(f"Duplicate activity IDs in {assay_id}")
            seen_ids.update(ids)
            manifest["files"].append({
                "path": str(page_path.relative_to(ROOT)), "query_url": url,
                "sha256": sha256(page_path), "records": len(activities),
            })
            if not page["page_meta"].get("next"):
                if rows != page["page_meta"]["total_count"]:
                    raise ValueError(f"Incomplete assay download: {assay_id}")
                break
            offset += len(activities)
            if not activities:
                raise ValueError(f"Empty nonterminal activity page: {assay_id}")
        manifest["assays"][assay_id] = {"raw_activity_count": rows, "document": doc}
    for path in sorted(out.glob("*.json")):
        if path.name == "snapshot_manifest.json":
            continue
        manifest["files"].append({
            "path": str(path.relative_to(ROOT)), "sha256": sha256(path),
        })
    path = out / "snapshot_manifest.json"
    if path.exists():
        prior = json.loads(path.read_text(encoding="utf-8"))
        for item in prior["files"]:
            local = ROOT / item["path"]
            if sha256(local) != item["sha256"]:
                raise ValueError(f"Frozen snapshot changed: {local}")
        print("Existing snapshot verified; metadata left unchanged")
    else:
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(manifest["assays"], sort_keys=True))


def read_activities(snapshot: Path, assay_id: str) -> list[dict]:
    rows = []
    paths = sorted((snapshot / "activities" / assay_id).glob("page_*.json"))
    if not paths:
        raise FileNotFoundError(f"Missing activity pages for {assay_id}")
    for path in paths:
        rows.extend(json.loads(path.read_text(encoding="utf-8"))["activities"])
    return rows


def numeric_value(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result > 0 else None


def audit() -> None:
    config = load_config()
    snapshot = ROOT / config["source"]["snapshot_dir"]
    if (snapshot / "cohort_lock.json").exists():
        raise RuntimeError("External cohort is locked; verify hashes instead of rewriting it")
    frozen = json.loads((snapshot / "snapshot_manifest.json").read_text(encoding="utf-8"))
    for item in frozen["files"]:
        if sha256(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"Snapshot checksum failed: {item['path']}")
    original = pd.read_csv(ORIGINAL)
    original_source = pd.read_csv(
        ROOT / "data/external_v2/original_bace_source/source_provenance.csv"
    )
    original_parent_ids = set(original_source["source_parent_chembl_id"].dropna())
    original_pubmed_ids = set(original_source["source_pubmed_id"].dropna().astype(str))
    standard_config = yaml.safe_load(STANDARDIZATION.read_text(encoding="utf-8"))
    original_sets = {
        "raw_exact": set(original["raw_canonical_smiles"].dropna()),
        "standardized": set(original["standardized_smiles"].dropna()),
        "inchikey": set(original["standardized_inchikey"].dropna()),
        "no_stereo": set(original["standardized_smiles_no_stereo"].dropna()),
        "scaffold": set(original["scaffold"].dropna()),
    }
    manifest_rows: list[dict] = []
    provenance_rows: list[dict] = []
    structure_cache: dict[str, dict] = {}
    for candidate in config["source"]["assays"]:
        assay_id = candidate["assay_id"]
        assay = json.loads((snapshot / f"assay_{assay_id}.json").read_text(encoding="utf-8"))
        doc = json.loads((snapshot / f"document_{assay['document_chembl_id']}.json").read_text(encoding="utf-8"))
        if assay.get("assay_organism") != config["source"]["organism"]:
            raise ValueError(f"Assay organism is not human: {assay_id}")
        if assay.get("assay_type") != config["eligibility"]["assay_type"]:
            raise ValueError(f"Assay is not biochemical binding/inhibition: {assay_id}")
        provenance_rows.append({
            "assay_chembl_id": assay_id,
            "document_chembl_id": assay["document_chembl_id"],
            "document_title": doc.get("title"),
            "document_year": doc.get("year"),
            "document_doi": doc.get("doi"),
            "document_pubmed_id": doc.get("pubmed_id"),
            "patent_id": doc.get("patent_id"),
            "patent_priority_date": candidate["patent_priority_date"],
            "patent_url": candidate["patent_url"],
            "original_source_latest_year": int(original_source["source_year"].max()),
            "original_source_pubmed_ids_available": len(original_pubmed_ids),
            "document_pubmed_overlap": (
                str(doc["pubmed_id"]) in original_pubmed_ids
                if doc.get("pubmed_id") else None
            ),
            "original_assay_ids_available": False,
            "direct_assay_id_overlap_assessable": False,
            "original_patent_ids_available": False,
            "direct_patent_id_overlap_assessable": False,
            "assay_organism": assay.get("assay_organism"),
            "assay_type": assay.get("assay_type"),
            "bao_label": assay.get("bao_label"),
        })
        for row in read_activities(snapshot, assay_id):
            reasons = []
            if row.get("target_chembl_id") != config["source"]["target_chembl_id"]:
                reasons.append("wrong_target")
            if row.get("target_organism") != config["source"]["organism"]:
                reasons.append("wrong_species")
            if row.get("assay_type") != config["eligibility"]["assay_type"]:
                reasons.append("wrong_assay_type")
            for field, expected in (
                ("standard_type", "standard_type"),
                ("standard_units", "standard_units"),
                ("standard_relation", "standard_relation"),
            ):
                if row.get(field) != config["eligibility"][expected]:
                    reasons.append(f"wrong_{field}")
            value = numeric_value(row.get("standard_value"))
            if value is None:
                reasons.append("missing_or_invalid_ic50")
            if row.get("potential_duplicate") != 0:
                reasons.append("chembl_potential_duplicate")
            if row.get("data_validity_comment") not in config["eligibility"]["acceptable_data_validity_comments"]:
                reasons.append("adverse_data_validity_comment")
            smiles = row.get("canonical_smiles")
            if not smiles:
                reasons.append("missing_smiles")
                structure = {}
            else:
                if smiles not in structure_cache:
                    structure_cache[smiles] = standardize_smiles(smiles, standard_config)
                structure = structure_cache[smiles]
                if not structure.get("is_valid_smiles"):
                    reasons.append("invalid_smiles")
            standardized = structure.get("standardized_smiles")
            overlap = {
                "raw_exact": bool(structure.get("raw_canonical_smiles") in original_sets["raw_exact"]),
                "standardized": bool(standardized in original_sets["standardized"]),
                "inchikey": bool(structure.get("standardized_inchikey") in original_sets["inchikey"]),
                "no_stereo": bool(structure.get("standardized_smiles_no_stereo") in original_sets["no_stereo"]),
                "scaffold": bool(structure.get("scaffold") in original_sets["scaffold"]),
            }
            if overlap["raw_exact"] or overlap["standardized"] or overlap["inchikey"]:
                reasons.append("original_compound_overlap")
            parent_overlap = row.get("parent_molecule_chembl_id") in original_parent_ids
            if parent_overlap:
                reasons.append("original_parent_chembl_overlap")
            if not row.get("parent_molecule_chembl_id"):
                reasons.append("missing_parent_id_review")
            manifest_rows.append({
                "assay_chembl_id": assay_id,
                "candidate_role": candidate["role"],
                "activity_id": row.get("activity_id"),
                "record_id": row.get("record_id"),
                "molecule_chembl_id": row.get("molecule_chembl_id"),
                "parent_molecule_chembl_id": row.get("parent_molecule_chembl_id"),
                "target_chembl_id": row.get("target_chembl_id"),
                "target_organism": row.get("target_organism"),
                "document_chembl_id": row.get("document_chembl_id"),
                "document_year": row.get("document_year"),
                "document_doi": doc.get("doi"),
                "document_pubmed_id": doc.get("pubmed_id"),
                "patent_id": doc.get("patent_id"),
                "patent_priority_date": candidate["patent_priority_date"],
                "src_id": row.get("src_id"),
                "assay_confidence_score": assay.get("confidence_score"),
                "standard_type": row.get("standard_type"),
                "standard_relation": row.get("standard_relation"),
                "standard_units": row.get("standard_units"),
                "standard_value": row.get("standard_value"),
                "numeric_ic50_nm": value,
                "potential_duplicate": row.get("potential_duplicate"),
                "data_validity_comment": row.get("data_validity_comment"),
                "raw_smiles": smiles,
                "raw_canonical_smiles": structure.get("raw_canonical_smiles"),
                "standardized_smiles": standardized,
                "standardized_inchikey": structure.get("standardized_inchikey"),
                "standardized_smiles_no_stereo": structure.get("standardized_smiles_no_stereo"),
                "scaffold": structure.get("scaffold"),
                "component_count": structure.get("component_count"),
                "standardization_action_trace": structure.get("standardization_action_trace"),
                "overlap_raw_exact": int(overlap["raw_exact"]),
                "overlap_standardized": int(overlap["standardized"]),
                "overlap_inchikey": int(overlap["inchikey"]),
                "overlap_no_stereo": int(overlap["no_stereo"]),
                "overlap_scaffold": int(overlap["scaffold"]),
                "overlap_original_parent_chembl_id": int(parent_overlap),
                "record_exclusion_reasons": ";".join(reasons),
            })
    by_structure: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in manifest_rows:
        if not row["record_exclusion_reasons"] and row["standardized_smiles"]:
            by_structure[(row["assay_chembl_id"], row["standardized_smiles"])].append(row)
    cutoff = float(config["eligibility"]["maximum_ic50_nm_for_active"])
    cohort_rows = []
    for (assay_id, smiles), records in sorted(by_structure.items()):
        values = [float(r["numeric_ic50_nm"]) for r in records]
        classes = {int(value <= cutoff) for value in values}
        if len(classes) > 1:
            for record in records:
                record["record_exclusion_reasons"] = "cross_threshold_replicates"
            continue
        cohort_rows.append({
            "assay_chembl_id": assay_id,
            "candidate_role": records[0]["candidate_role"],
            "standardized_smiles": smiles,
            "standardized_inchikey": records[0]["standardized_inchikey"],
            "scaffold": records[0]["scaffold"],
            "activity_ids": ";".join(str(r["activity_id"]) for r in records),
            "parent_molecule_chembl_ids": ";".join(sorted({str(r["parent_molecule_chembl_id"]) for r in records})),
            "replicate_count": len(records),
            "minimum_ic50_nm": min(values),
            "maximum_ic50_nm": max(values),
            "median_log10_ic50_nm": statistics.median(math.log10(v) for v in values),
            "label": classes.pop(),
        })
    outputs = config["outputs"]
    manifest_path = ROOT / outputs["curation_manifest"]
    cohort_path = ROOT / outputs["cohort"]
    summary_path = ROOT / outputs["audit_summary"]
    provenance_path = ROOT / "results_v2/external_wp2_provenance_audit.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    cohort_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    pd.DataFrame(cohort_rows).to_csv(cohort_path, index=False)
    pd.DataFrame(provenance_rows).to_csv(provenance_path, index=False)
    by_assay = {}
    for candidate in config["source"]["assays"]:
        assay_id = candidate["assay_id"]
        all_rows = [row for row in manifest_rows if row["assay_chembl_id"] == assay_id]
        included = [row for row in cohort_rows if row["assay_chembl_id"] == assay_id]
        by_assay[assay_id] = {
            "role": candidate["role"],
            "raw_activity_records": len(all_rows),
            "record_exclusion_reasons": dict(Counter(
                reason for row in all_rows
                for reason in row["record_exclusion_reasons"].split(";") if reason
            )),
            "resolved_structures": len(included),
            "active_structures": sum(row["label"] == 1 for row in included),
            "inactive_structures": sum(row["label"] == 0 for row in included),
            "scaffolds": len({row["scaffold"] for row in included}),
            "original_structure_overlap_records": sum(
                row["overlap_raw_exact"] or row["overlap_standardized"] or row["overlap_inchikey"]
                for row in all_rows
            ),
            "original_scaffold_overlap_records": sum(row["overlap_scaffold"] for row in all_rows),
            "original_parent_chembl_overlap_records": sum(
                row["overlap_original_parent_chembl_id"] for row in all_rows
            ),
        }
    summary = {
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "chembl_release": frozen["chembl_release"],
        "chembl_release_date": frozen["chembl_release_date"],
        "raw_bace_sha256": sha256(ROOT / "data/raw/bace.csv"),
        "original_standardized_sha256": sha256(ORIGINAL),
        "source_si_sha256": sha256(ROOT / "data/external_v2/original_bace_source/ci6b00290_si_001.zip"),
        "config_sha256": sha256(CONFIG),
        "standardization_config_sha256": sha256(STANDARDIZATION),
        "rdkit_version": rdBase.rdkitVersion,
        "assays": by_assay,
        "cross_assay_structure_overlap": len(
            set(r["standardized_smiles"] for r in cohort_rows if r["assay_chembl_id"] == config["source"]["assays"][0]["assay_id"])
            & set(r["standardized_smiles"] for r in cohort_rows if r["assay_chembl_id"] == config["source"]["assays"][1]["assay_id"])
        ),
        "curation_manifest_sha256": sha256(manifest_path),
        "cohort_sha256": sha256(cohort_path),
        "provenance_audit_sha256": sha256(provenance_path),
        "model_scoring_performed": False,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(by_assay, indent=2, sort_keys=True))


def lock() -> None:
    config = load_config()
    snapshot = ROOT / config["source"]["snapshot_dir"]
    summary = json.loads((ROOT / config["outputs"]["audit_summary"]).read_text(encoding="utf-8"))
    paths = [
        snapshot / "snapshot_manifest.json",
        ROOT / config["outputs"]["cohort"],
        ROOT / config["outputs"]["curation_manifest"],
        ROOT / "results_v2/external_wp2_provenance_audit.csv",
        ROOT / "results_v2/external_wp2_original_label_summary.json",
        CONFIG,
        STANDARDIZATION,
        ROOT / "src/curate_external_wp2.py",
        ROOT / "src/extract_bace_si_wp2.py",
        ROOT / "report_v2/external_eligibility_protocol.md",
    ]
    lock_path = snapshot / "cohort_lock.json"
    if lock_path.exists():
        saved = json.loads(lock_path.read_text(encoding="utf-8"))
        if any(sha256(ROOT / path) != checksum for path, checksum in saved["files"].items()):
            raise ValueError("Locked cohort or its curation inputs have changed")
        print("Existing cohort lock verified")
        return
    if summary["model_scoring_performed"]:
        raise ValueError("Cannot lock a cohort after model scoring")
    frozen = {
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_external_model_scoring",
        "primary_assay": "CHEMBL5736880",
        "secondary_assay": "CHEMBL5734173",
        "chembl_release": summary["chembl_release"],
        "files": {str(path.relative_to(ROOT)): sha256(path) for path in paths},
    }
    with lock_path.open("x", encoding="utf-8") as stream:
        json.dump(frozen, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(f"Cohort locked: {lock_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fetch", action="store_true")
    mode.add_argument("--audit", action="store_true")
    mode.add_argument("--lock", action="store_true")
    args = parser.parse_args()
    if args.fetch:
        fetch()
    elif args.audit:
        audit()
    else:
        lock()
