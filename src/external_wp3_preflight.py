"""Read-only, label-blind readiness check for the one-time external evaluation.

This program hashes the locked cohort as bytes. It never parses the cohort CSV,
imports a model, or calculates an external prediction or metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COHORT_LOCK = ROOT / "data/external_v2/chembl_37_wp2/cohort_lock.json"
BUNDLE = ROOT / "models_v2/external_frozen_bundle/manifest.json"
OUTPUT = ROOT / "results_v2/external_wp3_preflight.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _within_root(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError(f"Artifact path escapes project root: {relative}")
    return path


def check() -> dict:
    failures: list[str] = []
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
        failures.append("Cannot identify the current code revision.")

    cohort_files: dict[str, str] = {}
    if not COHORT_LOCK.is_file():
        failures.append("WP2 cohort lock is missing.")
    else:
        lock = json.loads(COHORT_LOCK.read_text())
        if lock.get("status") != "frozen_before_external_model_scoring":
            failures.append("WP2 cohort lock has the wrong status.")
        for relative, expected in lock.get("files", {}).items():
            try:
                path = _within_root(relative)
                actual = sha256_file(path)
            except (OSError, ValueError) as exc:
                failures.append(f"WP2 locked file unavailable: {relative}: {exc}")
                continue
            cohort_files[relative] = actual
            if actual != expected:
                failures.append(f"WP2 locked file changed: {relative}")

    bundle_info = None
    if not BUNDLE.is_file():
        failures.append(
            "No frozen v2 deployment bundle exists. Phase 3 saved predictions, "
            "not deployable outer models; Phase 4/5 saved context-specific "
            "calibration and fusion fits, not a single locked external pipeline."
        )
    else:
        try:
            bundle_info = json.loads(BUNDLE.read_text())
            required = {
                "created_before_external_scoring", "training_data_sha256", "code_revision",
                "rf_checkpoint", "chemberta_checkpoint", "embedding_scaler",
                "encoder_revision", "rf_calibrator", "chemberta_calibrator",
                "c1_weight", "d2_parameters", "probability_clip", "thresholds",
                "ranking_tie_break", "analysis_script", "artifact_sha256",
            }
            missing = sorted(required - bundle_info.keys())
            if missing:
                failures.append(f"Frozen bundle manifest missing keys: {missing}")
            hashes = bundle_info.get("artifact_sha256", {})
            if not isinstance(hashes, dict) or not hashes:
                failures.append("Frozen bundle has no artifact hashes.")
            else:
                for relative, expected in hashes.items():
                    try:
                        actual = sha256_file(_within_root(relative))
                    except (OSError, ValueError) as exc:
                        failures.append(f"Bundle artifact unavailable: {relative}: {exc}")
                        continue
                    if actual != expected:
                        failures.append(f"Bundle artifact changed: {relative}")
            if bundle_info.get("created_before_external_scoring") is not True:
                failures.append("Bundle is not attested as frozen before external scoring.")
            if bundle_info.get("code_revision") != revision:
                failures.append("Bundle code revision differs from the current revision.")
        except (OSError, ValueError, TypeError) as exc:
            failures.append(f"Frozen bundle manifest cannot be validated: {exc}")

    return {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "READY" if not failures else "NO_GO_BEFORE_LABEL_EXPOSURE",
        "external_labels_loaded": False,
        "external_predictions_generated": False,
        "external_metrics_calculated": False,
        "code_revision": revision,
        "cohort_lock_sha256": sha256_file(COHORT_LOCK) if COHORT_LOCK.is_file() else None,
        "checked_wp2_file_sha256": cohort_files,
        "bundle_manifest": str(BUNDLE.relative_to(ROOT)),
        "bundle_present": bundle_info is not None,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = check()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{result['status']}: {len(result['failures'])} issue(s)")
    for failure in result["failures"]:
        print(f"- {failure}")
    raise SystemExit(0 if result["status"] == "READY" else 2)


if __name__ == "__main__":
    main()
