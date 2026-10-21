"""Safety checks for the label-blind WP3 preflight."""

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import external_wp3_preflight as preflight


def test_preflight_hashes_cohort_without_parsing_labels(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight, "ROOT", tmp_path)
    monkeypatch.setattr(preflight, "COHORT_LOCK", tmp_path / "cohort_lock.json")
    monkeypatch.setattr(preflight, "BUNDLE", tmp_path / "missing_bundle.json")
    cohort = tmp_path / "eligible_cohort.csv"
    cohort.write_bytes(b"not,a,parseable,cohort\n\xff\xfe\n")
    expected = hashlib.sha256(cohort.read_bytes()).hexdigest()
    preflight.COHORT_LOCK.write_text(json.dumps({
        "status": "frozen_before_external_model_scoring",
        "files": {"eligible_cohort.csv": expected},
    }))

    result = preflight.check()

    assert result["status"] == "NO_GO_BEFORE_LABEL_EXPOSURE"
    assert result["checked_wp2_file_sha256"]["eligible_cohort.csv"] == expected
    assert result["external_labels_loaded"] is False
    assert result["external_predictions_generated"] is False
    assert result["external_metrics_calculated"] is False


def test_preflight_rejects_changed_locked_file(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight, "ROOT", tmp_path)
    monkeypatch.setattr(preflight, "COHORT_LOCK", tmp_path / "cohort_lock.json")
    monkeypatch.setattr(preflight, "BUNDLE", tmp_path / "missing_bundle.json")
    (tmp_path / "eligible_cohort.csv").write_bytes(b"changed")
    preflight.COHORT_LOCK.write_text(json.dumps({
        "status": "frozen_before_external_model_scoring",
        "files": {"eligible_cohort.csv": "0" * 64},
    }))

    result = preflight.check()

    assert any("locked file changed" in item for item in result["failures"])
