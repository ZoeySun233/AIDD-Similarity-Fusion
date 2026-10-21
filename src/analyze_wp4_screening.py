"""Exploratory accept/defer analysis using only frozen WP3 external predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
try:
    from audit_panel_alignment import require_matplotlib_panel_alignment
except ImportError:
    require_matplotlib_panel_alignment = None


STYLE = {
    "c1_boundary_distance": ("C1 boundary distance", "#176D80", "o"),
    "expert_agreement": ("Expert agreement", "#6E7B8B", "s"),
    "training_similarity": ("Training similarity", "#C28B2C", "^"),
}

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.spines.right": False, "axes.spines.top": False,
    "axes.linewidth": 0.8, "legend.frameon": False,
    "svg.fonttype": "none", "pdf.fonttype": 42,
})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sorted_indices(score: np.ndarray, identity: np.ndarray) -> np.ndarray:
    """Higher reliability first, with an identity-only exact tie-break."""
    return np.lexsort((identity, -score))


def analyze_assay(frame: pd.DataFrame, config: dict) -> list[dict]:
    frame = frame.reset_index(drop=True)
    n = len(frame)
    y = frame.label.to_numpy(int)
    p = frame.c1_probability.to_numpy(float)
    identities = (frame.assay_chembl_id.astype(str) + "|" + frame.standardized_inchikey.astype(str)).to_numpy()
    scores = {
        "c1_boundary_distance": np.abs(p - 0.5),
        "expert_agreement": -np.abs(frame.rf_probability.to_numpy(float) - frame.chem_probability.to_numpy(float)),
        "training_similarity": frame.max_similarity.to_numpy(float),
    }
    k = max(1, math.ceil(float(config["screening_fraction_of_original_cohort"]) * n))
    rows = []
    for rule in config["deferral_rules"]:
        ordering = sorted_indices(scores[rule], identities)
        for target_coverage in config["coverage_fractions"]:
            retained_n = min(n, max(k, math.ceil(float(target_coverage) * n)))
            retained = ordering[:retained_n]
            deferred = ordering[retained_n:]
            # Screening ranking always uses frozen C1 activity probability.
            top = retained[sorted_indices(p[retained], identities[retained])[:k]]
            rows.append({
                "assay": str(frame.assay_chembl_id.iloc[0]),
                "rule": rule,
                "target_coverage": float(target_coverage),
                "coverage": retained_n / n,
                "original_n": n,
                "retained_n": retained_n,
                "deferred_n": len(deferred),
                "retained_active": int(y[retained].sum()),
                "retained_inactive": int(retained_n - y[retained].sum()),
                "retained_prevalence": float(y[retained].mean()),
                "deferred_active": int(y[deferred].sum()),
                "deferred_inactive": int(len(deferred) - y[deferred].sum()),
                "retained_brier": float(np.mean(np.square(y[retained] - p[retained]))),
                "deferred_brier": float(np.mean(np.square(y[deferred] - p[deferred]))) if len(deferred) else None,
                "retained_error_at_0_5": float(np.mean((p[retained] >= 0.5) != y[retained])),
                "fixed_screening_k": k,
                "top_k_hits": int(y[top].sum()),
                "precision_at_fixed_top_5pct": float(y[top].mean()),
            })
    return rows


def draw_primary(table: pd.DataFrame, output: Path, metric: str, ylabel: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5.25, 3.5))
    for rule, (label, color, marker) in STYLE.items():
        subset = table[table.rule == rule].sort_values("coverage")
        ax.plot(subset.coverage.to_numpy() * 100, subset[metric], color=color, marker=marker,
                markersize=4.2, linewidth=1.35, label=label)
    ax.set(xlim=(48, 102), xlabel="Coverage (% of original assay cohort)", ylabel=ylabel, title=title)
    ax.set_ylim(0, 1.05 if metric == "precision_at_fixed_top_5pct" else 0.25)
    ax.set_xticks([50, 60, 70, 80, 90, 100])
    ax.grid(axis="y", linewidth=0.45, color="#DFE3E8")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, columnspacing=1.1)
    fig.subplots_adjust(bottom=0.28)
    fig.canvas.draw()
    if require_matplotlib_panel_alignment is not None:
        require_matplotlib_panel_alignment(
            fig, json_out=str(output) + ".alignment.json",
            overlay_svg=str(output) + ".alignment.svg",
            tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True,
        )
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def benchmark_original_inputs(config: dict) -> dict:
    """Time one cold frozen-pipeline call on original structures, not external labels."""
    from evaluate_external_wp3_deployment import predict_components
    from chemberta import choose_encoder_device

    training = pd.read_csv(ROOT / config["benchmark"]["input_csv"], usecols=["source_row", "standardized_smiles"])
    training = training.sort_values("source_row").head(int(config["benchmark"]["n_original_molecules"]))
    manifest = json.loads((ROOT / config["frozen_manifest"]).read_text())
    deployment_config = yaml.safe_load((ROOT / "configs_v2/external_wp3_deployment.yaml").read_text())
    start = time.perf_counter()
    predicted = predict_components(training[["standardized_smiles"]], deployment_config, manifest)
    elapsed = time.perf_counter() - start
    if len(predicted) != len(training):
        raise ValueError("Inference benchmark returned an incomplete batch.")
    return {
        "n_original_training_molecules": len(training),
        "elapsed_seconds": elapsed,
        "seconds_per_molecule_amortized": elapsed / len(training),
        "scope": "One cold batch call including model loading, local frozen ChemBERTa encoding, RF, calibration, C1/D2 and similarity; excludes external-cohort scoring and network downloads.",
        "torch_threads_max": 4,
        "encoder_device": str(choose_encoder_device()),
        "machine": platform.machine(),
        "platform": platform.platform(),
    }


def run(config_path: Path, skip_benchmark: bool = False) -> Path:
    config = yaml.safe_load(config_path.read_text())
    input_path = ROOT / config["input_predictions"]
    frame = pd.read_csv(input_path)
    required = {"assay_chembl_id", "standardized_inchikey", "label", "c1_probability",
                "rf_probability", "chem_probability", "max_similarity"}
    if missing := required - set(frame):
        raise ValueError(f"Frozen prediction table is missing: {sorted(missing)}")
    if set(frame.assay_chembl_id) != {config["primary_assay"], config["secondary_assay"]}:
        raise ValueError("Assay identity changed from the locked WP4 protocol.")
    if len(frame) != 349 or frame.duplicated(["assay_chembl_id", "standardized_inchikey"]).any():
        raise ValueError("External paired-prediction rows are missing or duplicated.")
    rows = []
    for assay in (config["primary_assay"], config["secondary_assay"]):
        rows.extend(analyze_assay(frame[frame.assay_chembl_id == assay], config))
    result = pd.DataFrame(rows)
    output = ROOT / config["output_directory"]
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / "coverage_screening.csv", index=False)
    primary = result[result.assay == config["primary_assay"]]
    draw_primary(primary, output / "coverage_brier", "retained_brier", "Mean C1 Brier loss in retained molecules", "Primary assay: error after deferral")
    draw_primary(primary, output / "coverage_precision_top5", "precision_at_fixed_top_5pct", "Precision at fixed top-8 budget", "Primary assay: early-screening precision")
    bookkeeping_start = time.perf_counter()
    for assay in (config["primary_assay"], config["secondary_assay"]):
        analyze_assay(frame[frame.assay_chembl_id == assay], config)
    bookkeeping_seconds = time.perf_counter() - bookkeeping_start
    timing = {"deferral_analysis_seconds_for_349_saved_predictions": bookkeeping_seconds}
    if not skip_benchmark:
        timing["original_structure_inference"] = benchmark_original_inputs(config)
    else:
        timing["original_structure_inference"] = "not_measured"
    metadata = {
        "stage": config["stage"], "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation": "Exploratory post-WP3 analysis on already viewed external labels; no model refit or new independent validation.",
        "input_predictions_sha256": sha256(input_path),
        "frozen_manifest_sha256": sha256(ROOT / config["frozen_manifest"]),
        "config_sha256": sha256(config_path),
        "protocol_sha256": sha256(ROOT / "report_v2/wp4_screening_protocol.md"),
        "assay_counts": frame.assay_chembl_id.value_counts().to_dict(),
        "timing": timing,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs_v2/wp4_screening.yaml")
    parser.add_argument("--skip-benchmark", action="store_true")
    args = parser.parse_args()
    result = run(args.config.resolve(), skip_benchmark=args.skip_benchmark)
    print(f"Exploratory WP4 results: {result}")


if __name__ == "__main__":
    main()
