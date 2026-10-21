"""Descriptive outer-test reliability diagram for the frozen Phase 5 fusions.

The same BACE molecules occur in both scaffold-CV repeats. Training seeds are
averaged before binning; plotted bins are descriptive, not independent cohorts.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    from audit_panel_alignment import require_matplotlib_panel_alignment
except ImportError:
    require_matplotlib_panel_alignment = None


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results_v2/fusion_predictions_v2.csv"
FIGURE = ROOT / "results_v2/figures/phase6_fusion_reliability"
SOURCE_DATA = ROOT / "results_v2/phase6_reliability_bins.csv"
METHODS = ("C1_fixed_weight", "D2_similarity_entropy")


def reliability_bins(predictions: pd.DataFrame) -> pd.DataFrame:
    selected = predictions[predictions.method.isin(METHODS)].copy()
    if set(selected.method) != set(METHODS) or set(selected.split) != {"outer_test"}:
        raise ValueError("Expected frozen outer-test predictions for both fusions.")
    keys = ["repeat", "outer_fold", "source_row", "label", "method"]
    if selected.duplicated(keys + ["seed"]).any():
        raise ValueError("Duplicate seed prediction.")
    counts = selected.groupby(keys).seed.nunique()
    if not counts.eq(3).all():
        raise ValueError("Every repeat-molecule must have three training seeds.")
    averaged = selected.groupby(keys, as_index=False).probability.mean()
    wide = averaged.pivot(index=keys[:-1], columns="method", values="probability").reset_index()
    if len(wide) != 2 * 1513:
        raise ValueError("Expected each BACE molecule once per outer repeat.")
    if wide[list(METHODS)].isna().any().any():
        raise ValueError("Unpaired fusion predictions.")
    wide["bin"] = pd.qcut(wide[METHODS[0]], q=10, labels=False, duplicates="raise") + 1
    rows = []
    for method in METHODS:
        for bin_number, group in wide.groupby("bin", sort=True):
            rows.append({
                "method": method, "bin": int(bin_number),
                "n_repeat_molecules": len(group),
                "n_active_repeat_molecules": int(group.label.sum()),
                "mean_predicted": float(group[method].mean()),
                "observed_fraction": float(group.label.mean()),
                "bin_basis": "C1_fixed_weight seed-mean probability deciles",
            })
    result = pd.DataFrame(rows)
    if len(result) != 20 or result.n_repeat_molecules.sum() != 2 * len(wide):
        raise ValueError("Unexpected reliability-bin coverage.")
    return result


def plot(data: pd.DataFrame) -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7,
        "axes.spines.right": False, "axes.spines.top": False,
        "axes.linewidth": 0.8, "legend.frameon": False,
    })
    figure, axis = plt.subplots(figsize=(3.5, 3.25), layout="constrained")
    axis.plot([0, 1], [0, 1], color="#b6bec6", linewidth=1, label="Perfect calibration", zorder=1)
    for method, label, color, marker, linestyle in (
        (METHODS[0], "C1 fixed fusion", "#4f5d6b", "o", "-"),
        (METHODS[1], "D2 similarity + entropy", "#1763a6", "s", "--"),
    ):
        subset = data[data.method == method].sort_values("bin")
        axis.plot(subset.mean_predicted, subset.observed_fraction,
                  color=color, marker=marker, markersize=3, linewidth=1.3,
                  linestyle=linestyle, label=label, zorder=3)
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted activity probability",
             ylabel="Observed active fraction")
    axis.set_aspect("equal", adjustable="box")
    axis.legend(loc="lower left", bbox_to_anchor=(0, 1.02), fontsize=6.5,
                handlelength=2.5, borderaxespad=0)
    axis.grid(color="#e7ebef", linewidth=0.5)
    figure.canvas.draw()
    if require_matplotlib_panel_alignment is not None:
        require_matplotlib_panel_alignment(
            figure, json_out=f"{FIGURE}.alignment.json",
            overlay_svg=f"{FIGURE}.alignment.svg", tolerance_pt=1.5,
            gutter_tolerance_pt=1.5, strict=True,
        )
    figure.savefig(f"{FIGURE}.svg", bbox_inches="tight")
    figure.savefig(f"{FIGURE}.pdf", bbox_inches="tight")
    figure.savefig(f"{FIGURE}.png", dpi=600, bbox_inches="tight")
    figure.savefig(f"{FIGURE}.tiff", dpi=600, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    data = reliability_bins(pd.read_csv(INPUT))
    data.to_csv(SOURCE_DATA, index=False)
    plot(data)
    print(f"source_data: {SOURCE_DATA.relative_to(ROOT)}")
    print(f"figure: {FIGURE.relative_to(ROOT)}.pdf")


if __name__ == "__main__":
    main()
