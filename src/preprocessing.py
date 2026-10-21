"""Week 1 audit and scaffold split for the BACE classification task."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.utils.multiclass import type_of_target


SPLIT_NAMES = ("train", "valid", "test")
SPLIT_FRACTIONS = np.array([0.70, 0.15, 0.15], dtype=float)

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        # Log-axis exponent glyphs are scaled below their parent size; 8 pt
        # keeps every rendered PDF glyph above the 5 pt QA floor.
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_molecules(df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for row_id, smiles in enumerate(df["mol"]):
        mol = None if pd.isna(smiles) else Chem.MolFromSmiles(str(smiles))
        if mol is None:
            records.append(
                {
                    "row_id": row_id,
                    "is_valid_smiles": False,
                    "canonical_smiles": pd.NA,
                    "scaffold": pd.NA,
                }
            )
            continue
        records.append(
            {
                "row_id": row_id,
                "is_valid_smiles": True,
                "canonical_smiles": Chem.MolToSmiles(
                    mol, canonical=True, isomericSmiles=True
                ),
                "scaffold": MurckoScaffold.MurckoScaffoldSmiles(
                    mol=mol, includeChirality=False
                ),
            }
        )
    annotations = pd.DataFrame.from_records(records).set_index("row_id")
    return pd.concat([df.reset_index(drop=True), annotations], axis=1)


def clean_and_audit(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    required = {"mol", "Class", "pIC50"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df["Class"].isna().any():
        raise ValueError(f"Class contains {int(df['Class'].isna().sum())} missing labels.")

    label_type = type_of_target(df["Class"].to_numpy())
    if label_type != "binary" or not set(df["Class"].unique()).issubset({0, 1}):
        raise ValueError("Class must be a binary 0/1 target.")

    annotated = parse_molecules(df)
    valid = annotated[annotated["is_valid_smiles"]].copy()
    conflict_flags = valid.groupby("canonical_smiles")["Class"].transform("nunique") > 1
    conflict_rows = valid.loc[
        conflict_flags, ["mol", "canonical_smiles", "Class", "pIC50"]
    ].copy()

    # Conflicts are excluded. Consistent canonical duplicates keep their first
    # occurrence so the same molecule cannot leak across splits.
    cleaned = valid.loc[~conflict_flags].drop_duplicates(
        subset="canonical_smiles", keep="first"
    )
    cleaned = cleaned.reset_index(names="source_row")

    audit = {
        "rows_raw": int(len(df)),
        "missing_smiles": int(df["mol"].isna().sum()),
        "missing_class_labels": int(df["Class"].isna().sum()),
        "invalid_smiles": int((~annotated["is_valid_smiles"]).sum()),
        "raw_smiles_duplicate_rows": int(df["mol"].duplicated(keep=False).sum()),
        "canonical_duplicate_rows": int(
            valid["canonical_smiles"].duplicated(keep=False).sum()
        ),
        "conflicting_label_groups": int(
            conflict_rows["canonical_smiles"].nunique()
        ),
        "conflicting_label_rows": int(len(conflict_rows)),
        "rows_cleaned": int(len(cleaned)),
        "rows_removed": int(len(df) - len(cleaned)),
        "class_counts_raw": {
            str(int(key)): int(value)
            for key, value in df["Class"].value_counts().sort_index().items()
        },
        "positive_fraction_raw": float(df["Class"].mean()),
        "unique_scaffolds_cleaned": int(cleaned["scaffold"].nunique()),
        "empty_scaffold_rows": int((cleaned["scaffold"] == "").sum()),
    }
    return cleaned, audit, conflict_rows


def scaffold_split(df: pd.DataFrame) -> pd.Series:
    """Allocate intact scaffold groups while balancing size and class counts."""

    groups: list[tuple[str, np.ndarray, int, int]] = []
    for scaffold, group in df.groupby("scaffold", sort=False, dropna=False):
        indices = group.index.to_numpy()
        groups.append(
            (str(scaffold), indices, int(len(group)), int(group["Class"].sum()))
        )

    positive_rate = float(df["Class"].mean())
    groups.sort(
        key=lambda item: (
            item[2],
            abs(item[3] / item[2] - positive_rate),
            item[0],
        ),
        reverse=True,
    )

    target_rows = SPLIT_FRACTIONS * len(df)
    target_positives = SPLIT_FRACTIONS * float(df["Class"].sum())
    row_counts = np.zeros(3, dtype=float)
    positive_counts = np.zeros(3, dtype=float)
    assignments = pd.Series(index=df.index, dtype="object", name="split")

    for _, indices, size, positives in groups:
        candidate_scores: list[float] = []
        for split_index in range(3):
            candidate_rows = row_counts.copy()
            candidate_positives = positive_counts.copy()
            candidate_rows[split_index] += size
            candidate_positives[split_index] += positives
            row_error = np.square((candidate_rows - target_rows) / target_rows).sum()
            class_error = np.square(
                (candidate_positives - target_positives)
                / np.maximum(target_positives, 1.0)
            ).sum()
            overflow = np.maximum(candidate_rows - target_rows, 0.0) / target_rows
            candidate_scores.append(
                row_error + 2.0 * class_error + 10.0 * np.square(overflow).sum()
            )

        chosen = int(np.argmin(candidate_scores))
        assignments.loc[indices] = SPLIT_NAMES[chosen]
        row_counts[chosen] += size
        positive_counts[chosen] += positives

    if assignments.isna().any():
        raise RuntimeError("At least one molecule was not assigned to a split.")
    return assignments


def verify_split(df: pd.DataFrame) -> dict[str, object]:
    scaffold_sets = {
        name: set(df.loc[df["split"] == name, "scaffold"]) for name in SPLIT_NAMES
    }
    overlaps = {
        "train_valid": len(scaffold_sets["train"] & scaffold_sets["valid"]),
        "train_test": len(scaffold_sets["train"] & scaffold_sets["test"]),
        "valid_test": len(scaffold_sets["valid"] & scaffold_sets["test"]),
    }
    if any(overlaps.values()):
        raise AssertionError(f"Scaffold leakage detected: {overlaps}")

    observed = df["split"].value_counts(normalize=True).reindex(SPLIT_NAMES)
    ratio_error = float(np.max(np.abs(observed.to_numpy() - SPLIT_FRACTIONS)))
    if ratio_error > 0.01:
        raise AssertionError(
            f"Split ratios differ by more than one percentage point: {observed}"
        )
    return {
        "scaffold_overlap_counts": overlaps,
        "all_rows_assigned": bool(df["split"].notna().all()),
        "ratio_max_absolute_error": ratio_error,
    }


def build_split_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in SPLIT_NAMES:
        subset = df[df["split"] == name]
        counts = subset["Class"].value_counts()
        rows.append(
            {
                "split": name,
                "n_compounds": int(len(subset)),
                "fraction": float(len(subset) / len(df)),
                "n_inactive": int(counts.get(0, 0)),
                "n_active": int(counts.get(1, 0)),
                "active_fraction": float(subset["Class"].mean()),
                "n_scaffolds": int(subset["scaffold"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def save_figure(fig: plt.Figure, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_class_balance(df: pd.DataFrame, output_dir: Path) -> None:
    counts = df["Class"].value_counts().reindex([0, 1], fill_value=0)
    fig, ax = plt.subplots(figsize=(3.50, 2.75), constrained_layout=True)
    bars = ax.bar(
        ["Inactive (0)", "Active (1)"], counts, color=["#8C96A3", "#4C78A8"]
    )
    ax.set_ylabel("Number of compounds")
    ax.set_title("BACE class balance")
    ax.bar_label(bars, labels=[f"{n}\n({n / len(df):.1%})" for n in counts], padding=3)
    ax.set_ylim(0, max(counts) * 1.18)
    save_figure(fig, output_dir / "class_balance")


def plot_scaffold_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    sizes = df.groupby("scaffold", dropna=False).size().sort_values(ascending=False)
    ranks = np.arange(1, len(sizes) + 1)
    if not (np.all(ranks > 0) and np.all(sizes.to_numpy() > 0)):
        raise ValueError("Log-scale scaffold plot requires strictly positive values.")
    fig, ax = plt.subplots(figsize=(3.50, 2.75), constrained_layout=True)
    ax.scatter(ranks, sizes.to_numpy(), s=11, color="#4C78A8", alpha=0.78, linewidths=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Scaffold rank (log scale)")
    ax.set_ylabel("Compounds per scaffold (log scale)")
    ax.set_title("Long-tailed scaffold sizes")
    ax.text(
        0.98,
        0.96,
        f"{len(sizes)} unique scaffolds\nlargest group: {int(sizes.iloc[0])}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color="#3A3A3A",
    )
    save_figure(fig, output_dir / "scaffold_distribution")


def plot_split_balance(summary: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.50, 2.75), constrained_layout=True)
    x = np.arange(len(summary))
    inactive = summary["n_inactive"].to_numpy()
    active = summary["n_active"].to_numpy()
    ax.bar(x, inactive, color="#AAB2BD", label="Inactive (0)")
    ax.bar(x, active, bottom=inactive, color="#4C78A8", label="Active (1)")
    ax.set_xticks(x, [name.capitalize() for name in summary["split"]])
    ax.set_ylabel("Number of compounds")
    ax.set_title("Class composition after scaffold split")
    ax.legend(loc="upper right")
    for position, row in summary.iterrows():
        ax.text(
            position,
            row["n_compounds"] + 18,
            f"n={int(row['n_compounds'])}\nactive={row['active_fraction']:.1%}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    ax.set_ylim(0, summary["n_compounds"].max() * 1.18)
    save_figure(fig, output_dir / "split_class_balance")


def run(raw_csv: Path, processed_dir: Path, results_dir: Path) -> dict[str, object]:
    processed_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(raw_csv)
    cleaned, audit, conflict_rows = clean_and_audit(raw)
    cleaned["split"] = scaffold_split(cleaned)
    checks = verify_split(cleaned)
    summary = build_split_summary(cleaned)

    key_columns = [
        "source_row",
        "CID",
        "mol",
        "canonical_smiles",
        "scaffold",
        "Class",
        "pIC50",
        "split",
    ]
    available_columns = [column for column in key_columns if column in cleaned.columns]
    cleaned[available_columns].to_csv(
        processed_dir / "bace_scaffold_split.csv", index=False
    )
    for split_name in SPLIT_NAMES:
        cleaned.loc[cleaned["split"] == split_name, available_columns].to_csv(
            processed_dir / f"{split_name}.csv", index=False
        )
    conflict_rows.to_csv(results_dir / "conflicting_labels.csv", index=False)

    scaffold_summary = (
        cleaned.groupby("scaffold", dropna=False)
        .agg(
            n_compounds=("canonical_smiles", "size"),
            n_active=("Class", "sum"),
            active_fraction=("Class", "mean"),
            split=("split", "first"),
        )
        .reset_index()
        .sort_values(["n_compounds", "scaffold"], ascending=[False, True])
    )
    scaffold_summary.to_csv(results_dir / "scaffold_summary.csv", index=False)
    summary.to_csv(results_dir / "split_summary.csv", index=False)

    project_root = Path(__file__).resolve().parents[1]
    try:
        raw_csv_display = str(raw_csv.resolve().relative_to(project_root))
    except ValueError:
        raw_csv_display = raw_csv.name

    audit.update(
        {
            "raw_csv": raw_csv_display,
            "raw_sha256": sha256_file(raw_csv),
            "rdkit_version": rdBase.rdkitVersion,
            "split_method": "label-balanced Bemis-Murcko scaffold group allocation",
            "requested_split_fractions": dict(zip(SPLIT_NAMES, SPLIT_FRACTIONS)),
            "split_checks": checks,
        }
    )
    with (results_dir / "data_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)

    plot_class_balance(cleaned, figures_dir)
    plot_scaffold_distribution(cleaned, figures_dir)
    plot_split_balance(summary, figures_dir)
    return {"audit": audit, "split_summary": summary.to_dict(orient="records")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--raw-csv", type=Path, default=root / "data/raw/bace.csv")
    parser.add_argument("--processed-dir", type=Path, default=root / "data/processed")
    parser.add_argument("--results-dir", type=Path, default=root / "results")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = run(arguments.raw_csv, arguments.processed_dir, arguments.results_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
