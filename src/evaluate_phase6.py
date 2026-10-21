"""Phase 6 paired scaffold-level evaluation of frozen Phase 5 predictions.

The prespecified primary comparison is D2 minus C1. Corrected inner-fold
predictions are used only for threshold selection and a dated sensitivity
analysis; they do not replace the frozen outer-test predictions.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import yaml
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, brier_score_loss,
    f1_score, log_loss, matthews_corrcoef, precision_recall_curve,
    roc_auc_score,
)

from fusion_v2 import (
    entropy, fit_full_candidate, paired_context_predictions,
    similarity_features,
)
from train_base_models_v2 import sha256_file


ROOT = Path(__file__).resolve().parents[1]
CORE_METHODS = [
    "A1_calibrated_rf", "B1_calibrated_chemberta", "C0_equal_weight",
    "C1_fixed_weight", "D0_original_similarity_gate", "D1_shrunk_similarity",
    "D2_similarity_entropy", "D3_similarity_entropy_topk",
    "D4_similarity_entropy_topk_variance", "E0_logistic_stacking",
]


def selected_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y, p)
    score = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best = np.flatnonzero(np.isclose(score, score.max(), atol=1e-12))
    return float(thresholds[best[-1]])  # highest threshold on a tie


def top_k_metrics(y: np.ndarray, p: np.ndarray, source: np.ndarray, fraction: float) -> dict[str, float | int]:
    count = min(len(y), max(1, math.ceil(fraction * len(y))))
    order = np.lexsort((source, -p))[:count]
    active_count = int(y[order].sum())
    prevalence = float(y.mean())
    precision = active_count / count
    return {
        "selected_n": count, "selected_active_n": active_count,
        "precision": precision, "recall": active_count / int(y.sum()) if y.sum() else np.nan,
        "enrichment_factor": precision / prevalence if prevalence else np.nan,
    }


def score_context(frame: pd.DataFrame, threshold: float | None) -> dict[str, float | int]:
    y = frame.label.to_numpy(int)
    p = frame.probability.to_numpy(float)
    source = frame.source_row.to_numpy(int)
    result: dict[str, float | int] = {
        "n_molecules": len(y), "n_active": int(y.sum()), "active_fraction": float(y.mean()),
        "average_precision": float(average_precision_score(y, p)) if y.sum() else np.nan,
        "brier_score": float(brier_score_loss(y, p)),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
        "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])),
    }
    for fraction, suffix in ((0.05, "5"), (0.10, "10")):
        top = top_k_metrics(y, p, source, fraction)
        result.update({
            f"top{suffix}_n": top["selected_n"], f"top{suffix}_active_n": top["selected_active_n"],
            f"precision_top{suffix}": top["precision"],
            f"recall_top{suffix}": top["recall"], f"ef{suffix}": top["enrichment_factor"],
        })
    result["f1_threshold"] = np.nan if threshold is None else threshold
    if threshold is None:
        result.update({"f1": np.nan, "mcc": np.nan, "balanced_accuracy": np.nan})
    else:
        label = (p >= threshold).astype(int)
        result.update({
            "f1": float(f1_score(y, label, zero_division=0)),
            "mcc": float(matthews_corrcoef(y, label)),
            "balanced_accuracy": float(balanced_accuracy_score(y, label)),
        })
    return result


def read_outer_predictions(config: dict[str, Any], data: pd.DataFrame) -> pd.DataFrame:
    inputs = config["inputs"]
    fusion = pd.read_csv(ROOT / inputs["phase5_predictions_csv"])
    if set(fusion.split) != {"outer_test"}:
        raise ValueError("Phase 5 fusion table contains unexpected splits.")
    base = pd.read_csv(ROOT / inputs["calibrated_base_predictions_csv"])
    base = base[base.split == "outer_test"].copy()
    base["method"] = base.model.map({"morgan_rf": "A1_calibrated_rf", "frozen_chemberta_mlp": "B1_calibrated_chemberta"})
    base["probability"] = base.probability_platt
    base["alpha_rf"] = np.nan
    base = base[["CID", "source_row", "repeat", "outer_fold", "seed", "method", "split", "label", "probability", "alpha_rf"]]
    fusion = fusion[base.columns]
    table = pd.concat([fusion, base], ignore_index=True)
    annotations = data[["source_row", "scaffold", "standardized_structure_conflict"]]
    table = table.merge(annotations, on="source_row", validate="many_to_one")
    expected = data.set_index("source_row").Class
    if not np.array_equal(table.label.to_numpy(int), expected.loc[table.source_row].to_numpy(int)):
        raise ValueError("Prediction labels differ from standardized data.")
    if table.duplicated(["repeat", "outer_fold", "seed", "method", "source_row"]).any():
        raise ValueError("Duplicate prediction identity.")
    return table


def make_thresholds(inner: pd.DataFrame) -> tuple[dict[tuple[int, int, int, str], float], pd.DataFrame]:
    thresholds: dict[tuple[int, int, int, str], float] = {}
    rows: list[dict[str, Any]] = []
    keys = ["repeat", "outer_fold", "seed", "method"]
    for context, frame in inner.groupby(keys, sort=True):
        if frame.source_row.duplicated().any() or not frame.probability.between(0, 1).all():
            raise ValueError(f"Invalid independent inner probabilities: {context}")
        value = selected_f1_threshold(frame.label.to_numpy(int), frame.probability.to_numpy(float))
        thresholds[context] = value
        rows.append(dict(zip(keys, context), threshold=value, fit_rows=len(frame),
                         fit_positive_fraction=float(frame.label.mean()),
                         fit_source="independent_inner_validation", objective="maximize_f1",
                         evaluation_use="outer_test_only"))
    return thresholds, pd.DataFrame(rows)


def fold_metric_table(outer: pd.DataFrame, thresholds: dict[tuple[int, int, int, str], float]) -> pd.DataFrame:
    keys = ["repeat", "outer_fold", "seed", "method"]
    rows = []
    for context, frame in outer.groupby(keys, sort=True):
        if frame.label.nunique() < 2:
            raise ValueError(f"Outer-test fold has only one class: {context}")
        rows.append({**dict(zip(keys, context)), **score_context(frame, thresholds.get(context))})
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def paired_fold_differences(metrics: pd.DataFrame, comparisons: list[tuple[str, str]]) -> pd.DataFrame:
    rows = []
    score_columns = ("average_precision", "brier_score", "roc_auc", "log_loss", "precision_top5", "precision_top10", "recall_top5", "ef5", "ef10")
    for candidate, baseline in comparisons:
        selected = metrics[metrics.method.isin((candidate, baseline))]
        pivot = selected.pivot(index=["repeat", "outer_fold", "seed"], columns="method", values=list(score_columns))
        if pivot.isna().any().any() or len(pivot) != 30:
            raise ValueError(f"Incomplete paired comparison: {candidate} vs {baseline}")
        for (repeat, fold), seed_rows in pivot.groupby(level=["repeat", "outer_fold"]):
            item = {"repeat": repeat, "outer_fold": fold, "candidate": candidate, "baseline": baseline,
                    "n_seeds": len(seed_rows)}
            for score in score_columns:
                delta = seed_rows[(score, candidate)] - seed_rows[(score, baseline)]
                item[f"delta_{score}"] = float(delta.mean())
                item[f"seed_sd_delta_{score}"] = float(delta.std(ddof=1))
            rows.append(item)
    return pd.DataFrame(rows).sort_values(["candidate", "repeat", "outer_fold"]).reset_index(drop=True)


def scaffold_cluster_bootstrap(
    outer: pd.DataFrame, data: pd.DataFrame, config: dict[str, Any], comparisons: list[tuple[str, str]],
) -> pd.DataFrame:
    spec = config["evaluation"]
    n_reps = int(spec["bootstrap_replicates"])
    rng = np.random.default_rng(int(spec["bootstrap_seed"]))
    scaffolds = sorted(set(data.scaffold.astype(str)))
    scaffold_index = {value: i for i, value in enumerate(scaffolds)}
    methods = sorted({method for pair in comparisons for method in pair})
    indexed: dict[tuple[int, int, int, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for context, frame in outer[outer.method.isin(methods)].groupby(["repeat", "outer_fold", "seed", "method"]):
        indexed[context] = (frame.label.to_numpy(int), frame.probability.to_numpy(float),
                            frame.scaffold.map(scaffold_index).to_numpy(int))
    expected = 2 * 5 * 3 * len(methods)
    if len(indexed) != expected:
        raise ValueError("Bootstrap lacks paired outer contexts.")
    rows = []
    replicate = 0
    attempts = 0
    while replicate < n_reps:
        attempts += 1
        if attempts > n_reps * 2:
            raise RuntimeError("Too many one-class scaffold-bootstrap draws.")
        counts = np.bincount(rng.integers(len(scaffolds), size=len(scaffolds)), minlength=len(scaffolds))
        point: dict[tuple[int, int, int, str], tuple[float, float]] = {}
        invalid = False
        for context, (y, p, cluster) in indexed.items():
            weight = counts[cluster]
            active_weight = int(weight[y == 1].sum())
            inactive_weight = int(weight[y == 0].sum())
            if not active_weight or not inactive_weight:
                invalid = True
                break
            point[context] = (
                float(average_precision_score(y, p, sample_weight=weight)),
                float(np.average((y - p) ** 2, weights=weight)),
            )
        if invalid:
            continue
        replicate += 1
        for candidate, baseline in comparisons:
            fold_ap, fold_brier = [], []
            for repeat in (1, 2):
                for outer_fold in range(1, 6):
                    seed_ap, seed_brier = [], []
                    for seed in (11, 22, 33):
                        candidate_score = point[(repeat, outer_fold, seed, candidate)]
                        baseline_score = point[(repeat, outer_fold, seed, baseline)]
                        seed_ap.append(candidate_score[0] - baseline_score[0])
                        seed_brier.append(candidate_score[1] - baseline_score[1])
                    fold_ap.append(float(np.mean(seed_ap)))
                    fold_brier.append(float(np.mean(seed_brier)))
            rows.append({"bootstrap_replicate": replicate, "candidate": candidate, "baseline": baseline,
                         "delta_average_precision": float(np.mean(fold_ap)),
                         "delta_brier_score": float(np.mean(fold_brier)),
                         "outer_folds_won": int(np.count_nonzero(np.asarray(fold_ap) > 0))})
        if replicate % 250 == 0:
            print(f"scaffold bootstrap: {replicate}/{n_reps}", flush=True)
    return pd.DataFrame(rows)


def scaffold_similarity_frames(data: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[tuple[int, int], pd.DataFrame]]:
    inputs = config["inputs"]
    eval_spec = config["evaluation"]
    source_to_position = {int(value): i for i, value in enumerate(data.source_row)}
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    fingerprints = [generator.GetFingerprint(Chem.MolFromSmiles(smiles)) for smiles in data.standardized_smiles]
    similarity_rows = []
    development_frames: dict[tuple[int, int], pd.DataFrame] = {}
    for repeat in (1, 2):
        outer = pd.read_csv(ROOT / inputs["split_directory"] / f"outer_repeat_{repeat}.csv")[["source_row", "outer_fold"]]
        for fold in range(1, 6):
            inner = pd.read_csv(ROOT / inputs["split_directory"] / f"inner_repeat_{repeat}_outer_fold_{fold}.csv")[["source_row", "inner_fold"]]
            dev_parts = []
            for inner_fold in (1, 2, 3):
                query = inner.loc[inner.inner_fold == inner_fold, "source_row"].to_numpy(int)
                reference = inner.loc[inner.inner_fold != inner_fold, "source_row"].to_numpy(int)
                query_pos = np.asarray([source_to_position[value] for value in query])
                reference_pos = np.asarray([source_to_position[value] for value in reference])
                values = similarity_features(query_pos, reference_pos, fingerprints, data.iloc[reference_pos].Class.to_numpy(int), 5)
                dev_parts.append(pd.DataFrame({"source_row": query, "max_similarity": values[:, 0]}))
            development = pd.concat(dev_parts, ignore_index=True)
            development_frames[(repeat, fold)] = development
            cutoff = float(development.max_similarity.quantile(float(eval_spec["low_similarity_quantile"])))
            test_sources = outer.loc[outer.outer_fold == fold, "source_row"].to_numpy(int)
            dev_sources = outer.loc[outer.outer_fold != fold, "source_row"].to_numpy(int)
            test_pos = np.asarray([source_to_position[value] for value in test_sources])
            dev_pos = np.asarray([source_to_position[value] for value in dev_sources])
            values = similarity_features(test_pos, dev_pos, fingerprints, data.iloc[dev_pos].Class.to_numpy(int), 5)
            similarity_rows.append(pd.DataFrame({
                "repeat": repeat, "outer_fold": fold, "source_row": test_sources,
                "max_similarity": values[:, 0], "topk_mean_similarity": values[:, 1],
                "topk_active_fraction": values[:, 2], "low_similarity_cutoff": cutoff,
                "low_similarity": values[:, 0] <= cutoff,
            }))
    return pd.concat(similarity_rows, ignore_index=True), development_frames


def similarity_strata(outer: pd.DataFrame, similarity: pd.DataFrame, cliffs: pd.DataFrame) -> pd.DataFrame:
    cliff_sources = set(cliffs.loc[cliffs.potency_cliff_ge_1_log10.astype(bool), "source_row_a"]) | set(cliffs.loc[cliffs.potency_cliff_ge_1_log10.astype(bool), "source_row_b"])
    annotated = outer.merge(similarity, on=["repeat", "outer_fold", "source_row"], validate="many_to_one")
    annotated["cliff_in_phase1"] = annotated.source_row.isin(cliff_sources)
    rows = []
    for name, values in (("low_similarity", True), ("high_similarity", False), ("activity_cliff_any", True), ("not_activity_cliff", False), ("conflicting_structure_label", True)):
        column = "low_similarity" if "similarity" in name else ("cliff_in_phase1" if "cliff" in name else "standardized_structure_conflict")
        subset = annotated[annotated[column] == values]
        for context, frame in subset[subset.method.isin(CORE_METHODS)].groupby(["repeat", "outer_fold", "seed", "method"]):
            y, p = frame.label.to_numpy(int), frame.probability.to_numpy(float)
            rows.append({"stratum": name, "repeat": context[0], "outer_fold": context[1], "seed": context[2],
                         "method": context[3], "n_molecules": len(frame), "n_active": int(y.sum()),
                         "active_fraction": float(y.mean()), "average_precision": float(average_precision_score(y, p)) if y.sum() else np.nan,
                         "brier_score": float(brier_score_loss(y, p))})
    return pd.DataFrame(rows)


def gate_diagnostics(outer: pd.DataFrame, params: pd.DataFrame, similarity: pd.DataFrame) -> pd.DataFrame:
    relevant = outer[outer.method.isin(("C1_fixed_weight", "D1_shrunk_similarity", "D2_similarity_entropy"))]
    wide = relevant.pivot(index=["repeat", "outer_fold", "seed", "source_row"], columns="method", values="alpha_rf").reset_index()
    wide = wide.merge(similarity, on=["repeat", "outer_fold", "source_row"], validate="many_to_one")
    rows = []
    for method in ("D1_shrunk_similarity", "D2_similarity_entropy"):
        for context, frame in wide.groupby(["repeat", "outer_fold", "seed"]):
            delta = frame[method].to_numpy() - frame.C1_fixed_weight.to_numpy()
            subset = params[(params.repeat == context[0]) & (params.outer_fold == context[1]) &
                            (params.seed == context[2]) & (params.method == method)]
            if len(subset) != 1:
                raise ValueError(f"Missing unique fitted gate parameters: {context}, {method}")
            rows.append({"repeat": context[0], "outer_fold": context[1], "seed": context[2], "method": method,
                         "lambda": float(subset.iloc[0]["lambda"]),
                         "alpha_fixed": float(frame.C1_fixed_weight.iloc[0]),
                         "alpha_mean": float(frame[method].mean()), "alpha_sd": float(frame[method].std(ddof=1)),
                         "alpha_min": float(frame[method].min()), "alpha_max": float(frame[method].max()),
                         "mean_absolute_deviation_from_fixed": float(np.abs(delta).mean()),
                         "max_absolute_deviation_from_fixed": float(np.abs(delta).max())})
    # Compare which expert makes the smaller per-molecule squared error as a descriptive mechanism check.
    probs = outer[outer.method.isin(("A1_calibrated_rf", "B1_calibrated_chemberta"))].pivot(
        index=["repeat", "outer_fold", "seed", "source_row", "label"], columns="method", values="probability"
    ).reset_index().merge(similarity, on=["repeat", "outer_fold", "source_row"], validate="many_to_one")
    for context, frame in probs.groupby(["repeat", "outer_fold", "seed"]):
        y = frame.label.to_numpy(int)
        advantage = (y - frame.B1_calibrated_chemberta.to_numpy()) ** 2 - (y - frame.A1_calibrated_rf.to_numpy()) ** 2
        correlation = spearmanr(frame.max_similarity.to_numpy(), advantage).statistic
        rows.append({"repeat": context[0], "outer_fold": context[1], "seed": context[2], "method": "expert_relative_brier",
                     "similarity_vs_rf_advantage_spearman": float(correlation),
                     "rf_advantage_mean_low_similarity": float(advantage[frame.low_similarity.to_numpy(bool)].mean()) if frame.low_similarity.any() else np.nan,
                     "rf_advantage_mean_high_similarity": float(advantage[~frame.low_similarity.to_numpy(bool)].mean()) if (~frame.low_similarity).any() else np.nan})
    return pd.DataFrame(rows)


def scaffold_influence(outer: pd.DataFrame, data: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    primary = outer[outer.method.isin(("C1_fixed_weight", "D2_similarity_entropy"))]
    top = data.scaffold.value_counts().head(int(config["evaluation"]["maximum_scaffold_influence_checks"]))
    rows = []
    for scaffold, size in top.items():
        subset = primary[primary.scaffold != scaffold]
        deltas = []
        for context, frame in subset.groupby(["repeat", "outer_fold", "seed"]):
            wide = frame.pivot(index="source_row", columns="method", values="probability")
            labels = frame.drop_duplicates("source_row").set_index("source_row").loc[wide.index, "label"].to_numpy(int)
            if len(np.unique(labels)) < 2:
                continue
            deltas.append({"repeat": context[0], "outer_fold": context[1], "seed": context[2],
                           "delta": float(average_precision_score(labels, wide.D2_similarity_entropy) -
                                          average_precision_score(labels, wide.C1_fixed_weight))})
        differences = pd.DataFrame(deltas).groupby(["repeat", "outer_fold"]).delta.mean()
        rows.append({"excluded_scaffold": scaffold, "global_scaffold_n": int(size),
                     "evaluable_outer_folds": len(differences),
                     "mean_delta_average_precision": float(differences.mean()),
                     "outer_folds_won": int((differences > 0).sum())})
    return pd.DataFrame(rows)


def similarity_permutations(
    data: pd.DataFrame, config: dict[str, Any], similarity: pd.DataFrame,
    development_similarity: dict[tuple[int, int], pd.DataFrame], outer: pd.DataFrame,
) -> pd.DataFrame:
    spec = config["evaluation"]
    phase4 = pd.read_csv(ROOT / config["inputs"]["calibrated_base_predictions_csv"])
    baseline = outer[outer.method == "C1_fixed_weight"]
    rows = []
    gate_config = {"optimization": yaml.safe_load((ROOT / config["inputs"]["phase5_config_yaml"]).read_text())["optimization"]}
    for repeat in (1, 2):
        for fold in range(1, 6):
            dev_sim = development_similarity[(repeat, fold)]
            test_sim = similarity[(similarity.repeat == repeat) & (similarity.outer_fold == fold)]
            for seed in (11, 22, 33):
                paired = paired_context_predictions(phase4, repeat, fold, seed)
                dev = paired[paired.split == "oof_development"].merge(dev_sim, on="source_row", validate="one_to_one")
                test = paired[paired.split == "outer_test"].merge(test_sim[["source_row", "max_similarity"]], on="source_row", validate="one_to_one")
                for frame in (dev, test):
                    frame["entropy_difference"] = entropy(frame.chem_probability.to_numpy()) - entropy(frame.rf_probability.to_numpy())
                fixed = baseline[(baseline.repeat == repeat) & (baseline.outer_fold == fold) & (baseline.seed == seed)]
                fixed = fixed.set_index("source_row").loc[test.source_row]
                fixed_ap = float(average_precision_score(test.label, fixed.probability))
                for permutation in range(1, int(spec["permutation_replicates"]) + 1):
                    random = np.random.default_rng(int(spec["permutation_seed"]) + permutation * 100_000 + repeat * 1_000 + fold * 10 + seed)
                    perm_dev, perm_test = dev.copy(), test.copy()
                    perm_dev["max_similarity"] = random.permutation(perm_dev.max_similarity.to_numpy())
                    perm_test["max_similarity"] = random.permutation(perm_test.max_similarity.to_numpy())
                    for method in spec["permutation_methods"]:
                        probability, alpha, _ = fit_full_candidate(perm_dev, perm_test, method, gate_config)
                        rows.append({"permutation_replicate": permutation, "repeat": repeat, "outer_fold": fold,
                                     "seed": seed, "method": method,
                                     "delta_average_precision_vs_C1": float(average_precision_score(test.label, probability) - fixed_ap),
                                     "mean_alpha_rf": float(np.mean(alpha))})
            print(f"similarity permutations complete: repeat={repeat} outer_fold={fold}", flush=True)
    return pd.DataFrame(rows)


def run(config_path: Path) -> dict[str, Path]:
    config = yaml.safe_load(config_path.read_text())
    inputs, outputs = config["inputs"], config["outputs"]
    data = pd.read_csv(ROOT / inputs["standardized_csv"], keep_default_na=False).sort_values("source_row")
    inner = pd.read_csv(ROOT / outputs["independent_inner_predictions_csv_gz"])
    outer = read_outer_predictions(config, data)
    thresholds, threshold_audit = make_thresholds(inner)
    metrics = fold_metric_table(outer, thresholds)
    comparisons = [tuple(config["evaluation"]["primary_comparison"]), tuple(config["evaluation"]["supporting_comparison"])]
    paired = paired_fold_differences(metrics, comparisons)
    bootstrap = scaffold_cluster_bootstrap(outer, data, config, comparisons)
    similarity, dev_similarity = scaffold_similarity_frames(data, config)
    cliffs = pd.read_csv(ROOT / inputs["phase1_activity_cliffs_csv"])
    strata = similarity_strata(outer, similarity, cliffs)
    gate_params = pd.read_csv(ROOT / inputs["phase5_gate_parameters_csv"])
    gate = gate_diagnostics(outer, gate_params, similarity)
    influence = scaffold_influence(outer, data, config)
    permutations = similarity_permutations(data, config, similarity, dev_similarity, outer)
    tables = {
        "fold_metrics_csv": metrics, "paired_differences_csv": paired, "scaffold_bootstrap_csv": bootstrap,
        "threshold_audit_csv": threshold_audit, "similarity_strata_csv": strata,
        "gate_diagnostics_csv": gate, "scaffold_influence_csv": influence,
        "similarity_permutations_csv": permutations,
    }
    output_paths = {key: ROOT / outputs[key] for key in tables}
    for key, table in tables.items():
        output_paths[key].parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(output_paths[key], index=False)
    metadata_path = ROOT / outputs["evaluation_metadata_json"]
    metadata = {
        "phase": "phase6_statistical_evaluation", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_phase5_primary_results_and_post_phase5_sensitivities",
        "config_sha256": sha256_file(config_path),
        "phase5_predictions_sha256": sha256_file(ROOT / inputs["phase5_predictions_csv"]),
        "phase5_config_sha256": sha256_file(ROOT / inputs["phase5_config_yaml"]),
        "independent_inner_predictions_sha256": sha256_file(ROOT / outputs["independent_inner_predictions_csv_gz"]),
        "average_precision_definition": "sklearn.metrics.average_precision_score",
        "statistical_unit": "seed means within each repeat x outer fold; repeats reuse molecules",
        "bootstrap_unit": "standardized scaffold; shared resample across repeats, seeds and methods",
        "threshold_selection": "maximize F1 on independent inner predictions, apply only to outer test",
        "output_rows": {key: len(table) for key, table in tables.items()},
        "dependency_versions": {
            "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
            "rdkit": Chem.rdBase.rdkitVersion,
        },
        "script_sha256": sha256_file(Path(__file__)),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    output_paths["evaluation_metadata_json"] = metadata_path
    resolved_path = ROOT / outputs["resolved_config_json"]
    resolved_path.write_text(json.dumps(config, indent=2) + "\n")
    output_paths["resolved_config_json"] = resolved_path
    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs_v2/phase6_statistical_evaluation.yaml")
    args = parser.parse_args()
    for name, path in run(args.config.resolve()).items():
        print(f"{name}: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
