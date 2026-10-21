"""Conditional third-representation control: a small directed MPNN.

This model is trained from molecular graphs without a pretrained encoder. It
uses the same outer/inner scaffold manifests, seeds, OOF Platt calibration and
RF reference as the Phase 7 ChemBERTa ablations.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader

from phase7_ablation_runner import (
    ROOT, apply_platt, assert_fit_boundary, fit_platt, score, similarity_labels,
)
from train_base_models_v2 import sha256_file


@dataclass
class Graph:
    atom: torch.Tensor
    bond: torch.Tensor
    src: torch.Tensor
    dst: torch.Tensor
    reverse: torch.Tensor


def one_hot(value: Any, choices: list[Any]) -> list[float]:
    return [float(value == choice) for choice in choices]


def molecular_graph(smiles: str) -> Graph:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("Invalid standardized molecular graph.")
    hybrid = [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
              Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
              Chem.rdchem.HybridizationType.SP3D2]
    chirality = [Chem.rdchem.ChiralType.CHI_UNSPECIFIED, Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                 Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW, Chem.rdchem.ChiralType.CHI_OTHER]
    atom_features = []
    for atom in molecule.GetAtoms():
        atom_features.append([
            atom.GetAtomicNum() / 100, atom.GetDegree() / 6, atom.GetFormalCharge() / 4,
            float(atom.GetIsAromatic()), atom.GetTotalNumHs() / 5, float(atom.IsInRing()),
            atom.GetMass() / 200,
            *one_hot(atom.GetHybridization(), hybrid), *one_hot(atom.GetChiralTag(), chirality),
        ])
    bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                  Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    stereos = [Chem.rdchem.BondStereo.STEREONONE, Chem.rdchem.BondStereo.STEREOZ,
               Chem.rdchem.BondStereo.STEREOE, Chem.rdchem.BondStereo.STEREOANY]
    bond_features, src, dst, reverse = [], [], [], []
    for bond in molecule.GetBonds():
        feature = [*one_hot(bond.GetBondType(), bond_types), float(bond.GetIsConjugated()),
                   float(bond.IsInRing()), *one_hot(bond.GetStereo(), stereos)]
        first = len(src)
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        src.extend((a, b))
        dst.extend((b, a))
        reverse.extend((first + 1, first))
        bond_features.extend((feature, feature))
    return Graph(torch.tensor(atom_features, dtype=torch.float32),
                 torch.tensor(bond_features, dtype=torch.float32).reshape(-1, 10),
                 torch.tensor(src, dtype=torch.long), torch.tensor(dst, dtype=torch.long),
                 torch.tensor(reverse, dtype=torch.long))


def collate_graphs(graphs: list[Graph], labels: np.ndarray) -> tuple[torch.Tensor, ...]:
    atom, bond, src, dst, reverse, atom_to_molecule = [], [], [], [], [], []
    atom_offset = edge_offset = 0
    for molecule_index, graph in enumerate(graphs):
        atom.append(graph.atom)
        bond.append(graph.bond)
        src.append(graph.src + atom_offset)
        dst.append(graph.dst + atom_offset)
        reverse.append(graph.reverse + edge_offset)
        atom_to_molecule.append(torch.full((len(graph.atom),), molecule_index, dtype=torch.long))
        atom_offset += len(graph.atom)
        edge_offset += len(graph.src)
    return (torch.cat(atom), torch.cat(bond), torch.cat(src), torch.cat(dst),
            torch.cat(reverse), torch.cat(atom_to_molecule), torch.tensor(labels, dtype=torch.float32))


class DirectedMPNN(nn.Module):
    def __init__(self, atom_dim: int, bond_dim: int, hidden_dim: int, steps: int, dropout: float) -> None:
        super().__init__()
        self.steps = steps
        self.initial = nn.Linear(atom_dim + bond_dim, hidden_dim)
        self.update = nn.Linear(hidden_dim, hidden_dim)
        self.atom_readout = nn.Linear(atom_dim + hidden_dim, hidden_dim)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim),
                                  nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
        atom, bond, src, dst, reverse, atom_to_molecule, _ = batch
        hidden_dim = self.initial.out_features
        edge_initial = torch.relu(self.initial(torch.cat((atom[src], bond), dim=1)))
        edge_hidden = edge_initial
        for _ in range(self.steps - 1):
            incoming = torch.zeros((len(atom), hidden_dim), dtype=atom.dtype)
            incoming.index_add_(0, dst, edge_hidden)
            message = incoming[src] - edge_hidden[reverse]
            edge_hidden = torch.relu(edge_initial + self.update(message))
        atom_incoming = torch.zeros((len(atom), hidden_dim), dtype=atom.dtype)
        atom_incoming.index_add_(0, dst, edge_hidden)
        atom_hidden = torch.relu(self.atom_readout(torch.cat((atom, atom_incoming), dim=1)))
        count = int(atom_to_molecule.max()) + 1
        molecule_sum = torch.zeros((count, hidden_dim), dtype=atom.dtype)
        molecule_sum.index_add_(0, atom_to_molecule, atom_hidden)
        molecule_count = torch.bincount(atom_to_molecule, minlength=count).clamp_min(1).unsqueeze(1)
        return self.head(molecule_sum / molecule_count).squeeze(-1)


def probabilities(model: DirectedMPNN, graphs: list[Graph], positions: np.ndarray, batch_size: int = 128) -> np.ndarray:
    model.eval()
    output = []
    with torch.inference_mode():
        for start in range(0, len(positions), batch_size):
            group = positions[start:start + batch_size]
            batch = collate_graphs([graphs[index] for index in group], np.zeros(len(group)))
            output.append(torch.sigmoid(model(batch)).numpy())
    return np.concatenate(output)


def fit_predict(
    graphs: list[Graph], labels: np.ndarray, train: np.ndarray, query: np.ndarray,
    spec: dict[str, Any], seed: int, max_epochs: int, patience: int | None,
) -> tuple[np.ndarray, int, float, float]:
    if set(train) & set(query):
        raise ValueError("D-MPNN training and query rows overlap.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    fit_start = time.perf_counter()
    model = DirectedMPNN(graphs[0].atom.shape[1], graphs[0].bond.shape[1],
                         int(spec["hidden_dim"]), int(spec["message_steps"]), float(spec["dropout"]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(spec["learning_rate"]),
                                  weight_decay=float(spec["weight_decay"]))
    positives = max(int(np.sum(labels[train] == 1)), 1)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(int(np.sum(labels[train] == 0)) / positives))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(train.tolist(), batch_size=int(spec["batch_size"]), shuffle=True, generator=generator,
                        collate_fn=lambda positions: collate_graphs([graphs[index] for index in positions], labels[positions]))
    best_state, best_epoch, best_ap, stale = None, 1, -np.inf, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch), batch[-1])
            loss.backward()
            optimizer.step()
        if patience is not None:
            ap = average_precision_score(labels[query], probabilities(model, graphs, query))
            if ap > best_ap + 1e-7:
                best_state, best_epoch, best_ap, stale = copy.deepcopy(model.state_dict()), epoch, ap, 0
            else:
                stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    fitted_seconds = time.perf_counter() - fit_start
    infer_start = time.perf_counter()
    result = probabilities(model, graphs, query)
    return result, best_epoch if patience is not None else max_epochs, fitted_seconds, time.perf_counter() - infer_start


def run(config_path: Path, only_repeat: int | None, only_fold: int | None) -> dict[str, Path]:
    config = yaml.safe_load(config_path.read_text())
    torch.set_num_threads(int(config["encoder"]["torch_threads"]))
    data = pd.read_csv(ROOT / config["inputs"]["standardized_csv"], keep_default_na=False).sort_values("source_row").reset_index(drop=True)
    source_to_position = {int(source): position for position, source in enumerate(data.source_row)}
    labels = data.Class.to_numpy(int)
    start = time.perf_counter()
    graphs = [molecular_graph(value) for value in data.standardized_smiles]
    graph_build_seconds_per_molecule = (time.perf_counter() - start) / len(data)
    fingerprint_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    fingerprints = [fingerprint_generator.GetFingerprint(Chem.MolFromSmiles(value)) for value in data.standardized_smiles]
    rf = pd.read_csv(ROOT / config["inputs"]["calibrated_rf_csv"])
    rf = rf[(rf.model == "morgan_rf") & (rf.split == "outer_test")]
    prediction_rows, metric_rows, search_rows, audit_rows = [], [], [], []
    spec, experiment = config["dmpnn"], config["experiment"]
    for repeat in experiment["repeats"]:
        if only_repeat is not None and repeat != only_repeat:
            continue
        outer_manifest = pd.read_csv(ROOT / config["inputs"]["split_directory"] / f"outer_repeat_{repeat}.csv")
        for outer_fold in experiment["outer_folds"]:
            if only_fold is not None and outer_fold != only_fold:
                continue
            inner_manifest = pd.read_csv(ROOT / config["inputs"]["split_directory"] /
                                         f"inner_repeat_{repeat}_outer_fold_{outer_fold}.csv")
            development = np.asarray([source_to_position[int(value)] for value in inner_manifest.source_row], dtype=int)
            inner_folds = inner_manifest.inner_fold.to_numpy(int)
            outer_sources = outer_manifest.loc[outer_manifest.outer_fold == outer_fold, "source_row"].to_numpy(int)
            outer_test = np.asarray([source_to_position[value] for value in outer_sources], dtype=int)
            if set(development) & set(outer_test) or len(development) + len(outer_test) != len(data):
                raise ValueError("D-MPNN inner/outer partitions are incomplete.")
            test_similarity, cutoff = similarity_labels(inner_folds, development, outer_test, fingerprints,
                                                        float(experiment["low_similarity_development_quantile"]))
            low = test_similarity <= cutoff
            epochs = []
            for inner_fold in experiment["inner_folds"]:
                train, valid = development[inner_folds != inner_fold], development[inner_folds == inner_fold]
                context = {"variant": "dmpnn", "repeat": repeat, "outer_fold": outer_fold,
                           "seed": int(experiment["selection_seed"]), "inner_fold": inner_fold,
                           "candidate_index": 1}
                audit_rows.append(assert_fit_boundary(train, valid, outer_test, data, "inner_parameter_search", context))
                probability, epoch, fit_seconds, infer_seconds = fit_predict(
                    graphs, labels, train, valid, spec, int(experiment["selection_seed"]),
                    int(spec["max_epochs"]), int(spec["patience"]),
                )
                epochs.append(epoch)
                search_rows.append({**context, "params_json": json.dumps(spec, sort_keys=True),
                                    "selected_epoch": epoch, "fit_seconds": fit_seconds,
                                    "inference_seconds": infer_seconds, "selected": True,
                                    **score(labels[valid], probability)})
            selected_epochs = int(np.median(epochs))
            inner_selection_ap = float(np.mean([row["average_precision"] for row in search_rows
                                                if row["repeat"] == repeat and row["outer_fold"] == outer_fold]))
            for seed in experiment["training_seeds"]:
                seed = int(seed)
                oof = np.full(len(development), np.nan)
                fit_seconds_total = 0.0
                for inner_fold in experiment["inner_folds"]:
                    train_mask, valid_mask = inner_folds != inner_fold, inner_folds == inner_fold
                    train, valid = development[train_mask], development[valid_mask]
                    context = {"variant": "dmpnn", "repeat": repeat, "outer_fold": outer_fold,
                               "seed": seed, "inner_fold": inner_fold, "candidate_index": 1}
                    audit_rows.append(assert_fit_boundary(train, valid, outer_test, data, "calibration_oof_fit", context))
                    probability, _, fit_seconds, _ = fit_predict(
                        graphs, labels, train, valid, spec, seed, selected_epochs, None)
                    oof[valid_mask] = probability
                    fit_seconds_total += fit_seconds
                if not np.isfinite(oof).all():
                    raise ValueError("D-MPNN OOF predictions incomplete.")
                platt = fit_platt(labels[development], oof, config["calibration"])
                context = {"variant": "dmpnn", "repeat": repeat, "outer_fold": outer_fold,
                           "seed": seed, "inner_fold": 0, "candidate_index": 1}
                audit_rows.append(assert_fit_boundary(development, outer_test, outer_test, data,
                                                      "outer_full_development_fit", context))
                raw, _, fit_seconds, infer_seconds = fit_predict(
                    graphs, labels, development, outer_test, spec, seed, selected_epochs, None)
                fit_seconds_total += fit_seconds
                calibrated = apply_platt(platt, raw, float(config["calibration"]["probability_clip"]))
                rf_context = rf[(rf.repeat == repeat) & (rf.outer_fold == outer_fold) & (rf.seed == seed)]
                rf_context = rf_context.set_index("source_row").loc[outer_sources]
                if not np.array_equal(rf_context.label.to_numpy(int), labels[outer_test]):
                    raise ValueError("D-MPNN and RF reference labels differ.")
                rf_probability = rf_context.probability_platt.to_numpy(float)
                inference_ms = 1000 * (graph_build_seconds_per_molecule + infer_seconds / len(outer_test))
                for position, raw_value, calibrated_value, similarity, low_flag in zip(
                    outer_test, raw, calibrated, test_similarity, low,
                ):
                    row = data.iloc[position]
                    prediction_rows.append({"CID": row.CID, "source_row": int(row.source_row),
                                            "repeat": repeat, "outer_fold": outer_fold, "seed": seed,
                                            "variant": "dmpnn", "split": "outer_test", "label": int(row.Class),
                                            "raw_probability": float(raw_value),
                                            "probability": float(calibrated_value),
                                            "max_similarity": float(similarity), "low_similarity_cutoff": cutoff,
                                            "low_similarity": bool(low_flag), "scaffold": row.scaffold})
                for stratum, selected in (("all", np.ones(len(outer_test), dtype=bool)),
                                          ("low_similarity", low), ("high_similarity", ~low)):
                    y = labels[outer_test][selected]
                    result = score(y, calibrated[selected])
                    rf_result = score(y, rf_probability[selected])
                    metric_rows.append({"variant": "dmpnn", "repeat": repeat, "outer_fold": outer_fold,
                                        "seed": seed, "stratum": stratum, "n_molecules": len(y),
                                        "n_active": int(y.sum()), "active_fraction": float(y.mean()),
                                        **result, "rf_average_precision": rf_result["average_precision"],
                                        "rf_brier_score": rf_result["brier_score"],
                                        "delta_ap_vs_rf": result["average_precision"] - rf_result["average_precision"],
                                        "delta_brier_vs_rf": result["brier_score"] - rf_result["brier_score"],
                                        "selected_params_json": json.dumps(spec, sort_keys=True),
                                        "selected_epochs": selected_epochs,
                                        "inner_selection_ap": inner_selection_ap,
                                        "fit_seconds_oof_plus_final": fit_seconds_total,
                                        "approx_inference_ms_per_molecule": inference_ms})
            print(f"dmpnn: repeat={repeat} outer_fold={outer_fold} complete", flush=True)
    work = ROOT / "results_v2/phase7_work"
    work.mkdir(parents=True, exist_ok=True)
    paths = {"predictions": work / "dmpnn_predictions.csv.gz", "metrics": work / "dmpnn_metrics.csv",
             "selection": work / "dmpnn_selection.csv", "fit_audit": work / "dmpnn_fit_audit.csv",
             "metadata": work / "dmpnn_metadata.json"}
    pd.DataFrame(prediction_rows).sort_values(["repeat", "outer_fold", "seed", "source_row"]).to_csv(
        paths["predictions"], index=False, compression={"method": "gzip", "mtime": 0})
    pd.DataFrame(metric_rows).sort_values(["repeat", "outer_fold", "seed", "stratum"]).to_csv(paths["metrics"], index=False)
    pd.DataFrame(search_rows).sort_values(["repeat", "outer_fold", "inner_fold"]).to_csv(paths["selection"], index=False)
    pd.DataFrame(audit_rows).to_csv(paths["fit_audit"], index=False)
    paths["metadata"].write_text(json.dumps({
        "phase": "phase7_conditional_dmpnn", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "post_phase5_exploratory", "variant": "dmpnn",
        "config_sha256": sha256_file(config_path), "graph_build_seconds_per_molecule": graph_build_seconds_per_molecule,
        "prediction_rows": len(prediction_rows), "metric_rows": len(metric_rows),
        "fit_audit_rows": len(audit_rows), "script_sha256": sha256_file(Path(__file__)),
    }, indent=2) + "\n")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs_v2/phase7_expert_ablations.yaml")
    parser.add_argument("--repeat", type=int)
    parser.add_argument("--outer-fold", type=int)
    args = parser.parse_args()
    for name, path in run(args.config.resolve(), args.repeat, args.outer_fold).items():
        print(f"{name}: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
