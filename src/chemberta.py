"""Week 3 frozen ChemBERTa embedding + MLP activity baseline.

Only the frozen train and validation scaffold splits are read. ChemBERTa is
used in evaluation mode with every encoder parameter frozen; pooled embeddings
are cached so repeated MLP experiments do not rerun the transformer. The test
split remains untouched until the final comparison stage.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import platform
import random
import time
from copy import deepcopy
from pathlib import Path

import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from evaluate import classification_metrics


CHECKPOINT = "DeepChem/ChemBERTa-77M-MLM"
SELECTION_SEED = 42
ROBUSTNESS_SEEDS = (11, 22, 33, 44, 55)
MAX_LENGTH = 256
ENCODER_BATCH_SIZE = 32
MLP_BATCH_SIZE = 64
MAX_EPOCHS = 200
PATIENCE = 25

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


class ActivityMLP(nn.Module):
    """Single-hidden-layer classifier trained on frozen embeddings."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def masked_mean_pool(
    hidden_state: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Average non-padding token states, including model special tokens."""

    weights = attention_mask.unsqueeze(-1).to(hidden_state.dtype)
    denominator = weights.sum(dim=1).clamp_min(1.0)
    return (hidden_state * weights).sum(dim=1) / denominator


def choose_encoder_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_frozen_encoder(cache_dir: Path, local_files_only: bool = False):
    """Load the tokenizer and fully frozen encoder without encoding data."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir.resolve()))
    os.environ.setdefault("TRANSFORMERS_CACHE", str((cache_dir / "hub").resolve()))
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        CHECKPOINT, cache_dir=cache_dir, local_files_only=local_files_only
    )
    model = AutoModel.from_pretrained(
        CHECKPOINT, cache_dir=cache_dir, local_files_only=local_files_only
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError("ChemBERTa encoder must remain frozen.")
    device = choose_encoder_device()
    model.to(device)
    model.eval()
    return tokenizer, model, device


def encode_with_frozen_encoder(
    smiles: pd.Series,
    tokenizer,
    model,
    device: torch.device,
    batch_size: int = ENCODER_BATCH_SIZE,
    max_length: int = MAX_LENGTH,
) -> tuple[np.ndarray, dict[str, object]]:
    """Encode SMILES with an already loaded frozen encoder."""

    values = smiles.astype(str).tolist()
    token_lengths = [len(tokenizer(value, add_special_tokens=True)["input_ids"]) for value in values]
    embeddings: list[np.ndarray] = []
    truncated = 0
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            batch_values = values[start : start + batch_size]
            encoded = tokenizer(
                batch_values,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            truncated += sum(length > max_length for length in token_lengths[start : start + batch_size])
            encoded = {key: value.to(device) for key, value in encoded.items()}
            output = model(**encoded)
            pooled = masked_mean_pool(output.last_hidden_state, encoded["attention_mask"])
            embeddings.append(pooled.cpu().numpy().astype(np.float32))

    matrix = np.concatenate(embeddings, axis=0)
    metadata = {
        "checkpoint": CHECKPOINT,
        "checkpoint_commit": getattr(model.config, "_commit_hash", None),
        "encoder_class": model.__class__.__name__,
        "hidden_size": int(model.config.hidden_size),
        "pooling": "attention_masked_mean",
        "max_length": max_length,
        "maximum_observed_token_length": int(max(token_lengths)),
        "truncated_molecules": int(truncated),
        "device": str(device),
        "encoder_trainable_parameters": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
    }
    return matrix, metadata


def extract_embeddings(
    smiles: pd.Series,
    cache_dir: Path,
    batch_size: int = ENCODER_BATCH_SIZE,
    max_length: int = MAX_LENGTH,
) -> tuple[np.ndarray, dict[str, object]]:
    """Encode canonical SMILES with a completely frozen ChemBERTa encoder."""

    tokenizer, model, device = load_frozen_encoder(cache_dir)
    return encode_with_frozen_encoder(
        smiles, tokenizer, model, device, batch_size=batch_size, max_length=max_length
    )


def save_embedding_cache(
    path: Path,
    matrix: np.ndarray,
    labels: np.ndarray,
    smiles: pd.Series,
    split: str,
    metadata: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X=matrix,
        y=labels.astype(np.int8),
        smiles=smiles.astype(str).to_numpy(dtype=str),
    )
    payload = {**metadata, "split": split, "rows": int(len(matrix)), "dtype": str(matrix.dtype)}
    with path.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_or_create_embeddings(
    frame: pd.DataFrame,
    split: str,
    features_dir: Path,
    model_cache: Path,
    reuse: bool,
) -> tuple[np.ndarray, dict[str, object]]:
    path = features_dir / f"chemberta_77m_mlm_{split}.npz"
    metadata_path = path.with_suffix(".json")
    if reuse and path.exists() and metadata_path.exists():
        cached = np.load(path)
        if not np.array_equal(
            cached["smiles"], frame["canonical_smiles"].astype(str).to_numpy(dtype=str)
        ):
            raise ValueError(f"Cached {split} embedding rows do not match the current CSV.")
        with metadata_path.open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        return cached["X"].astype(np.float32), metadata

    matrix, metadata = extract_embeddings(frame["canonical_smiles"], model_cache)
    save_embedding_cache(
        path,
        matrix,
        frame["Class"].to_numpy(dtype=np.int8),
        frame["canonical_smiles"],
        split,
        metadata,
    )
    return matrix, metadata


def parameter_grid() -> list[dict[str, float | int]]:
    keys = ("hidden_dim", "dropout", "learning_rate", "weight_decay")
    values = ([64, 128], [0.1, 0.3], [1e-4, 3e-4], [1e-4])
    return [dict(zip(keys, combination)) for combination in itertools.product(*values)]


def probabilities(model: ActivityMLP, matrix: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        logits = model(torch.from_numpy(matrix).float())
        return torch.sigmoid(logits).numpy()


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    params: dict[str, float | int],
    seed: int,
) -> tuple[ActivityMLP, dict[str, object], list[dict[str, float]]]:
    """Train on train only; use scaffold validation for early stopping."""

    set_seed(seed)
    model = ActivityMLP(
        input_dim=X_train.shape[1],
        hidden_dim=int(params["hidden_dim"]),
        dropout=float(params["dropout"]),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
    )
    positive_weight = float(np.sum(y_train == 0) / np.sum(y_train == 1))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight))
    generator = torch.Generator().manual_seed(seed)
    dataset = TensorDataset(
        torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()
    )
    loader = DataLoader(
        dataset,
        batch_size=MLP_BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, object] | None = None
    best_pr_auc = -np.inf
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for batch_features, batch_labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_features), batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch_features)
        valid_probabilities = probabilities(model, X_valid)
        metrics = classification_metrics(y_valid, valid_probabilities)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": total_loss / len(dataset),
                "validation_pr_auc": metrics["pr_auc"],
                "validation_roc_auc": metrics["roc_auc"],
            }
        )
        if metrics["pr_auc"] > best_pr_auc + 1e-7:
            best_pr_auc = metrics["pr_auc"]
            best_state = deepcopy(model.state_dict())
            best_metrics = {**metrics, "best_epoch": epoch}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= PATIENCE:
            break

    if best_state is None or best_metrics is None:
        raise RuntimeError("MLP training did not produce a valid checkpoint.")
    model.load_state_dict(best_state)
    return model, best_metrics, history


def save_figure(fig: plt.Figure, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_roc(y_true: np.ndarray, scores: np.ndarray, output_dir: Path) -> None:
    metrics = classification_metrics(y_true, scores)
    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, scores)
    fig, ax = plt.subplots(figsize=(3.50, 2.90), constrained_layout=True)
    ax.plot(false_positive_rate, true_positive_rate, color="#E07B54", linewidth=1.8,
            label=f"ChemBERTa MLP (AUC={metrics['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], color="#9AA0A6", linestyle="--", linewidth=1.0, label="Chance")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="False-positive rate", ylabel="True-positive rate")
    ax.set_title("Scaffold-validation ROC curve")
    ax.legend(loc="lower right")
    save_figure(fig, output_dir / "chemberta_validation_roc")


def plot_precision_recall(y_true: np.ndarray, scores: np.ndarray, output_dir: Path) -> None:
    metrics = classification_metrics(y_true, scores)
    precision, recall, _ = precision_recall_curve(y_true, scores)
    prevalence = float(np.mean(y_true))
    fig, ax = plt.subplots(figsize=(3.50, 2.90), constrained_layout=True)
    ax.plot(recall, precision, color="#E07B54", linewidth=1.8,
            label=f"ChemBERTa MLP (AP={metrics['pr_auc']:.3f})")
    ax.axhline(prevalence, color="#9AA0A6", linestyle="--", linewidth=1.0,
               label=f"Prevalence ({prevalence:.3f})")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Recall", ylabel="Precision")
    ax.set_title("Validation precision–recall curve")
    ax.legend(loc="lower left")
    save_figure(fig, output_dir / "chemberta_validation_precision_recall")


def plot_seed_robustness(seed_metrics: pd.DataFrame, output_dir: Path) -> None:
    columns = ["roc_auc", "pr_auc"]
    means = seed_metrics[columns].mean().to_numpy()
    standard_deviations = seed_metrics[columns].std(ddof=1).to_numpy()
    x = np.arange(2)
    lower = max(0.0, float(np.min(means - standard_deviations) - 0.03))
    upper = min(1.0, float(np.max(means + standard_deviations) + 0.03))
    fig, ax = plt.subplots(figsize=(3.50, 2.90), constrained_layout=True)
    ax.errorbar(x, means, yerr=standard_deviations, fmt="o", markersize=6,
                color="#E07B54", ecolor="#E07B54", elinewidth=1.2, capsize=4)
    ax.set_xticks(x, ["ROC-AUC", "PR-AUC"])
    ax.set_ylabel("Validation score")
    ax.set_ylim(lower, upper)
    ax.set_title("ChemBERTa MLP stability across seeds")
    ax.text(0.98, 0.05, "Mean ± one SD; n=5 seeds", transform=ax.transAxes,
            ha="right", va="bottom", color="#4A4A4A")
    save_figure(fig, output_dir / "chemberta_validation_seed_stability")


def run(project_root: Path, reuse_embeddings: bool = True) -> dict[str, object]:
    train_path = project_root / "data/processed/train.csv"
    valid_path = project_root / "data/processed/valid.csv"
    results_dir = project_root / "results"
    figures_dir = results_dir / "figures"
    models_dir = project_root / "models"
    features_dir = project_root / "data/processed/features"
    model_cache = models_dir / "huggingface"
    for directory in (results_dir, figures_dir, models_dir, features_dir, model_cache):
        directory.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(train_path)
    valid = pd.read_csv(valid_path)
    required = {"canonical_smiles", "Class", "scaffold"}
    for name, frame in (("train", train), ("valid", valid)):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")
    if not set(train["scaffold"].fillna("")).isdisjoint(set(valid["scaffold"].fillna(""))):
        raise AssertionError("Train/validation scaffold leakage detected.")

    X_train_raw, train_embedding_metadata = load_or_create_embeddings(
        train, "train", features_dir, model_cache, reuse_embeddings
    )
    X_valid_raw, valid_embedding_metadata = load_or_create_embeddings(
        valid, "valid", features_dir, model_cache, reuse_embeddings
    )
    y_train = train["Class"].to_numpy(dtype=np.int8)
    y_valid = valid["Class"].to_numpy(dtype=np.int8)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_valid = scaler.transform(X_valid_raw).astype(np.float32)

    search_rows: list[dict[str, object]] = []
    for params in parameter_grid():
        started = time.perf_counter()
        _, metrics, _ = train_mlp(X_train, y_train, X_valid, y_valid, params, SELECTION_SEED)
        search_rows.append({**params, **metrics, "fit_seconds": time.perf_counter() - started})
    search = pd.DataFrame(search_rows).sort_values(
        ["pr_auc", "roc_auc", "brier_score"], ascending=[False, False, True]
    )
    search.insert(0, "rank", np.arange(1, len(search) + 1))
    search.to_csv(results_dir / "chemberta_hyperparameter_search.csv", index=False)
    best_params = {
        "hidden_dim": int(search.iloc[0]["hidden_dim"]),
        "dropout": float(search.iloc[0]["dropout"]),
        "learning_rate": float(search.iloc[0]["learning_rate"]),
        "weight_decay": float(search.iloc[0]["weight_decay"]),
    }

    final_model, canonical_metrics, history = train_mlp(
        X_train, y_train, X_valid, y_valid, best_params, SELECTION_SEED
    )
    valid_probabilities = probabilities(final_model, X_valid)
    canonical_metrics = {
        **classification_metrics(y_valid, valid_probabilities),
        "best_epoch": int(canonical_metrics["best_epoch"]),
    }
    pd.DataFrame(history).to_csv(results_dir / "chemberta_training_history.csv", index=False)

    robustness_rows: list[dict[str, object]] = []
    for seed in ROBUSTNESS_SEEDS:
        model, metrics, _ = train_mlp(X_train, y_train, X_valid, y_valid, best_params, seed)
        seed_probabilities = probabilities(model, X_valid)
        row = {
            "model": "Frozen ChemBERTa MLP",
            "split": "valid",
            "seed": seed,
            **classification_metrics(y_valid, seed_probabilities),
            "best_epoch": int(metrics["best_epoch"]),
        }
        robustness_rows.append(row)
    robustness = pd.DataFrame(robustness_rows)

    metrics_path = results_dir / "metrics.csv"
    metric_table = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    if not metric_table.empty:
        metric_table = metric_table[metric_table["model"] != "Frozen ChemBERTa MLP"]
    additions = pd.DataFrame(
        [
            {
                "model": "Frozen ChemBERTa MLP",
                "split": "valid",
                "seed": SELECTION_SEED,
                **{key: value for key, value in canonical_metrics.items() if key != "best_epoch"},
            },
            *[
                {key: value for key, value in row.items() if key != "best_epoch"}
                for row in robustness_rows
            ],
        ]
    )
    pd.concat([metric_table, additions], ignore_index=True).to_csv(metrics_path, index=False)

    predictions = valid[["CID", "canonical_smiles", "scaffold", "Class"]].copy()
    predictions["probability_active"] = valid_probabilities
    predictions.to_csv(results_dir / "chemberta_validation_predictions.csv", index=False)

    model_path = models_dir / "chemberta_mlp.pt"
    torch.save(
        {
            "state_dict": final_model.state_dict(),
            "input_dim": int(X_train.shape[1]),
            "params": best_params,
            "checkpoint": CHECKPOINT,
            "pooling": "attention_masked_mean",
        },
        model_path,
    )
    scaler_path = models_dir / "chemberta_embedding_scaler.joblib"
    joblib.dump(scaler, scaler_path)

    summary = {
        "scope": "Week 3 train/validation only; test.csv was not read",
        "selection_rule": "maximize validation PR-AUC, then ROC-AUC, then minimize Brier score",
        "selection_seed": SELECTION_SEED,
        "robustness_seeds": list(ROBUSTNESS_SEEDS),
        "encoder": train_embedding_metadata,
        "validation_encoder": valid_embedding_metadata,
        "best_params": best_params,
        "train_rows": int(len(train)),
        "valid_rows": int(len(valid)),
        "train_csv_sha256": file_sha256(train_path),
        "valid_csv_sha256": file_sha256(valid_path),
        "canonical_validation_metrics": canonical_metrics,
        "five_seed_validation_mean": {
            key: float(robustness[key].mean())
            for key in ("roc_auc", "pr_auc", "brier_score", "f1", "balanced_accuracy", "mcc")
        },
        "five_seed_validation_std": {
            key: float(robustness[key].std(ddof=1))
            for key in ("roc_auc", "pr_auc", "brier_score", "f1", "balanced_accuracy", "mcc")
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
        },
        "model_artifact": str(model_path.relative_to(project_root)),
        "scaler_artifact": str(scaler_path.relative_to(project_root)),
    }
    with (results_dir / "chemberta_baseline_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    plot_roc(y_valid, valid_probabilities, figures_dir)
    plot_precision_recall(y_valid, valid_probabilities, figures_dir)
    plot_seed_robustness(robustness, figures_dir)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--refresh-embeddings",
        action="store_true",
        help="Ignore compatible local embedding caches and rerun the frozen encoder.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    output = run(arguments.project_root.resolve(), reuse_embeddings=not arguments.refresh_embeddings)
    print(json.dumps(output, indent=2))
