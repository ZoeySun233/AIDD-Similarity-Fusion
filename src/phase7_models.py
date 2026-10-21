"""Single-factor Phase 7 ChemBERTa expert variants.

Every fit receives explicit training and query positions. The caller owns the
nested scaffold manifests and must never pass outer-test rows as training data.
"""

from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel

from chemberta import ActivityMLP


@dataclass
class FeatureBundle:
    mean: np.ndarray
    cls: np.ndarray
    hidden: np.ndarray
    mask: np.ndarray
    augmented_mean: np.ndarray
    input_ids: np.ndarray
    input_mask: np.ndarray
    labels: np.ndarray


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def predict_mlp(model: nn.Module, features: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        return torch.sigmoid(model(torch.from_numpy(features).float())).cpu().numpy()


class AttentionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, center: np.ndarray, scale: np.ndarray) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.zeros(input_dim))
        self.classifier = ActivityMLP(input_dim, hidden_dim, dropout)
        self.register_buffer("center", torch.from_numpy(center).float())
        self.register_buffer("scale", torch.from_numpy(scale).float())

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = (hidden @ self.query) / hidden.shape[-1] ** 0.5
        scores = scores.masked_fill(~mask.bool(), torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        pooled = (hidden * weights.unsqueeze(-1)).sum(dim=1)
        return self.classifier((pooled - self.center) / self.scale)


class FineTunedHead(nn.Module):
    def __init__(self, encoder: nn.Module, input_dim: int, hidden_dim: int,
                 dropout: float, center: np.ndarray, scale: np.ndarray) -> None:
        super().__init__()
        self.encoder = encoder
        self.classifier = ActivityMLP(input_dim, hidden_dim, dropout)
        self.register_buffer("center", torch.from_numpy(center).float())
        self.register_buffer("scale", torch.from_numpy(scale).float())

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Drop only batch-wide padding before self-attention. Non-padding token
        # positions are unchanged, while CPU fine-tuning becomes much faster.
        width = int(attention_mask.sum(dim=1).max())
        input_ids = input_ids[:, :width]
        attention_mask = attention_mask[:, :width]
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        weights = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        return self.classifier((pooled - self.center) / self.scale)


def _binary_loss(labels: np.ndarray) -> nn.Module:
    positive = max(int(np.sum(labels == 1)), 1)
    negative = int(np.sum(labels == 0))
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negative / positive))


def _predict_attention(model: AttentionHead, features: FeatureBundle, positions: np.ndarray, batch_size: int = 128) -> np.ndarray:
    model.eval()
    output = []
    with torch.inference_mode():
        for start in range(0, len(positions), batch_size):
            index = positions[start:start + batch_size]
            hidden = torch.from_numpy(features.hidden[index].astype(np.float32))
            mask = torch.from_numpy(features.mask[index])
            output.append(torch.sigmoid(model(hidden, mask)).numpy())
    return np.concatenate(output)


def _predict_finetuned(model: FineTunedHead, features: FeatureBundle, positions: np.ndarray, batch_size: int = 128) -> np.ndarray:
    model.eval()
    output = []
    with torch.inference_mode():
        for start in range(0, len(positions), batch_size):
            index = positions[start:start + batch_size]
            ids = torch.from_numpy(features.input_ids[index]).long()
            mask = torch.from_numpy(features.input_mask[index]).long()
            output.append(torch.sigmoid(model(ids, mask)).numpy())
    return np.concatenate(output)


def fit_predict(
    variant: str, train: np.ndarray, query: np.ndarray, features: FeatureBundle,
    params: dict[str, Any], seed: int, max_epochs: int, patience: int | None,
    encoder_config: dict[str, Any], finetuning_config: dict[str, Any],
) -> tuple[np.ndarray, int, float, float]:
    """Fit on train only; optionally use query for inner-fold early stopping."""
    if set(train) & set(query):
        raise ValueError("Fit and query rows overlap.")
    set_seed(seed)
    y_train = features.labels[train]
    fit_start = time.perf_counter()
    if variant in {"mean_mlp", "cls_mlp", "mean_logistic", "mean_mlp_aug2"}:
        matrix = features.cls if variant == "cls_mlp" else features.mean
        scaler = StandardScaler().fit(matrix[train])
        X_train = scaler.transform(matrix[train]).astype(np.float32)
        X_query = scaler.transform(matrix[query]).astype(np.float32)
        if variant == "mean_mlp_aug2":
            extra = features.augmented_mean[train].reshape(-1, matrix.shape[1])
            X_train = np.vstack((X_train, scaler.transform(extra).astype(np.float32)))
            y_train = np.concatenate((y_train, np.repeat(y_train, features.augmented_mean.shape[1])))
        if variant == "mean_logistic":
            model = LogisticRegression(C=float(params["C"]), class_weight="balanced", solver="lbfgs", max_iter=1000)
            model.fit(X_train, y_train)
            fitted_seconds = time.perf_counter() - fit_start
            infer_start = time.perf_counter()
            probability = model.predict_proba(X_query)[:, 1]
            return probability, 0, fitted_seconds, time.perf_counter() - infer_start
        model = ActivityMLP(X_train.shape[1], int(params["hidden_dim"]), float(params["dropout"]))
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(params["learning_rate"]), weight_decay=float(params["weight_decay"]))
        loss_fn = _binary_loss(y_train)
        X_tensor = torch.from_numpy(X_train).float()
        y_tensor = torch.from_numpy(y_train).float()
        best_state, best_epoch, best_score, stale = None, 1, -np.inf, 0
        for epoch in range(1, max_epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(X_tensor), y_tensor)
            loss.backward()
            optimizer.step()
            if patience is not None:
                score = average_precision_score(features.labels[query], predict_mlp(model, X_query))
                if score > best_score + 1e-7:
                    best_state, best_epoch, best_score, stale = copy.deepcopy(model.state_dict()), epoch, score, 0
                else:
                    stale += 1
                if stale >= patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        fitted_seconds = time.perf_counter() - fit_start
        infer_start = time.perf_counter()
        probability = predict_mlp(model, X_query)
        return probability, best_epoch if patience is not None else max_epochs, fitted_seconds, time.perf_counter() - infer_start

    if variant == "attention_mlp":
        scaler = StandardScaler().fit(features.mean[train])
        center, scale = scaler.mean_.astype(np.float32), scaler.scale_.astype(np.float32)
        model = AttentionHead(features.mean.shape[1], int(params["hidden_dim"]), float(params["dropout"]), center, scale)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(params["learning_rate"]), weight_decay=float(params["weight_decay"]))
        loss_fn = _binary_loss(y_train)
        generator = torch.Generator().manual_seed(seed)
        dataset = TensorDataset(torch.from_numpy(features.hidden[train].astype(np.float32)),
                                torch.from_numpy(features.mask[train]), torch.from_numpy(y_train).float())
        loader = DataLoader(dataset, batch_size=64, shuffle=True, generator=generator)
        best_state, best_epoch, best_score, stale = None, 1, -np.inf, 0
        for epoch in range(1, max_epochs + 1):
            model.train()
            for hidden, mask, labels in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(hidden, mask), labels)
                loss.backward()
                optimizer.step()
            if patience is not None:
                score = average_precision_score(features.labels[query], _predict_attention(model, features, query))
                if score > best_score + 1e-7:
                    best_state, best_epoch, best_score, stale = copy.deepcopy(model.state_dict()), epoch, score, 0
                else:
                    stale += 1
                if stale >= patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        fitted_seconds = time.perf_counter() - fit_start
        infer_start = time.perf_counter()
        probability = _predict_attention(model, features, query)
        return probability, best_epoch if patience is not None else max_epochs, fitted_seconds, time.perf_counter() - infer_start

    if variant in {"mean_mlp_ft_last0", "mean_mlp_ft_last1", "mean_mlp_ft_last2"}:
        layers = int(variant[-1])
        encoder = AutoModel.from_pretrained(encoder_config["checkpoint"], cache_dir=Path(encoder_config["local_cache"]),
                                            local_files_only=bool(encoder_config["local_files_only"]), add_pooling_layer=False)
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
        if layers:
            for layer in encoder.encoder.layer[-layers:]:
                for parameter in layer.parameters():
                    parameter.requires_grad_(True)
        scaler = StandardScaler().fit(features.mean[train])
        model = FineTunedHead(encoder, features.mean.shape[1], int(finetuning_config["hidden_dim"]),
                              float(finetuning_config["dropout"]), scaler.mean_.astype(np.float32),
                              scaler.scale_.astype(np.float32))
        parameter_groups = [{"params": model.classifier.parameters(), "lr": float(finetuning_config["head_learning_rate"])}]
        if layers:
            parameter_groups.insert(0, {"params": [p for p in encoder.parameters() if p.requires_grad],
                                        "lr": float(finetuning_config["learning_rate"])})
        optimizer = torch.optim.AdamW(parameter_groups, weight_decay=float(finetuning_config["weight_decay"]))
        loss_fn = _binary_loss(y_train)
        # Length-bucketed batches avoid padding every molecule to the longest
        # SMILES in the corpus. Batch order is reshuffled on each epoch.
        lengths = features.input_mask[train].sum(axis=1)
        ordered = train[np.argsort(lengths, kind="stable")]
        batch_size = int(finetuning_config["batch_size"])
        buckets = [ordered[start:start + batch_size] for start in range(0, len(ordered), batch_size)]
        random_batches = np.random.default_rng(seed)
        best_state, best_epoch, best_score, stale = None, 1, -np.inf, 0
        for epoch in range(1, max_epochs + 1):
            model.train()
            model.encoder.eval()
            for layer in encoder.encoder.layer[-layers:] if layers else ():
                layer.train()
            for bucket_index in random_batches.permutation(len(buckets)):
                positions = buckets[bucket_index]
                ids = torch.from_numpy(features.input_ids[positions]).long()
                mask = torch.from_numpy(features.input_mask[positions]).long()
                labels = torch.from_numpy(features.labels[positions]).float()
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(ids, mask), labels)
                loss.backward()
                optimizer.step()
            if patience is not None:
                score = average_precision_score(features.labels[query], _predict_finetuned(model, features, query))
                if score > best_score + 1e-7:
                    best_state, best_epoch, best_score, stale = copy.deepcopy(model.state_dict()), epoch, score, 0
                else:
                    stale += 1
                if stale >= patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        fitted_seconds = time.perf_counter() - fit_start
        infer_start = time.perf_counter()
        probability = _predict_finetuned(model, features, query)
        return probability, best_epoch if patience is not None else max_epochs, fitted_seconds, time.perf_counter() - infer_start

    raise ValueError(f"Unknown Phase 7 variant: {variant}")
