"""Build label-free, locally cached ChemBERTa features for Phase 7.

Randomized SMILES are generated for all molecular structures without labels,
but their embeddings may only enter model fits for training rows. Validation
and outer-test predictions always use the standardized canonical SMILES.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from rdkit import Chem
from transformers import AutoModel, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def randomized_smiles(smiles: str, source_row: int, copies: int, base_seed: int) -> list[str]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid standardized molecule: {source_row}")
    canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    result = []
    for index in range(copies):
        random = np.random.default_rng(base_seed + source_row * 1009 + index)
        order = random.permutation(molecule.GetNumAtoms()).tolist()
        variant = Chem.MolToSmiles(Chem.RenumberAtoms(molecule, order), canonical=False, isomericSmiles=True)
        check = Chem.MolFromSmiles(variant)
        if check is None or Chem.MolToSmiles(check, canonical=True, isomericSmiles=True) != canonical:
            raise ValueError(f"Randomized SMILES changed molecular identity: {source_row}")
        result.append(variant)
    return result


def encode(
    values: list[str], tokenizer: Any, encoder: Any, *, max_length: int,
    batch_size: int, return_tokens: bool,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, float, int]:
    pooled: list[np.ndarray] = []
    token_parts: list[np.ndarray] = []
    mask_parts: list[np.ndarray] = []
    maximum = max(len(tokenizer(value, add_special_tokens=True)["input_ids"]) for value in values)
    if maximum > max_length:
        raise ValueError(f"Token truncation would occur: {maximum} > {max_length}")
    start = time.perf_counter()
    encoder.eval()
    with torch.inference_mode():
        for offset in range(0, len(values), batch_size):
            batch = tokenizer(values[offset:offset + batch_size], padding=True, truncation=True,
                              max_length=max_length, return_tensors="pt")
            hidden = encoder(**batch).last_hidden_state
            weights = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            mean = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
            pooled.append(mean.cpu().numpy().astype(np.float32))
            if return_tokens:
                pad = maximum - hidden.shape[1]
                token_parts.append(torch.nn.functional.pad(hidden, (0, 0, 0, pad)).cpu().numpy().astype(np.float16))
                mask_parts.append(torch.nn.functional.pad(batch["attention_mask"], (0, pad)).cpu().numpy().astype(np.uint8))
    seconds = time.perf_counter() - start
    tokens = np.concatenate(token_parts) if return_tokens else None
    masks = np.concatenate(mask_parts) if return_tokens else None
    return np.concatenate(pooled), tokens, masks, seconds, maximum


def run(config_path: Path) -> dict[str, Path]:
    config = yaml.safe_load(config_path.read_text())
    inputs, outputs, spec = config["inputs"], config["outputs"], config["encoder"]
    torch.set_num_threads(int(spec["torch_threads"]))
    data_path = ROOT / inputs["standardized_csv"]
    data = pd.read_csv(data_path, keep_default_na=False).sort_values("source_row").reset_index(drop=True)
    if len(data) != 1513 or data.source_row.duplicated().any():
        raise ValueError("Unexpected standardized BACE identity.")
    mean_path = ROOT / inputs["frozen_mean_cache"]
    with np.load(mean_path) as cached:
        if not np.array_equal(cached["source_row"], data.source_row.to_numpy()):
            raise ValueError("Phase 3 frozen mean cache has different source-row order.")
        original_mean = cached["X"].astype(np.float32)
    token_path, augmented_path = ROOT / outputs["token_cache"], ROOT / outputs["augmented_cache"]
    metadata_path = ROOT / outputs["feature_metadata_json"]
    for path in (token_path, augmented_path, metadata_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = spec["checkpoint"]
    cache = ROOT / spec["local_cache"]
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir=cache, local_files_only=bool(spec["local_files_only"]))
    encoder = AutoModel.from_pretrained(checkpoint, cache_dir=cache, local_files_only=bool(spec["local_files_only"]), add_pooling_layer=False)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    values = data.standardized_smiles.astype(str).tolist()
    canonical_mean, token_hidden, token_mask, canonical_seconds, max_tokens = encode(
        values, tokenizer, encoder, max_length=int(spec["max_length"]),
        batch_size=int(spec["batch_size"]), return_tokens=True,
    )
    difference = float(np.max(np.abs(canonical_mean - original_mean)))
    if difference > 0.0002:
        raise ValueError(f"Frozen encoder disagrees with the Phase 3 mean cache: {difference}")
    cls = token_hidden[:, 0].astype(np.float32)
    np.savez_compressed(token_path, source_row=data.source_row.to_numpy(), mean=original_mean,
                        cls=cls, hidden=token_hidden, mask=token_mask)
    copies = int(config["frozen_head"]["augmentation_n"])
    randomized = [randomized_smiles(value, int(source), copies, int(config["frozen_head"]["augmentation_seed"]))
                  for source, value in zip(data.source_row, values)]
    flat = [value for variants in randomized for value in variants]
    augmented_mean, _, _, augmented_seconds, aug_max_tokens = encode(
        flat, tokenizer, encoder, max_length=int(spec["max_length"]),
        batch_size=int(spec["batch_size"]), return_tokens=False,
    )
    np.savez_compressed(augmented_path, source_row=data.source_row.to_numpy(),
                        smiles=np.asarray(randomized, dtype=str),
                        X=augmented_mean.reshape(len(data), copies, -1).astype(np.float32))
    metadata = {
        "phase": "phase7_label_free_features", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": sha256_file(config_path), "data_sha256": sha256_file(data_path),
        "phase3_mean_cache_sha256": sha256_file(mean_path), "checkpoint": checkpoint,
        "checkpoint_commit": getattr(encoder.config, "_commit_hash", None),
        "source_rows": len(data), "hidden_size": int(encoder.config.hidden_size),
        "canonical_max_tokens": max_tokens, "augmentation_max_tokens": aug_max_tokens,
        "canonical_encoder_seconds": canonical_seconds, "augmentation_encoder_seconds": augmented_seconds,
        "mean_cache_max_absolute_difference": difference, "randomized_smiles_per_molecule": copies,
        "randomized_smiles_duplicate_canonical_n": sum(value == source for source, group in zip(values, randomized) for value in group),
        "training_only_augmentation_policy": "Randomized embeddings may be used only in model-fitting rows; held inner and outer test inputs remain standardized canonical SMILES.",
        "token_cache_sha256": sha256_file(token_path), "augmented_cache_sha256": sha256_file(augmented_path),
        "script_sha256": sha256_file(Path(__file__)),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return {"token_cache": token_path, "augmented_cache": augmented_path, "metadata": metadata_path}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs_v2/phase7_expert_ablations.yaml")
    args = parser.parse_args()
    for name, path in run(args.config.resolve()).items():
        print(f"{name}: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
