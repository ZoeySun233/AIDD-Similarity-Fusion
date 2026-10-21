"""RDKit Morgan fingerprint utilities used by the classical baselines."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator


@dataclass(frozen=True)
class MorganConfig:
    radius: int = 2
    n_bits: int = 2048
    include_chirality: bool = True
    use_bond_types: bool = True


def smiles_sha256(smiles: Iterable[str]) -> str:
    joined = "\n".join(str(value) for value in smiles).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


def morgan_fingerprints(
    smiles: Iterable[str], config: MorganConfig = MorganConfig()
) -> np.ndarray:
    smiles_list = [str(value) for value in smiles]
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=config.radius,
        fpSize=config.n_bits,
        includeChirality=config.include_chirality,
        useBondTypes=config.use_bond_types,
    )
    matrix = np.zeros((len(smiles_list), config.n_bits), dtype=np.uint8)
    invalid: list[int] = []
    for row_index, value in enumerate(smiles_list):
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            invalid.append(row_index)
            continue
        fingerprint = generator.GetFingerprint(molecule)
        DataStructs.ConvertToNumpyArray(fingerprint, matrix[row_index])
    if invalid:
        raise ValueError(f"Invalid SMILES at row indices: {invalid[:10]}")
    return matrix


def save_fingerprint_cache(
    output_path: Path,
    fingerprints: np.ndarray,
    labels: np.ndarray,
    smiles: Iterable[str],
    config: MorganConfig,
    split_name: str,
) -> None:
    smiles_list = [str(value) for value in smiles]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X=fingerprints,
        y=np.asarray(labels, dtype=np.int8),
        smiles=np.asarray(smiles_list, dtype=str),
    )
    metadata = {
        "split": split_name,
        "n_samples": int(fingerprints.shape[0]),
        "n_features": int(fingerprints.shape[1]),
        "dtype": str(fingerprints.dtype),
        "density": float(fingerprints.mean()),
        "smiles_sha256": smiles_sha256(smiles_list),
        "rdkit_version": rdBase.rdkitVersion,
        "config": asdict(config),
    }
    with output_path.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
