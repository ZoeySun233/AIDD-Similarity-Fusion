import sys
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chemberta import ActivityMLP, masked_mean_pool  # noqa: E402


class Week3ChemBERTaTests(unittest.TestCase):
    def test_masked_mean_pool_ignores_padding(self):
        hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]]])
        mask = torch.tensor([[1, 1, 0]])
        pooled = masked_mean_pool(hidden, mask)
        torch.testing.assert_close(pooled, torch.tensor([[2.0, 3.0]]))

    def test_embedding_caches_align_with_split_rows(self):
        for split in ("train", "valid"):
            csv = pd.read_csv(ROOT / f"data/processed/{split}.csv")
            cache = np.load(
                ROOT / f"data/processed/features/chemberta_77m_mlm_{split}.npz"
            )
            self.assertEqual(cache["X"].shape[0], len(csv))
            self.assertEqual(cache["X"].shape[1], 384)
            np.testing.assert_array_equal(
                cache["smiles"], csv["canonical_smiles"].astype(str).to_numpy()
            )

    def test_saved_model_reproduces_saved_validation_predictions(self):
        artifact = torch.load(
            ROOT / "models/chemberta_mlp.pt", map_location="cpu", weights_only=True
        )
        model = ActivityMLP(
            input_dim=artifact["input_dim"],
            hidden_dim=artifact["params"]["hidden_dim"],
            dropout=artifact["params"]["dropout"],
        )
        model.load_state_dict(artifact["state_dict"])
        scaler = joblib.load(ROOT / "models/chemberta_embedding_scaler.joblib")
        cached = np.load(
            ROOT / "data/processed/features/chemberta_77m_mlm_valid.npz"
        )
        features = scaler.transform(cached["X"]).astype(np.float32)
        model.eval()
        with torch.inference_mode():
            actual = torch.sigmoid(model(torch.from_numpy(features))).numpy()
        expected = pd.read_csv(
            ROOT / "results/chemberta_validation_predictions.csv"
        )["probability_active"].to_numpy()
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
