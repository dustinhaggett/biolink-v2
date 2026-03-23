"""Core v1 model loading and batched inference utilities."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from gensim.models import KeyedVectors


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9\-]+")


class MLPClassifier(nn.Module):
    """Exact v1 MLP architecture (no deviations)."""

    def __init__(self, input_dim: int = 800, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class BioLinkModel:
    def __init__(
        self,
        weights_path: str | Path,
        biowordvec_path: str | Path,
        drugs_list_path: str | Path = "data/drugs_list.txt",
    ):
        # Use MPS locally when available, with CPU fallback.
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        # Load exact v1 MLP architecture and checkpoint.
        self.model = MLPClassifier(input_dim=800, hidden_dim=256, dropout=0.3)
        state_dict = torch.load(str(weights_path), map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # Load BioWordVec vectors (200-dim).
        self.word_vectors = KeyedVectors.load_word2vec_format(str(biowordvec_path), binary=True)
        self.embedding_dim = int(self.word_vectors.vector_size)
        if self.embedding_dim != 200:
            raise ValueError(f"Expected BioWordVec dim 200, got {self.embedding_dim}")

        self.drug_names = self._load_drug_names(drugs_list_path)
        # Pre-embed and cache all drugs for fast full ranking.
        self.drug_embeddings = self._embed_texts(self.drug_names)
        if self.drug_embeddings.shape != (7164, 200):
            raise ValueError(
                f"Expected cached drug matrix shape (7164, 200), got {self.drug_embeddings.shape}"
            )

    def _load_drug_names(self, drugs_list_path: str | Path) -> List[str]:
        path = Path(drugs_list_path)
        if not path.exists():
            raise FileNotFoundError(f"Drugs list not found: {path}")
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def _tokenize(self, text: str) -> List[str]:
        if not isinstance(text, str):
            text = str(text)
        return TOKEN_PATTERN.findall(text.lower())

    def _encode_text(self, text: str) -> np.ndarray:
        tokens = self._tokenize(text)
        vectors = [self.word_vectors[token] for token in tokens if token in self.word_vectors.key_to_index]
        if not vectors:
            return np.zeros(self.embedding_dim, dtype=np.float32)
        return np.mean(vectors, axis=0).astype(np.float32)

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        return np.vstack([self._encode_text(text) for text in texts]).astype(np.float32)

    def encode_disease(self, ctd_disease_name: str) -> np.ndarray:
        # BioWordVec encode -> 200-dim vector
        return self._encode_text(ctd_disease_name)

    def encode_drug(self, drug_name: str) -> np.ndarray:
        # BioWordVec encode -> 200-dim vector
        return self._encode_text(drug_name)

    def feature_vector(self, drug_vec: np.ndarray, disease_vec: np.ndarray) -> np.ndarray:
        # [drug, disease, |drug - disease|, drug * disease] -> 800-dim
        # Note: diff is np.abs(drug_vec - disease_vec), matching v1 build_pair_features exactly
        drug_vec = np.asarray(drug_vec, dtype=np.float32)
        disease_vec = np.asarray(disease_vec, dtype=np.float32)
        diff = np.abs(drug_vec - disease_vec)
        prod = drug_vec * disease_vec
        return np.concatenate([drug_vec, disease_vec, diff, prod], axis=0).astype(np.float32)

    def raw_logit(self, drug_name: str, disease_vec: np.ndarray) -> float:
        # Returns pre-sigmoid logit for a single drug
        drug_vec = self.encode_drug(drug_name)
        features = self.feature_vector(drug_vec, disease_vec)
        x = torch.from_numpy(features).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logit = self.model(x).item()
        return float(logit)

    def score_all_drugs(self, disease_vec: np.ndarray) -> List[Tuple[str, float]]:
        # Returns [(drug_name, raw_logit), ...] for all drugs in drugs_list
        # Sorted descending by logit
        disease_vec = np.asarray(disease_vec, dtype=np.float32).reshape(1, self.embedding_dim)
        disease_matrix = np.broadcast_to(disease_vec, self.drug_embeddings.shape)

        # Build all 7,164 x 800 pair features in one batched numpy call.
        features = np.concatenate(
            [
                self.drug_embeddings,
                disease_matrix,
                np.abs(self.drug_embeddings - disease_matrix),
                self.drug_embeddings * disease_matrix,
            ],
            axis=1,
        ).astype(np.float32, copy=False)

        x = torch.from_numpy(features).to(self.device)
        with torch.no_grad():
            logits = self.model(x).detach().cpu().numpy()

        sort_idx = np.argsort(logits)[::-1]
        return [(self.drug_names[i], float(logits[i])) for i in sort_idx]


if __name__ == "__main__":
    model = BioLinkModel(
        weights_path="models/biolink_v1.pt",
        biowordvec_path="data/BioWordVec_PubMed_MIMICIII_d200.vec.bin",
        drugs_list_path="data/drugs_list.txt",
    )
    query_disease = "Hypertension"
    disease_vector = model.encode_disease(query_disease)
    ranked = model.score_all_drugs(disease_vector)

    print(f"Top 10 drugs for '{query_disease}' (raw logits):")
    for rank, (drug_name, logit) in enumerate(ranked[:10], start=1):
        print(f"{rank:2d}. {drug_name} -> {logit:.6f}")
