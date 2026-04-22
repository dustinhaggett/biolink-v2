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
        diseases_list_path: str | Path | None = None,
    ):
        # Use MPS locally when available, with CPU fallback.
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        # Load exact v1 MLP architecture and checkpoint.
        self.model = MLPClassifier(input_dim=800, hidden_dim=256, dropout=0.3)
        state_dict = torch.load(str(weights_path), map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.embedding_dim = 200

        # Try loading cached embeddings first (avoids 13GB BioWordVec file).
        data_dir = Path(drugs_list_path).parent
        drug_cache = data_dir / "drug_embeddings.npy"
        disease_cache = data_dir / "disease_embeddings.npy"
        use_cache = drug_cache.exists()

        if use_cache:
            self.word_vectors = None
            self.drug_names = self._load_names(drugs_list_path)
            self.drug_embeddings = np.load(str(drug_cache))

            self.disease_names: List[str] = []
            self.disease_embeddings: np.ndarray | None = None
            if diseases_list_path and disease_cache.exists():
                self.disease_names = self._load_names(diseases_list_path)
                self.disease_embeddings = np.load(str(disease_cache))
        else:
            # Full BioWordVec path (local development).
            self.word_vectors = KeyedVectors.load_word2vec_format(str(biowordvec_path), binary=True)
            if int(self.word_vectors.vector_size) != 200:
                raise ValueError(f"Expected BioWordVec dim 200, got {self.word_vectors.vector_size}")

            self.drug_names = self._load_names(drugs_list_path)
            self.drug_embeddings = self._embed_texts(self.drug_names)

            self.disease_names: List[str] = []
            self.disease_embeddings: np.ndarray | None = None
            if diseases_list_path:
                diseases_path = Path(diseases_list_path)
                if diseases_path.exists():
                    self.disease_names = self._load_names(diseases_list_path)
                    self.disease_embeddings = self._embed_texts(self.disease_names)

    def _load_names(self, path: str | Path) -> List[str]:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"List file not found: {p}")
        return [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]

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
        if self.word_vectors is None and self.disease_embeddings is not None:
            try:
                idx = self.disease_names.index(ctd_disease_name)
                return self.disease_embeddings[idx]
            except ValueError:
                return np.zeros(self.embedding_dim, dtype=np.float32)
        return self._encode_text(ctd_disease_name)

    def encode_drug(self, drug_name: str) -> np.ndarray:
        if self.word_vectors is None:
            try:
                idx = self.drug_names.index(drug_name)
                return self.drug_embeddings[idx]
            except ValueError:
                return np.zeros(self.embedding_dim, dtype=np.float32)
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

    def score_all_diseases(self, drug_vec: np.ndarray) -> List[Tuple[str, float]]:
        """Returns [(disease_name, raw_logit), ...] sorted descending."""
        if self.disease_embeddings is None or len(self.disease_names) == 0:
            raise RuntimeError("Disease embeddings not loaded. Pass diseases_list_path to constructor.")

        drug_vec = np.asarray(drug_vec, dtype=np.float32).reshape(1, self.embedding_dim)
        drug_matrix = np.broadcast_to(drug_vec, self.disease_embeddings.shape)

        # Feature order: [drug, disease, |drug-disease|, drug*disease]
        features = np.concatenate(
            [
                drug_matrix,
                self.disease_embeddings,
                np.abs(drug_matrix - self.disease_embeddings),
                drug_matrix * self.disease_embeddings,
            ],
            axis=1,
        ).astype(np.float32, copy=False)

        x = torch.from_numpy(features).to(self.device)
        with torch.no_grad():
            logits = self.model(x).detach().cpu().numpy()

        sort_idx = np.argsort(logits)[::-1]
        return [(self.disease_names[i], float(logits[i])) for i in sort_idx]

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
