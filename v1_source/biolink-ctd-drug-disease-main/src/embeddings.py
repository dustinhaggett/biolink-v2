"""
Embedding utilities for BioWordVec and Transformer models.
"""

import re
import numpy as np
from typing import List
from gensim.models import KeyedVectors
from sentence_transformers import SentenceTransformer


# Tokenizer for BioWordVec
token_pattern = re.compile(r"[A-Za-z0-9\-]+")


def tokenize(text: str) -> List[str]:
    """Tokenize text for BioWordVec."""
    if not isinstance(text, str):
        text = str(text)
    return token_pattern.findall(text.lower())


class BioWordVecEmbedder:
    """BioWordVec embedding generator."""
    
    def __init__(self, model_path: str):
        """
        Initialize BioWordVec embedder.
        
        Args:
            model_path: Path to BioWordVec .bin file
        """
        self.model = KeyedVectors.load_word2vec_format(model_path, binary=True)
        self.emb_dim = self.model.vector_size
        
    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single text string.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        tokens = tokenize(text)
        vecs = [self.model[w] for w in tokens if w in self.model.key_to_index]
        if not vecs:
            return np.zeros(self.emb_dim, dtype=np.float32)
        return np.mean(vecs, axis=0).astype(np.float32)
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Embed a batch of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            Embedding matrix (N, emb_dim)
        """
        return np.vstack([self.embed_text(t) for t in texts])


class TransformerEmbedder:
    """Transformer sentence embedding generator."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize Transformer embedder.
        
        Args:
            model_name: HuggingFace model name
        """
        self.model = SentenceTransformer(model_name)
        self.emb_dim = self.model.get_sentence_embedding_dimension()
        
    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 64,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Embed a batch of texts.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for encoding
            normalize: Whether to normalize embeddings
            
        Returns:
            Embedding matrix (N, emb_dim)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
        return embeddings.astype(np.float32)


def build_pair_features(
    chem_emb: np.ndarray,
    dis_emb: np.ndarray
) -> np.ndarray:
    """
    Build interaction features from chemical and disease embeddings.
    
    Args:
        chem_emb: Chemical embeddings (N, d)
        dis_emb: Disease embeddings (N, d)
        
    Returns:
        Interaction features (N, 4d): [chem, disease, |chem - disease|, chem * disease]
    """
    diff = np.abs(chem_emb - dis_emb)
    prod = chem_emb * dis_emb
    return np.concatenate([chem_emb, dis_emb, diff, prod], axis=1)

