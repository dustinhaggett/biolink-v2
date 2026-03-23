"""
Data loading and preprocessing utilities for CTD chemical–disease data.
"""

import random
import pandas as pd
from sklearn.model_selection import train_test_split


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    import numpy as np
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_ctd_data(filepath: str) -> pd.DataFrame:
    """
    Load CTD chemicals–diseases data from TSV file.
    
    Args:
        filepath: Path to CTD_chemicals_diseases.tsv.gz file
        
    Returns:
        DataFrame with CTD data
    """
    cols = [
        "ChemicalName",
        "ChemicalID",
        "CasRN",
        "DiseaseName",
        "DiseaseID",
        "DirectEvidence",
        "InferenceGeneSymbol",
        "InferenceScore",
        "OmimIDs",
        "PubMedIDs"
    ]
    
    ctd = pd.read_csv(
        filepath,
        sep="\t",
        comment="#",
        header=None,
        names=cols,
        dtype=str,
        low_memory=False
    )
    
    return ctd


def create_pairs(ctd: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Create labeled drug–disease pairs from CTD data.
    
    Args:
        ctd: CTD DataFrame
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with columns: ChemicalName, DiseaseName, label
    """
    set_seed(seed)
    
    # Filter for therapeutic relationships
    thera = ctd[ctd["DirectEvidence"] == "therapeutic"].copy()
    
    # Keep only unique chemical-disease pairs
    pos_pairs = thera[["ChemicalName", "DiseaseName"]].drop_duplicates()
    
    # Generate negative samples
    chemicals = pos_pairs["ChemicalName"].unique().tolist()
    diseases = pos_pairs["DiseaseName"].unique().tolist()
    pos_set = set(zip(pos_pairs["ChemicalName"], pos_pairs["DiseaseName"]))
    
    num_pos = len(pos_pairs)
    num_neg = num_pos  # 1:1 ratio
    
    neg_samples = []
    attempts = 0
    max_attempts = num_neg * 10
    
    while len(neg_samples) < num_neg and attempts < max_attempts:
        c = random.choice(chemicals)
        d = random.choice(diseases)
        if (c, d) not in pos_set:
            neg_samples.append((c, d))
        attempts += 1
    
    # Combine positive and negative pairs
    pos_labeled = pos_pairs.copy()
    pos_labeled["label"] = 1
    
    neg_pairs = pd.DataFrame(neg_samples, columns=["ChemicalName", "DiseaseName"])
    neg_labeled = neg_pairs.copy()
    neg_labeled["label"] = 0
    
    pairs_df = pd.concat([pos_labeled, neg_labeled], ignore_index=True)
    pairs_df = pairs_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    
    return pairs_df


def create_splits(
    pairs_df: pd.DataFrame,
    test_size: float = 0.30,
    val_size: float = 0.50,
    seed: int = 42
):
    """
    Create stratified train/validation/test splits.
    
    Args:
        pairs_df: DataFrame with labeled pairs
        test_size: Proportion for test set
        val_size: Proportion of remaining data for validation
        seed: Random seed
        
    Returns:
        Tuple of (train_idx, val_idx, test_idx)
    """
    set_seed(seed)
    
    train_idx, temp_idx = train_test_split(
        pairs_df.index,
        test_size=test_size,
        stratify=pairs_df["label"],
        random_state=seed,
    )
    
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=val_size,
        stratify=pairs_df.loc[temp_idx, "label"],
        random_state=seed,
    )
    
    return train_idx, val_idx, test_idx

