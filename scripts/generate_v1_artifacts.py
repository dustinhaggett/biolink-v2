"""
Step 0: Generate v1 artifacts needed for BioLink v2.

Run this once before building anything else. Requires:
  - CTD_chemicals_diseases.tsv.gz in data/
  - BioWordVec_PubMed_MIMICIII_d200.vec.bin in data/
  - v1 source in v1_source/biolink-ctd-drug-disease-main/

Outputs (all written to data/ and models/):
  - data/val_logits.npy
  - data/val_labels.npy
  - data/drugs_list.txt
  - data/diseases_list.txt
  - models/biolink_v1.pt

Usage:
    python scripts/generate_v1_artifacts.py \
        --ctd data/CTD_chemicals_diseases.tsv.gz \
        --biowordvec data/BioWordVec_PubMed_MIMICIII_d200.vec.bin
"""

import argparse
import sys
import os
import numpy as np
import torch

# Add v1 source to path
V1_SRC = os.path.join(os.path.dirname(__file__), "..", "v1_source", "biolink-ctd-drug-disease-main")
sys.path.insert(0, V1_SRC)

from src.data_loader import load_ctd_data, create_pairs, create_splits, set_seed
from src.embeddings import BioWordVecEmbedder, build_pair_features
from src.models import MLPClassifier, train_model


def main(ctd_path: str, biowordvec_path: str):
    set_seed(42)
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    print("Loading CTD data...")
    ctd = load_ctd_data(ctd_path)
    pairs_df = create_pairs(ctd, seed=42)

    print(f"Total pairs: {len(pairs_df)} ({pairs_df['label'].sum()} positive)")

    train_idx, val_idx, test_idx = create_splits(pairs_df, test_size=0.30, val_size=0.50, seed=42)

    print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

    # Save drug and disease lists (unique names from full dataset)
    drugs = sorted(pairs_df["ChemicalName"].unique().tolist())
    diseases = sorted(pairs_df["DiseaseName"].unique().tolist())

    with open("data/drugs_list.txt", "w") as f:
        f.write("\n".join(drugs))
    print(f"Saved data/drugs_list.txt ({len(drugs)} drugs)")

    with open("data/diseases_list.txt", "w") as f:
        f.write("\n".join(diseases))
    print(f"Saved data/diseases_list.txt ({len(diseases)} diseases)")

    print("Loading BioWordVec (this takes ~30s)...")
    embedder = BioWordVecEmbedder(biowordvec_path)

    print("Embedding all pairs...")
    all_chems = pairs_df["ChemicalName"].tolist()
    all_dis = pairs_df["DiseaseName"].tolist()
    all_labels = pairs_df["label"].values.astype(np.float32)

    chem_embs = embedder.embed_batch(all_chems)
    dis_embs = embedder.embed_batch(all_dis)
    X_all = build_pair_features(chem_embs, dis_embs)

    X_train = X_all[train_idx]
    y_train = all_labels[train_idx]
    X_val = X_all[val_idx]
    y_val = all_labels[val_idx]

    print("Training model...")
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    model = train_model(
        X_train, y_train, X_val, y_val,
        input_dim=X_train.shape[1],
        num_epochs=10,
        batch_size=256,
        lr=1e-3,
        device=device,
        verbose=True,
    )

    # Save model weights
    torch.save(model.state_dict(), "models/biolink_v1.pt")
    print("Saved models/biolink_v1.pt")

    # Save val logits and labels for temperature scaling
    model.eval()
    X_val_t = torch.from_numpy(X_val).float().to(device)
    with torch.no_grad():
        val_logits = model(X_val_t).cpu().numpy()

    np.save("data/val_logits.npy", val_logits)
    np.save("data/val_labels.npy", y_val)
    print(f"Saved data/val_logits.npy ({len(val_logits)} samples)")
    print(f"Saved data/val_labels.npy ({len(y_val)} samples)")

    # Quick sanity check
    from sklearn.metrics import roc_auc_score
    val_probs = 1.0 / (1.0 + np.exp(-val_logits))
    auc = roc_auc_score(y_val, val_probs)
    print(f"\nVal AUC: {auc:.4f} (expect ~0.947)")
    print("\nStep 0 complete. All artifacts saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ctd", required=True, help="Path to CTD_chemicals_diseases.tsv.gz")
    parser.add_argument("--biowordvec", required=True, help="Path to BioWordVec .bin file")
    args = parser.parse_args()
    main(args.ctd, args.biowordvec)
