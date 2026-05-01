"""
Retrain the BioLink MLP with hard negatives from CTD marker/mechanism evidence.

The v1 model trained on:
  - positives: CTD therapeutic pairs (DirectEvidence == "therapeutic")
  - negatives: random (drug, disease) pairs from the vocabulary

This made the model good at distinguishing "therapeutic" from "random unrelated"
but blind to "harmful/causal" — see docs/POST_PRESENTATION_TODO.md for the
documented failures (cocaine for alcoholism, streptozocin for T2D, scopolamine
for amnesia, cyclosporine for Lyme).

This script retrains with:
  - positives: same therapeutic pairs (39,516)
  - HARD negatives: marker/mechanism pairs in v1 vocab,
        excluding any pair that's ALSO therapeutic (3,817 ambiguous pairs),
        prioritizing pairs with >=2 supporting publications,
        random-sampling weaker (1-pub) pairs to reach 1:1 class ratio.

Output:
    models/biolink_v2_hardneg.pt  — new model weights
    data/val_logits_hardneg.npy   — for refitting calibration
    data/val_labels_hardneg.npy
    data/temperature_hardneg.json — refit calibration

Usage:
    python scripts/retrain_with_hard_negatives.py
    python scripts/retrain_with_hard_negatives.py --epochs 15 --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.calibration import TemperatureScaler  # noqa: E402
from core.model import MLPClassifier  # noqa: E402

# Filenames
CTD_PATH = REPO_ROOT / "data" / "CTD_chemicals_diseases.tsv.gz"
DRUGS_LIST = REPO_ROOT / "data" / "drugs_list.txt"
DISEASES_LIST = REPO_ROOT / "data" / "diseases_list.txt"
DRUG_EMB = REPO_ROOT / "data" / "drug_embeddings.npy"
DISEASE_EMB = REPO_ROOT / "data" / "disease_embeddings.npy"

# Outputs
WEIGHTS_OUT = REPO_ROOT / "models" / "biolink_v2_hardneg.pt"
VAL_LOGITS_OUT = REPO_ROOT / "data" / "val_logits_hardneg.npy"
VAL_LABELS_OUT = REPO_ROOT / "data" / "val_labels_hardneg.npy"
TEMP_OUT = REPO_ROOT / "data" / "temperature_hardneg.json"


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def build_pair_dataset(seed: int, args) -> tuple[pd.DataFrame, dict]:
    """Build (drug, disease, label) tuples for retraining.

    Returns:
        pairs_df: DataFrame with ChemicalName, DiseaseName, label columns
        stats:    dict of dataset statistics for logging
    """
    log("Loading CTD...")
    ctd = pd.read_csv(
        CTD_PATH, sep="\t", comment="#", header=None,
        names=["ChemicalName", "ChemicalID", "CasRN", "DiseaseName", "DiseaseID",
               "DirectEvidence", "InferenceGeneSymbol", "InferenceScore",
               "OmimIDs", "PubMedIDs"],
        dtype=str, low_memory=False,
    )

    v1_drugs = set(DRUGS_LIST.read_text().splitlines())
    v1_diseases = set(DISEASES_LIST.read_text().splitlines())
    log(f"v1 vocab: {len(v1_drugs):,} drugs × {len(v1_diseases):,} diseases")

    # Positives: therapeutic in v1 vocab, deduped on (drug, disease)
    ther = ctd[ctd["DirectEvidence"] == "therapeutic"]
    ther = ther[ther["ChemicalName"].isin(v1_drugs) & ther["DiseaseName"].isin(v1_diseases)]
    pos_pairs = ther[["ChemicalName", "DiseaseName"]].drop_duplicates()
    pos_set = set(zip(pos_pairs["ChemicalName"], pos_pairs["DiseaseName"]))
    log(f"Positive pairs (therapeutic, unique): {len(pos_set):,}")

    # Hard negatives: marker/mechanism in v1 vocab, EXCLUDING therapeutic
    mark = ctd[ctd["DirectEvidence"] == "marker/mechanism"]
    mark = mark[mark["ChemicalName"].isin(v1_drugs) & mark["DiseaseName"].isin(v1_diseases)].copy()
    mark["n_pubs"] = mark["PubMedIDs"].fillna("").apply(lambda s: 0 if not s else len(s.split("|")))

    # Aggregate to one row per (drug, disease) with max pub count
    mark_agg = mark.groupby(["ChemicalName", "DiseaseName"], as_index=False)["n_pubs"].max()

    # Exclude pairs that are ALSO therapeutic (ambiguous — drug both treats and is associated)
    mark_agg["is_therapeutic"] = list(zip(mark_agg.ChemicalName, mark_agg.DiseaseName))
    mark_agg["is_therapeutic"] = mark_agg["is_therapeutic"].isin(pos_set)
    n_ambiguous = int(mark_agg["is_therapeutic"].sum())
    mark_agg = mark_agg[~mark_agg["is_therapeutic"]].copy()
    log(f"Marker/mechanism pairs in v1 vocab (deduped): {len(mark_agg):,} "
        f"(excluded {n_ambiguous:,} ambiguous pairs that are also therapeutic)")

    # Tier the hard-negative pool by evidence strength
    strong_hard = mark_agg[mark_agg["n_pubs"] >= 2]
    weak_hard = mark_agg[mark_agg["n_pubs"] == 1]
    log(f"  strong (≥2 pubs): {len(strong_hard):,}")
    log(f"  weak   (1 pub):   {len(weak_hard):,}")

    rng = np.random.default_rng(seed)
    n_pos = len(pos_set)

    # 2026-05-01 lesson learned: training with ONLY hard negatives created an inverse
    # popularity bias — the model demoted ALL well-studied drugs (because they dominate
    # marker/mechanism) and surfaced obscure plant compounds at the top of every disease.
    # Mixed sampling (some random + some hard) is necessary to keep the model anchored
    # to "this drug is unrelated to most diseases" while also learning "this drug is
    # specifically harmful for THIS disease."
    n_hard = int(n_pos * args.hard_negative_fraction)
    n_random = n_pos - n_hard
    log(f"Negative composition target: {n_hard:,} hard + {n_random:,} random = {n_pos:,} total "
        f"(hard fraction: {args.hard_negative_fraction:.2f})")

    # Sample hard negatives: prioritize strong (≥2 pubs), fill with weak (1 pub)
    if n_hard <= 0:
        hard_neg = pd.DataFrame(columns=["ChemicalName", "DiseaseName"])
        n_strong_used = 0
        n_weak_used = 0
    elif len(strong_hard) >= n_hard:
        hard_neg = strong_hard.sample(n=n_hard, random_state=seed)[["ChemicalName", "DiseaseName"]]
        n_strong_used = n_hard
        n_weak_used = 0
    else:
        n_strong_used = len(strong_hard)
        n_weak_needed = n_hard - n_strong_used
        weak_sample = weak_hard.sample(n=min(n_weak_needed, len(weak_hard)), random_state=seed)
        hard_neg = pd.concat(
            [strong_hard[["ChemicalName", "DiseaseName"]],
             weak_sample[["ChemicalName", "DiseaseName"]]],
            ignore_index=True,
        )
        n_weak_used = len(weak_sample)

    log(f"Hard negatives selected: {len(hard_neg):,} (strong: {n_strong_used:,}, weak: {n_weak_used:,})")

    # Sample random negatives to fill the rest, avoiding both positives AND hard negatives
    n_random_needed = n_pos - len(hard_neg)
    if n_random_needed > 0:
        log(f"Sampling {n_random_needed:,} random negatives (avoiding positives and hard negatives)...")
        all_drugs = sorted(v1_drugs)
        all_diseases = sorted(v1_diseases)
        hard_neg_set = set(zip(hard_neg["ChemicalName"], hard_neg["DiseaseName"]))
        random_negs = []
        attempts = 0
        max_attempts = n_random_needed * 20
        while len(random_negs) < n_random_needed and attempts < max_attempts:
            c = rng.choice(all_drugs)
            d = rng.choice(all_diseases)
            if (c, d) not in pos_set and (c, d) not in hard_neg_set:
                random_negs.append((c, d))
            attempts += 1
        rand_df = pd.DataFrame(random_negs, columns=["ChemicalName", "DiseaseName"])
        hard_neg = pd.concat([hard_neg, rand_df], ignore_index=True)

    # Combine into labeled dataset
    pos_labeled = pos_pairs.copy()
    pos_labeled["label"] = 1
    neg_labeled = hard_neg.copy()
    neg_labeled["label"] = 0

    pairs_df = pd.concat([pos_labeled, neg_labeled], ignore_index=True)
    pairs_df = pairs_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    stats = {
        "n_positive": int((pairs_df.label == 1).sum()),
        "n_negative": int((pairs_df.label == 0).sum()),
        "n_strong_hard_negatives": n_strong_used,
        "n_weak_hard_negatives": n_weak_used,
        "n_random_pad_negatives": max(0, n_pos - n_strong_used - n_weak_used),
        "n_ambiguous_excluded": n_ambiguous,
    }
    log(f"Final dataset: {len(pairs_df):,} pairs ({stats['n_positive']:,} pos / {stats['n_negative']:,} neg)")
    return pairs_df, stats


def build_features(pairs_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Convert pair names to 800-dim feature vectors using cached embeddings."""
    log("Loading cached embeddings...")
    drug_names = DRUGS_LIST.read_text().splitlines()
    disease_names = DISEASES_LIST.read_text().splitlines()
    drug_emb = np.load(DRUG_EMB)
    disease_emb = np.load(DISEASE_EMB)

    drug_idx = {n: i for i, n in enumerate(drug_names)}
    disease_idx = {n: i for i, n in enumerate(disease_names)}

    log(f"Building features for {len(pairs_df):,} pairs...")
    chem_vecs = np.vstack([drug_emb[drug_idx[c]] for c in pairs_df["ChemicalName"]])
    dis_vecs = np.vstack([disease_emb[disease_idx[d]] for d in pairs_df["DiseaseName"]])

    # Same as v1: [drug, disease, |drug - disease|, drug * disease]
    diff = np.abs(chem_vecs - dis_vecs)
    prod = chem_vecs * dis_vecs
    X = np.concatenate([chem_vecs, dis_vecs, diff, prod], axis=1).astype(np.float32)
    y = pairs_df["label"].values.astype(np.float32)
    log(f"Features: X={X.shape}, y={y.shape}")
    return X, y


def train(X_train, y_train, X_val, y_val, num_epochs: int, batch_size: int, lr: float, device: torch.device):
    """Train MLP with same hyperparameters as v1."""
    model = MLPClassifier(input_dim=X_train.shape[1], hidden_dim=256, dropout=0.3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    train_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).float(),
    )
    val_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_val).float(),
        torch.from_numpy(y_val).float(),
    )
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    log(f"Training for {num_epochs} epochs on {device}...")
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb), yb).item() * xb.size(0)
        val_loss /= len(val_ds)
        log(f"  Epoch {epoch:2d}/{num_epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10, help="Same as v1 default")
    parser.add_argument("--batch-size", type=int, default=256, help="Same as v1 default")
    parser.add_argument("--lr", type=float, default=1e-3, help="Same as v1 default")
    parser.add_argument("--test-size", type=float, default=0.30, help="Same as v1 default")
    parser.add_argument("--val-size", type=float, default=0.50, help="Same as v1 default (within 30% test pool)")
    parser.add_argument("--hard-negative-fraction", type=float, default=0.5,
                        help="Fraction of negatives that are hard (CTD marker/mechanism). "
                             "0.0 = pure random (matches v1), 1.0 = pure hard (creates inverse popularity bias). "
                             "0.5 = mixed (recommended).")
    parser.add_argument("--label", type=str, default="hardneg-mixed",
                        help="Tag for output files (e.g. 'hardneg-mixed' → biolink_v2_hardneg-mixed.pt).")
    args = parser.parse_args()

    # Output paths reflect the label so we can save multiple variants side-by-side
    global WEIGHTS_OUT, VAL_LOGITS_OUT, VAL_LABELS_OUT, TEMP_OUT
    WEIGHTS_OUT = REPO_ROOT / "models" / f"biolink_v2_{args.label}.pt"
    VAL_LOGITS_OUT = REPO_ROOT / "data" / f"val_logits_{args.label}.npy"
    VAL_LABELS_OUT = REPO_ROOT / "data" / f"val_labels_{args.label}.npy"
    TEMP_OUT = REPO_ROOT / "data" / f"temperature_{args.label}.json"

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    t0 = time.time()
    pairs_df, dataset_stats = build_pair_dataset(seed=args.seed, args=args)
    X, y = build_features(pairs_df)

    # Stratified pair-level splits — same as v1 for fair comparison
    train_idx, temp_idx = train_test_split(
        np.arange(len(pairs_df)),
        test_size=args.test_size,
        stratify=pairs_df["label"],
        random_state=args.seed,
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=args.val_size,
        stratify=pairs_df["label"].iloc[temp_idx],
        random_state=args.seed,
    )
    log(f"Splits: train={len(train_idx):,}, val={len(val_idx):,}, test={len(test_idx):,}")

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    model = train(
        X[train_idx], y[train_idx],
        X[val_idx], y[val_idx],
        num_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=device,
    )

    # Compute val logits for calibration
    model.eval()
    with torch.no_grad():
        val_logits = model(torch.from_numpy(X[val_idx]).float().to(device)).cpu().numpy()
        test_logits = model(torch.from_numpy(X[test_idx]).float().to(device)).cpu().numpy()

    val_labels = y[val_idx]
    test_labels = y[test_idx]

    # Sanity-check metrics on val + test
    val_probs = 1.0 / (1.0 + np.exp(-val_logits.astype(np.float64)))
    test_probs = 1.0 / (1.0 + np.exp(-test_logits.astype(np.float64)))
    val_auc = roc_auc_score(val_labels, val_probs)
    test_auc = roc_auc_score(test_labels, test_probs)
    val_ap = average_precision_score(val_labels, val_probs)
    test_ap = average_precision_score(test_labels, test_probs)

    log(f"\nMetrics (raw probabilities, no calibration):")
    log(f"  Val AUC: {val_auc:.4f}   AP: {val_ap:.4f}")
    log(f"  Test AUC: {test_auc:.4f}  AP: {test_ap:.4f}")
    log(f"  Compare to v1 random-negative model val AUC: 0.9497 (50/50 random negatives)")
    log(f"  Note: lower AUC here EXPECTED because hard negatives are HARDER to distinguish")

    # Save artifacts
    WEIGHTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), WEIGHTS_OUT)
    np.save(VAL_LOGITS_OUT, val_logits)
    np.save(VAL_LABELS_OUT, val_labels)
    log(f"Saved: {WEIGHTS_OUT.name}, {VAL_LOGITS_OUT.name}, {VAL_LABELS_OUT.name}")

    # Re-fit calibration: temperature on new val set + same prior_shift
    log("\nFitting calibration on new val set...")
    scaler = TemperatureScaler()
    new_T = scaler.fit(val_logits.astype(np.float64), val_labels.astype(np.float64), save_path=str(TEMP_OUT))
    # Re-add prior shift (same 1% prior assumption — independent of model)
    scaler.prior_shift = TemperatureScaler.shift_for_prior(p_train=0.5, p_real=0.01)
    scaler.save(str(TEMP_OUT))
    log(f"Saved: {TEMP_OUT.name} (T={new_T:.4f}, prior_shift={scaler.prior_shift:.4f})")

    elapsed = time.time() - t0
    log(f"\nDone in {elapsed:.1f}s. Run 'python scripts/run_regression.py --label after-hardneg' next,")
    log(f"  then point it at the new model+temperature for comparison (or update load paths).")
    log(f"\nDataset stats:")
    for k, v in dataset_stats.items():
        log(f"  {k}: {v:,}")


if __name__ == "__main__":
    main()
