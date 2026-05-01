#!/usr/bin/env python3
"""
Evaluate the BioLink v2 model against trivial baselines on per-disease ranking.

The reported AUC=0.947 from v1 came from a 50/50 pair-level held-out set.
That's not the task the system is actually used for. The real task is:
"given a disease, rank drugs by how likely each is therapeutic."

This script computes per-disease ranking AUC for:
  1. Trained MLP (the deployed model)
  2. Cosine similarity (BioWordVec embedding similarity, no ML)
  3. Drug popularity (rank by # of therapeutic indications in CTD globally)
  4. Random baseline (sanity check, should be ~0.5)

If the MLP barely beats cosine, the deep model isn't doing much work.

Usage:
    python scripts/eval_baselines.py                # Eval on regression test diseases (fast)
    python scripts/eval_baselines.py --all-diseases # Eval on all 2526 diseases (~10 min)
    python scripts/eval_baselines.py --max-diseases 200  # Sample N diseases
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.model import BioLinkModel  # noqa: E402

# Same 12 queries as run_regression.py — keeps the eval comparable.
REGRESSION_DISEASES = [
    "Fibromyalgia",
    "Dermatitis",
    "Migraine Disorders",
    "Arthritis, Rheumatoid",
    "Tuberculosis",
    "Asthma",
    "Lyme Disease",
    "Diabetes Mellitus, Type 2",
    "Alcoholism",
    "Sleep Initiation and Maintenance Disorders",
    "Attention Deficit Disorder with Hyperactivity",
    "Amnesia",
]


def load_therapeutic_pairs() -> set[tuple[str, str]]:
    """Load CTD therapeutic (drug, disease) pairs as a set of tuples."""
    print(f"[{datetime.now():%H:%M:%S}] Loading CTD therapeutic pairs...", file=sys.stderr)
    ctd = pd.read_csv(
        REPO_ROOT / "data" / "CTD_chemicals_diseases.tsv.gz",
        sep="\t",
        comment="#",
        header=None,
        names=[
            "ChemicalName", "ChemicalID", "CasRN", "DiseaseName", "DiseaseID",
            "DirectEvidence", "InferenceGeneSymbol", "InferenceScore",
            "OmimIDs", "PubMedIDs",
        ],
        dtype=str,
        low_memory=False,
    )
    ther = ctd[ctd["DirectEvidence"] == "therapeutic"]
    pairs = set(zip(ther["ChemicalName"], ther["DiseaseName"]))
    print(f"  {len(pairs):,} unique therapeutic pairs", file=sys.stderr)
    return pairs


def per_disease_metrics(
    disease: str,
    drug_names: list[str],
    drug_embeddings: np.ndarray,
    disease_vec: np.ndarray,
    model_logits: np.ndarray,
    drug_popularity: np.ndarray,
    therapeutic_pairs: set[tuple[str, str]],
    rng: np.random.Generator,
) -> dict:
    """Compute AUC for each ranking method on a single disease.

    Positives: drugs marked therapeutic for `disease` in CTD.
    Negatives: all other drugs in the model's vocabulary.
    """
    labels = np.array(
        [(drug, disease) in therapeutic_pairs for drug in drug_names], dtype=int
    )
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos

    # Need at least one positive AND one negative to compute AUC
    if n_pos == 0 or n_neg == 0:
        return {"n_pos": n_pos, "n_neg": n_neg, "skipped": True}

    # Cosine similarity baseline (BioWordVec only, no MLP)
    norm_drug = np.linalg.norm(drug_embeddings, axis=1) + 1e-8
    norm_dis = float(np.linalg.norm(disease_vec)) + 1e-8
    cosine = (drug_embeddings @ disease_vec) / (norm_drug * norm_dis)

    # Random baseline (sanity check)
    random_scores = rng.standard_normal(len(drug_names))

    metrics = {"n_pos": n_pos, "n_neg": n_neg, "skipped": False}
    for name, scores in [
        ("model", model_logits),
        ("cosine", cosine),
        ("popularity", drug_popularity),
        ("random", random_scores),
    ]:
        metrics[f"{name}_auc"] = float(roc_auc_score(labels, scores))
        metrics[f"{name}_ap"] = float(average_precision_score(labels, scores))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all-diseases", action="store_true", help="Eval on all 2526 diseases (slower).")
    parser.add_argument("--max-diseases", type=int, default=None, help="Cap on number of diseases to eval (random sample).")
    parser.add_argument("--regression-only", action="store_true", help="Eval only the 12 regression test diseases (fastest).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / f"baselines_{datetime.now():%Y%m%d_%H%M%S}.json")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # Load CTD ground truth
    therapeutic_pairs = load_therapeutic_pairs()

    # Build per-drug therapeutic count for popularity baseline
    drug_pop_count: dict[str, int] = {}
    for drug, _ in therapeutic_pairs:
        drug_pop_count[drug] = drug_pop_count.get(drug, 0) + 1

    # Load model + embeddings
    print(f"[{datetime.now():%H:%M:%S}] Loading model + embeddings...", file=sys.stderr)
    model = BioLinkModel(
        weights_path=str(REPO_ROOT / "models" / "biolink_v1.pt"),
        biowordvec_path=str(REPO_ROOT / "data" / "BioWordVec_PubMed_MIMICIII_d200.vec.bin"),
        drugs_list_path=str(REPO_ROOT / "data" / "drugs_list.txt"),
        diseases_list_path=str(REPO_ROOT / "data" / "diseases_list.txt"),
    )
    drug_names = model.drug_names
    drug_embeddings = model.drug_embeddings
    drug_popularity = np.array([drug_pop_count.get(d, 0) for d in drug_names], dtype=float)

    # Decide which diseases to evaluate
    if args.regression_only:
        diseases = [d for d in REGRESSION_DISEASES if d in model.disease_names]
    elif args.all_diseases:
        diseases = list(model.disease_names)
    else:
        diseases = REGRESSION_DISEASES.copy()

    if args.max_diseases and len(diseases) > args.max_diseases:
        diseases = list(rng.choice(diseases, size=args.max_diseases, replace=False))

    print(f"[{datetime.now():%H:%M:%S}] Evaluating {len(diseases)} diseases × {len(drug_names)} drugs", file=sys.stderr)

    per_disease_results: dict[str, dict] = {}
    for i, disease in enumerate(diseases, 1):
        if i % 50 == 0 or i == 1:
            print(f"[{datetime.now():%H:%M:%S}]   {i}/{len(diseases)}  {disease}", file=sys.stderr)

        disease_vec = model.encode_disease(disease)
        # Compute model logits for all drugs (batched)
        scored = model.score_all_drugs(disease_vec)
        # Reorder by drug_names order so labels align
        logit_lookup = dict(scored)
        model_logits = np.array([logit_lookup[d] for d in drug_names], dtype=float)

        per_disease_results[disease] = per_disease_metrics(
            disease=disease,
            drug_names=drug_names,
            drug_embeddings=drug_embeddings,
            disease_vec=disease_vec,
            model_logits=model_logits,
            drug_popularity=drug_popularity,
            therapeutic_pairs=therapeutic_pairs,
            rng=rng,
        )

    # Aggregate
    methods = ["model", "cosine", "popularity", "random"]
    aggregated = {}
    skipped = sum(1 for m in per_disease_results.values() if m.get("skipped"))
    valid = [m for m in per_disease_results.values() if not m.get("skipped")]
    print(f"\n[{datetime.now():%H:%M:%S}] Aggregating: {len(valid)} valid, {skipped} skipped (no positives or no negatives)", file=sys.stderr)

    for method in methods:
        aucs = np.array([m[f"{method}_auc"] for m in valid])
        aps = np.array([m[f"{method}_ap"] for m in valid])
        aggregated[method] = {
            "auc_mean": float(aucs.mean()),
            "auc_median": float(np.median(aucs)),
            "auc_std": float(aucs.std()),
            "auc_p25": float(np.percentile(aucs, 25)),
            "auc_p75": float(np.percentile(aucs, 75)),
            "ap_mean": float(aps.mean()),
            "ap_median": float(np.median(aps)),
        }

    # Print summary
    print(f"\n{'Method':<12} {'AUC mean':>10} {'AUC median':>11} {'AUC p25':>9} {'AUC p75':>9} {'AP mean':>10}")
    print("-" * 65)
    for method in methods:
        a = aggregated[method]
        print(f"{method:<12} {a['auc_mean']:>10.4f} {a['auc_median']:>11.4f} {a['auc_p25']:>9.4f} {a['auc_p75']:>9.4f} {a['ap_mean']:>10.4f}")

    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "n_diseases_evaluated": len(valid),
            "n_diseases_skipped": skipped,
            "n_drugs": len(drug_names),
            "n_therapeutic_pairs_in_ctd": len(therapeutic_pairs),
            "task": "per-disease ranking AUC: rank all drugs for each disease, positives = CTD therapeutic indications",
        },
        "aggregated": aggregated,
        "per_disease": per_disease_results,
    }
    args.out.write_text(json.dumps(output, indent=2))
    print(f"\n[{datetime.now():%H:%M:%S}] Wrote {args.out.relative_to(REPO_ROOT)}", file=sys.stderr)


if __name__ == "__main__":
    main()
