#!/usr/bin/env python3
"""
Regression test runner for BioLink v2.

Runs a fixed set of disease queries documented during pre-presentation review
(2026-04-30) and saves top-N drug rankings + calibrated probabilities to JSON.

Bypasses the LLM intent mapper — passes CTD/MeSH disease names directly to the
model — so runs are deterministic, fast, and require no API keys.

Usage:
    python scripts/run_regression.py
    python scripts/run_regression.py --label baseline
    python scripts/run_regression.py --label after-hard-negatives --top-n 30

Output: results/regression_<label>_<timestamp>.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.calibration import TemperatureScaler, confidence_tier  # noqa: E402
from core.model import BioLinkModel  # noqa: E402


# Test cases documented during pre-presentation review.
# Each entry: CTD/MeSH name, category, and human-readable expectations
# (the expectations are notes for reviewers — not asserted programmatically).
TEST_CASES = [
    # ─── WINS — must NOT degrade after fixes ────────────────────────────────
    {
        "name": "Fibromyalgia",
        "category": "win",
        "expectations": [
            "Pregabalin in top 5 (Lyrica — epilepsy drug, FDA-approved for fibro)",
            "Milnacipran in top 5 (Savella — antidepressant, FDA-approved for fibro)",
            "Duloxetine in top 10 (Cymbalta — antidepressant, FDA-approved for fibro)",
            "Naltrexone in top 10 (emerging discovery candidate — opioid antagonist)",
        ],
    },
    {
        "name": "Dermatitis",
        "category": "win",
        "expectations": [
            "Pimecrolimus in top 5 (Elidel — calcineurin inhibitor)",
            "Tacrolimus in top 10 (Protopic — transplant rejection drug)",
            "Cyclosporine in top 10 (transplant rejection drug, systemic for severe AD)",
        ],
    },
    {
        "name": "Migraine Disorders",
        "category": "win",
        "expectations": [
            "Topiramate in top 3 (epilepsy drug)",
            "Amitriptyline in top 10 (tricyclic antidepressant)",
        ],
    },
    {
        "name": "Arthritis, Rheumatoid",
        "category": "win",
        "expectations": [
            "Methotrexate in top 5 (first-line DMARD)",
            "Sulfasalazine, Hydroxychloroquine in top 10 (DMARDs)",
        ],
    },
    {
        "name": "Tuberculosis",
        "category": "win",
        "expectations": [
            "Isoniazid, Rifampin, Ethambutol, Pyrazinamide in top 10 (first-line RIPE)",
        ],
    },
    {
        "name": "Asthma",
        "category": "win",
        "expectations": [
            "Leukotriene antagonists (Montelukast, Pranlukast, Zafirlukast) in top 5",
            "Inhaled corticosteroids (Budesonide, Fluticasone, Beclomethasone) in top 10",
        ],
    },
    # ─── FAILURES — should improve after fixes ──────────────────────────────
    {
        "name": "Lyme Disease",
        "category": "failure",
        "expectations": [
            "Cyclosporine should drop OUT of top 10 (currently #1, immunosuppressant for bacterial infection)",
            "Methylprednisolone should drop (currently #2)",
            "Doxycycline, Azithromycin, Ceftriaxone should rise toward top 5",
        ],
    },
    {
        "name": "Diabetes Mellitus, Type 2",
        "category": "failure",
        "expectations": [
            "Streptozocin should drop OUT of top 20 (currently #10 — INDUCES diabetes in research)",
            "Nicotine should drop OUT of top 20 (currently #19 — increases diabetes risk)",
            "Metformin, GLP-1 agonists, sulfonylureas should remain visible",
        ],
    },
    {
        "name": "Alcoholism",
        "category": "failure",
        "expectations": [
            "Cocaine, Heroin, PCP, Methamphetamine should drop OUT of top 10 (currently #2-#5)",
            "Naltrexone (currently #6) should rise to top 3",
            "Baclofen, Disulfiram should remain visible",
        ],
    },
    {
        "name": "Sleep Initiation and Maintenance Disorders",
        "category": "failure",
        "expectations": [
            "Nicotine should drop OUT of top 10 (currently #1 — stimulant, disrupts sleep)",
            "Trazodone (currently #10) should rise into top 5",
            "Melatonin (currently #3) should remain visible",
        ],
    },
    {
        "name": "Attention Deficit Disorder with Hyperactivity",
        "category": "failure",
        "expectations": [
            "Methylphenidate should rise into top 5 (currently #8 — primary ADHD treatment)",
            "Dextroamphetamine should rise (currently #6)",
            "Cannabidiol, Nicotine, Scopolamine should drop OUT of top 5 (currently #1-#3)",
        ],
    },
    {
        "name": "Amnesia",
        "category": "failure",
        "expectations": [
            "Scopolamine should drop OUT of top 10 (currently #1 — INDUCES amnesia)",
            "Strychnine, Phencyclidine, Picrotoxin should drop OUT of top 20 (poisons/illicit)",
            "Donepezil, Galantamine, Rivastigmine should rise into top 5",
        ],
    },
]


def get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=2,
        )
        return result.stdout.strip()[:12] if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def run_query(
    model: BioLinkModel,
    scaler: TemperatureScaler,
    ctd_name: str,
    top_n: int,
) -> list[dict]:
    """Score all drugs against a single CTD disease name. No API calls."""
    disease_vec = model.encode_disease(ctd_name)
    scored = model.score_all_drugs(disease_vec)[:top_n]
    out = []
    for rank, (drug, logit) in enumerate(scored, start=1):
        proba = float(scaler.calibrated_proba(float(logit)))
        out.append(
            {
                "rank": rank,
                "drug": drug,
                "logit": float(logit),
                "proba": proba,
                "tier": confidence_tier(proba),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", type=str, default="baseline", help="Label for this run (e.g. 'baseline', 'after-hard-negatives').")
    parser.add_argument("--top-n", type=int, default=20, help="Top-N drugs to record per query.")
    parser.add_argument("--out", type=Path, default=None, help="Output JSON path (defaults to results/regression_<label>_<timestamp>.json).")
    args = parser.parse_args()

    if args.out is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out = REPO_ROOT / "results" / f"regression_{args.label}_{ts}.json"
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[{datetime.now():%H:%M:%S}] Loading model and embeddings...", file=sys.stderr)
    weights = REPO_ROOT / "models" / "biolink_v1.pt"
    biowordvec = REPO_ROOT / "data" / "BioWordVec_PubMed_MIMICIII_d200.vec.bin"
    drugs_list = REPO_ROOT / "data" / "drugs_list.txt"
    diseases_list = REPO_ROOT / "data" / "diseases_list.txt"
    temperature_path = REPO_ROOT / "data" / "temperature.json"

    model = BioLinkModel(
        weights_path=str(weights),
        biowordvec_path=str(biowordvec),
        drugs_list_path=str(drugs_list),
        diseases_list_path=str(diseases_list),
    )
    scaler = (
        TemperatureScaler.load(str(temperature_path))
        if temperature_path.exists()
        else TemperatureScaler(T=1.0)
    )

    # Validate test case names exist in the loaded diseases list.
    diseases_set = set(model.disease_names)
    missing = [case["name"] for case in TEST_CASES if case["name"] not in diseases_set]
    if missing:
        print(f"\n  ⚠️  Not in diseases_list.txt — will return zero-vector results:", file=sys.stderr)
        for name in missing:
            print(f"       {name!r}", file=sys.stderr)
        print("", file=sys.stderr)

    output = {
        "metadata": {
            "label": args.label,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "git_commit": get_git_commit(),
            "weights": str(weights.relative_to(REPO_ROOT)),
            "temperature_T": getattr(scaler, "T", None),
            "top_n": args.top_n,
            "n_drugs": len(model.drug_names),
            "n_diseases": len(model.disease_names),
        },
        "test_cases": {},
    }

    for case in TEST_CASES:
        name = case["name"]
        marker = "✓" if name in diseases_set else "⚠"
        print(f"[{datetime.now():%H:%M:%S}] {marker} {name} ({case['category']})", file=sys.stderr)
        results = run_query(model, scaler, name, args.top_n)
        output["test_cases"][name] = {
            "category": case["category"],
            "expectations": case["expectations"],
            "results": results,
        }

    args.out.write_text(json.dumps(output, indent=2))
    print(f"\n[{datetime.now():%H:%M:%S}] Wrote {args.out.relative_to(REPO_ROOT)}", file=sys.stderr)
    print(f"  Compare runs with: scripts/diff_regression.py <baseline.json> <new.json>  (TODO)", file=sys.stderr)


if __name__ == "__main__":
    main()
