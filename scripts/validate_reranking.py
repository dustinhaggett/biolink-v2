"""
End-to-end validation of harm-aware Perplexity re-ranking.

Calls the real Perplexity API for 6 designed (drug, disease) pairs to confirm:
  1. The new HARM_FOR_INDICATION field is being emitted by the model
  2. The parser extracts it correctly
  3. The reranker demotes HARMFUL pairs and preserves UNKNOWN pairs

Cost: ~$0.03 per run (6 sonar queries). Time: ~3 min.

Usage:
    python scripts/validate_reranking.py
    python scripts/validate_reranking.py --out results/rerank_validation.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import tomllib
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Designed test cases — each entry: (drug, disease, expected_harm, expected_verdict_one_of, category)
# Category is informational; controls how cases are grouped in the report.
CASES = [
    # ── Direction-blindness failures (should now be flagged HARMFUL) ──
    ("Cocaine",            "Alcoholism",                                "harmful",     ["conflicts", "insufficient"], "harm_fail"),
    ("Cyclosporine",       "Lyme Disease",                              "harmful",     ["conflicts", "insufficient"], "harm_fail"),
    ("Scopolamine",        "Amnesia",                                   "harmful",     ["conflicts", "insufficient"], "harm_fail"),
    ("Streptozocin",       "Diabetes Mellitus, Type 2",                 "harmful",     ["conflicts", "insufficient"], "harm_fail"),
    ("Nicotine",           "Sleep Initiation and Maintenance Disorders", "harmful",    ["conflicts", "insufficient"], "harm_fail"),
    ("Nicotine",           "Migraine Disorders",                        "harmful",     ["conflicts", "insufficient"], "harm_fail"),
    ("Heroin",             "Alcoholism",                                "harmful",     ["conflicts", "insufficient"], "harm_fail"),
    ("Phencyclidine",      "Alcoholism",                                "harmful",     ["conflicts", "insufficient"], "harm_fail"),
    ("Methylprednisolone", "Lyme Disease",                              "harmful",     ["conflicts", "insufficient"], "harm_fail"),
    ("Strychnine",         "Amnesia",                                   "harmful",     ["conflicts", "insufficient"], "harm_fail"),

    # ── Documented wins (should be SUPPORTS or STANDARD-OF-CARE, NOT HARMFUL) ──
    ("Naltrexone",         "Alcoholism",                                "not_harmful", ["supports", "standard-of-care"], "win"),
    ("Topiramate",         "Migraine Disorders",                        "not_harmful", ["supports", "standard-of-care"], "win"),
    ("Doxycycline",        "Lyme Disease",                              "not_harmful", ["supports", "standard-of-care"], "win"),
    ("Methotrexate",       "Arthritis, Rheumatoid",                     "not_harmful", ["supports", "standard-of-care"], "win"),
    ("Pregabalin",         "Fibromyalgia",                              "not_harmful", ["supports", "standard-of-care"], "win"),
    ("Tacrolimus",         "Dermatitis",                                "not_harmful", ["supports", "standard-of-care"], "win"),
    ("Donepezil",          "Amnesia",                                   "not_harmful", ["supports", "standard-of-care"], "win"),

    # ── Discovery candidates (emerging research, should be SUPPORTS or INSUFFICIENT, NOT HARMFUL) ──
    # These are the "naltrexone-for-fibromyalgia" pattern — preserve, don't demote.
    ("Naltrexone",         "Fibromyalgia",                              "not_harmful", ["supports", "insufficient", "conflicts"], "discovery"),
    ("Aspirin",            "Colorectal Neoplasms",                      "not_harmful", ["supports", "standard-of-care"], "discovery"),
    ("Sildenafil",         "Hypertension, Pulmonary",                   "not_harmful", ["supports", "standard-of-care"], "discovery"),
]


def load_perplexity_key() -> str:
    """Load API key from .streamlit/secrets.toml (parent biolink_v2 dir)."""
    candidates = [
        REPO_ROOT / ".streamlit" / "secrets.toml",
        Path("/Users/berlin/biolink_v2") / ".streamlit" / "secrets.toml",
    ]
    for path in candidates:
        if path.exists():
            with open(path, "rb") as f:
                data = tomllib.load(f)
            if "PERPLEXITY_API_KEY" in data:
                return data["PERPLEXITY_API_KEY"]
    if os.environ.get("PERPLEXITY_API_KEY"):
        return os.environ["PERPLEXITY_API_KEY"]
    raise RuntimeError("PERPLEXITY_API_KEY not found in .streamlit/secrets.toml or env")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / f"rerank_validation_{datetime.now():%Y%m%d_%H%M%S}.json")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Inject key into env so enrichment.perplexity finds it
    os.environ["PERPLEXITY_API_KEY"] = load_perplexity_key()

    from enrichment.perplexity import search_drug_disease  # noqa: E402
    from core.reranking import apply_evidence_reranking  # noqa: E402

    print(f"[{datetime.now():%H:%M:%S}] Validating {len(CASES)} drug-disease pairs against Perplexity...")
    print(f"  Expected ~{len(CASES) * 0.005:.2f} USD, ~{len(CASES) * 30}s")
    print()

    rows = []
    for drug, disease, exp_harm, exp_verdicts, category in CASES:
        t0 = time.time()
        ev = search_drug_disease(drug=drug, disease=disease)
        elapsed = time.time() - t0

        actual_harm = (ev.get("harm_for_indication") or "unknown").lower()
        actual_verdict = (ev.get("verdict") or "insufficient").lower()
        # For discovery candidates, "unknown" harm is also acceptable (preserves novel)
        if category == "discovery" and actual_harm == "unknown":
            harm_match = "✓"
        else:
            harm_match = "✓" if actual_harm == exp_harm else "✗"
        verdict_match = "✓" if actual_verdict in exp_verdicts else "✗"

        print(f"[{datetime.now():%H:%M:%S}] [{category:9s}] {drug} for {disease}  ({elapsed:.1f}s)")
        print(f"    expected harm:    {exp_harm:<12s}  actual: {actual_harm:<12s} {harm_match}")
        print(f"    expected verdict: {'/'.join(exp_verdicts):<35s}  actual: {actual_verdict:<25s} {verdict_match}")
        print(f"    tldr: {ev.get('tldr') or '(none)'}")
        if ev.get("error"):
            print(f"    ⚠️  error: {ev['error']}")
        print()

        rows.append({
            "drug": drug,
            "disease": disease,
            "category": category,
            "expected_harm": exp_harm,
            "actual_harm": actual_harm,
            "expected_verdict_one_of": exp_verdicts,
            "actual_verdict": actual_verdict,
            "tldr": ev.get("tldr"),
            "evidence_quality": ev.get("evidence_quality"),
            "has_interactions": ev.get("has_interactions"),
            "harm_match": harm_match == "✓",
            "verdict_match": verdict_match == "✓",
            "elapsed_seconds": round(elapsed, 1),
            "error": ev.get("error"),
            "full_evidence": ev,
        })

    # Validate the reranker on a synthetic equal-prob list using these verdicts
    print(f"\n[{datetime.now():%H:%M:%S}] Reranker simulation (synthetic equal-prob inputs):")
    synthetic_results = []
    for i, (drug, disease, _, _, _) in enumerate(CASES, start=1):
        ev = next(r["full_evidence"] for r in rows if r["drug"] == drug and r["disease"] == disease)
        synthetic_results.append({
            "drug": drug,
            "disease_for_query": disease,
            "proba": 0.5,
            "rank": i,
            "evidence": ev,
            "clinical_trials": [],
        })

    reranked = apply_evidence_reranking(synthetic_results)
    print(f"\n  After rerank (sorted by reranked_proba desc):")
    for r in reranked:
        print(f"    #{r['rank']:>2}  {r['drug']:<20s} for {r['disease_for_query']:<48s}  → {r['reranked_proba']:.3f}  ({r['rerank_reason']})")

    # Summary by category
    print(f"\n=== Summary by category ===")
    by_cat: dict[str, list[dict]] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r)
    for cat, rs in by_cat.items():
        n_harm = sum(1 for r in rs if r["harm_match"])
        n_verdict = sum(1 for r in rs if r["verdict_match"])
        print(f"  [{cat:10s}]  harm: {n_harm}/{len(rs)}   verdict: {n_verdict}/{len(rs)}")
    n_harm_ok = sum(1 for r in rows if r["harm_match"])
    n_verdict_ok = sum(1 for r in rows if r["verdict_match"])
    print(f"  TOTAL                harm: {n_harm_ok}/{len(rows)}   verdict: {n_verdict_ok}/{len(rows)}")

    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "n_cases": len(CASES),
            "n_harm_matches": n_harm_ok,
            "n_verdict_matches": n_verdict_ok,
        },
        "cases": rows,
    }
    args.out.write_text(json.dumps(output, indent=2))
    print(f"\n  Wrote {args.out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
