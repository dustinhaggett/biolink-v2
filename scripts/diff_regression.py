"""
Compare two regression test runs side-by-side.

Highlights:
  - Whether each documented expectation is now satisfied (failures improving)
  - Whether wins are preserved (didn't degrade)
  - Drugs that moved up/down significantly in the rankings

Usage:
    python scripts/diff_regression.py <before.json> <after.json>
    python scripts/diff_regression.py results/regression_baseline_*.json results/regression_after-hardneg_*.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Drugs to specifically watch for each regression case (positive and negative).
# Positive = should be in top-10 (these are real treatments); Negative = should NOT be.
WATCH = {
    "Fibromyalgia": {
        "positive": ["Pregabalin", "Milnacipran", "Duloxetine Hydrochloride", "Naltrexone"],
        "negative": [],
    },
    "Dermatitis": {
        "positive": ["pimecrolimus", "Tacrolimus", "Cyclosporine"],
        "negative": [],
    },
    "Migraine Disorders": {
        "positive": ["Topiramate", "Amitriptyline", "Sumatriptan"],
        "negative": ["Nicotine"],  # Causal/harmful — should drop
    },
    "Arthritis, Rheumatoid": {
        "positive": ["Methotrexate", "Sulfasalazine", "Hydroxychloroquine"],
        "negative": [],
    },
    "Tuberculosis": {
        "positive": ["Isoniazid", "Rifampin", "Ethambutol", "Pyrazinamide"],
        "negative": [],
    },
    "Asthma": {
        "positive": ["montelukast", "Budesonide", "Fluticasone", "Albuterol"],
        "negative": [],
    },
    "Lyme Disease": {
        "positive": ["Doxycycline", "Azithromycin", "Ceftriaxone"],
        "negative": ["Cyclosporine", "Methylprednisolone", "Prednisone"],
    },
    "Diabetes Mellitus, Type 2": {
        "positive": ["Metformin", "Liraglutide"],
        "negative": ["Streptozocin", "Nicotine"],  # Streptozocin INDUCES diabetes
    },
    "Alcoholism": {
        "positive": ["Naltrexone", "Disulfiram", "Baclofen"],
        "negative": ["Cocaine", "Heroin", "Phencyclidine", "Methamphetamine"],
    },
    "Sleep Initiation and Maintenance Disorders": {
        "positive": ["Trazodone", "Melatonin", "Diazepam"],
        "negative": ["Nicotine"],  # Stimulant — disrupts sleep
    },
    "Attention Deficit Disorder with Hyperactivity": {
        "positive": ["Methylphenidate", "Dextroamphetamine"],
        "negative": ["Cannabidiol", "Nicotine", "Scopolamine"],
    },
    "Amnesia": {
        "positive": ["Donepezil", "Galantamine", "Rivastigmine"],
        "negative": ["Scopolamine", "Strychnine", "Phencyclidine", "Picrotoxin"],  # Induce amnesia / poisons
    },
}


def rank_of(drug: str, results: list[dict]) -> int | None:
    """Return 1-indexed rank of drug in results, or None if not in top-N."""
    for r in results:
        if r["drug"].lower() == drug.lower():
            return r["rank"]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("before", type=Path, help="Baseline run JSON")
    parser.add_argument("after", type=Path, help="New run JSON")
    args = parser.parse_args()

    before = json.loads(args.before.read_text())
    after = json.loads(args.after.read_text())

    print(f"BEFORE: {args.before.name}  weights={before['metadata'].get('weights','?')}")
    print(f"AFTER:  {args.after.name}  weights={after['metadata'].get('weights','?')}")
    print()

    # Per-disease scoreboard
    score = {"win_preserved": 0, "win_degraded": 0,
             "failure_fixed": 0, "failure_persists": 0,
             "neutral": 0}

    for disease, watch in WATCH.items():
        if disease not in before["test_cases"] or disease not in after["test_cases"]:
            continue
        b_results = before["test_cases"][disease]["results"]
        a_results = after["test_cases"][disease]["results"]
        category = before["test_cases"][disease]["category"]

        marker = "✓" if category == "win" else "✗"
        print(f"{marker} {disease} ({category})")

        # Positive watchlist (should be in top 10)
        for drug in watch["positive"]:
            b_rank = rank_of(drug, b_results)
            a_rank = rank_of(drug, a_results)
            arrow = _arrow(b_rank, a_rank)
            tag = _judge(b_rank, a_rank, want_high=True)
            print(f"    [+] {drug:<35} {_fmt(b_rank)} → {_fmt(a_rank)}  {arrow} {tag}")

        # Negative watchlist (should NOT be in top 10)
        for drug in watch["negative"]:
            b_rank = rank_of(drug, b_results)
            a_rank = rank_of(drug, a_results)
            arrow = _arrow(b_rank, a_rank)
            tag = _judge(b_rank, a_rank, want_high=False)
            print(f"    [-] {drug:<35} {_fmt(b_rank)} → {_fmt(a_rank)}  {arrow} {tag}")

        # Tally
        if category == "win":
            top5_kept = sum(1 for d in watch["positive"] if (rank_of(d, a_results) or 99) <= 5)
            if top5_kept >= max(1, len(watch["positive"]) // 2):
                score["win_preserved"] += 1
            else:
                score["win_degraded"] += 1
        else:
            neg_dropped = sum(1 for d in watch["negative"]
                              if (rank_of(d, a_results) or 99) > 5
                              and (rank_of(d, b_results) or 99) <= 5)
            if neg_dropped > 0:
                score["failure_fixed"] += 1
            else:
                score["failure_persists"] += 1
        print()

    print("=" * 60)
    print(f"Wins preserved:    {score['win_preserved']}")
    print(f"Wins degraded:     {score['win_degraded']}")
    print(f"Failures improved: {score['failure_fixed']}")
    print(f"Failures persist:  {score['failure_persists']}")


def _fmt(rank: int | None) -> str:
    return f"#{rank:>2}" if rank is not None else "(>20)"


def _arrow(b: int | None, a: int | None) -> str:
    """Visual arrow showing direction of movement."""
    if b is None and a is None: return "  "
    if b is None: return "▲▲"  # Entered top-N
    if a is None: return "▼▼"  # Dropped out of top-N
    if a < b: return "▲ "  # Moved up (better rank = lower number)
    if a > b: return "▼ "  # Moved down
    return "= "


def _judge(b: int | None, a: int | None, want_high: bool) -> str:
    """want_high: True = drug should be in top-10 (positive); False = should NOT (negative)."""
    b_top = b is not None and b <= 10
    a_top = a is not None and a <= 10
    if want_high:
        if not b_top and a_top: return "✓ surfaced"
        if b_top and not a_top: return "✗ buried"
        if b_top and a_top: return "✓ kept"
        return "  still hidden"
    else:  # want low (drug should NOT be near top)
        if b_top and not a_top: return "✓ demoted"
        if not b_top and a_top: return "✗ surfaced (BAD)"
        if b_top and a_top:
            if a > b: return "▽ partial (better rank)"
            return "✗ still high"
        return "✓ stayed out"


if __name__ == "__main__":
    main()
