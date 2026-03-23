"""
Fit temperature scaling on the held-out validation set.

Run once after Step 0 artifacts are generated:
    python scripts/fit_temperature.py

Outputs:
    data/temperature.json  — {"T": <float>}

Also prints:
    - Fitted T
    - ECE before and after calibration
    - Reliability diagram (saved to results/reliability_diagram.png)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.calibration import TemperatureScaler, confidence_tier
from scipy.special import expit


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).

    ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|

    where bins are equal-width over [0, 1].
    """
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    N = len(probs)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs < hi)
        if i == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        if mask.sum() == 0:
            continue
        bin_conf = probs[mask].mean()
        bin_acc = labels[mask].mean()
        ece += (mask.sum() / N) * abs(bin_acc - bin_conf)

    return float(ece)


def plot_reliability_diagram(
    probs_before: np.ndarray,
    probs_after: np.ndarray,
    labels: np.ndarray,
    save_path: str = "results/reliability_diagram.png",
    n_bins: int = 10,
):
    """Plot reliability diagram comparing before/after calibration."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    def bin_accuracy(probs):
        accs = []
        for i in range(n_bins):
            lo, hi = bins[i], bins[i + 1]
            mask = (probs >= lo) & (probs < hi)
            if i == n_bins - 1:
                mask = (probs >= lo) & (probs <= hi)
            accs.append(labels[mask].mean() if mask.sum() > 0 else np.nan)
        return np.array(accs)

    acc_before = bin_accuracy(probs_before)
    acc_after = bin_accuracy(probs_after)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, accs, probs, title in [
        (axes[0], acc_before, probs_before, "Before Calibration (T=1.0)"),
        (axes[1], acc_after, probs_after, f"After Calibration (Temperature Scaling)"),
    ]:
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
        ax.bar(bin_centers, np.nan_to_num(accs), width=0.09, alpha=0.6, label="Actual accuracy")
        ax.plot(bin_centers, np.nan_to_num(accs), "o-", color="tab:blue")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(title)
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)

    plt.suptitle("BioLink v2 — Reliability Diagram", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Reliability diagram saved to {save_path}")


def main():
    print("Loading val logits and labels...")
    logits = np.load("data/val_logits.npy").astype(np.float64)
    labels = np.load("data/val_labels.npy").astype(np.float64)
    print(f"  {len(logits)} validation samples")

    # Before calibration
    probs_before = expit(logits).astype(np.float32)
    ece_before = expected_calibration_error(probs_before, labels)
    print(f"\nBefore calibration:")
    print(f"  ECE:  {ece_before:.4f}")
    print(f"  Predictions >0.99:   {(probs_before > 0.99).sum()}")
    print(f"  Predictions >0.9999: {(probs_before > 0.9999).sum()}")

    # Fit temperature
    print("\nFitting temperature T...")
    scaler = TemperatureScaler()
    T = scaler.fit(logits, labels, save_path="data/temperature.json")
    print(f"  Fitted T = {T:.4f}")

    # After calibration
    probs_after = scaler.calibrated_proba_batch(logits)
    ece_after = expected_calibration_error(probs_after, labels)
    print(f"\nAfter calibration (T={T:.4f}):")
    print(f"  ECE:  {ece_after:.4f}  (improvement: {ece_before - ece_after:.4f})")
    print(f"  Predictions >0.99:   {(probs_after > 0.99).sum()}")
    print(f"  Predictions >0.9999: {(probs_after > 0.9999).sum()}")

    # Confidence tier distribution
    tiers = [confidence_tier(float(p)) for p in probs_after]
    from collections import Counter
    counts = Counter(tiers)
    print(f"\nConfidence tier distribution (val set):")
    for tier in ["Strong", "Moderate", "Speculative"]:
        print(f"  {tier}: {counts.get(tier, 0)} ({100*counts.get(tier,0)/len(tiers):.1f}%)")

    # Reliability diagram
    plot_reliability_diagram(probs_before, probs_after, labels)

    print(f"\ndata/temperature.json written. T={T:.4f}")
    print("Step 2 complete.")


if __name__ == "__main__":
    main()
