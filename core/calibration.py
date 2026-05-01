"""
Temperature scaling calibration for BioLink v2.

Post-hoc calibration that scales raw logits by a learned temperature T
to produce honest probability estimates. T is fit by minimizing NLL on
the held-out validation set. Requires only one scalar parameter.
"""

import json
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit  # numerically stable sigmoid


class TemperatureScaler:
    """Post-hoc temperature scaling + prior correction.

    Two-parameter calibration:
      1) Temperature T scales logits to fix overconfidence within the training
         distribution (NLL minimization on a held-out 50/50 val set).
      2) prior_shift corrects for the train-vs-inference prior mismatch. The
         model trains on 1:1 positives:negatives but at inference time the
         base rate of "this drug treats this disease" is ~1% (a disease has
         tens of therapeutic candidates among ~7000 drugs). Without this shift
         every top-20 result ceilings at 99-100% and confidence is uninformative.

    The shift is subtracted from logit/T before the sigmoid:
        P = sigmoid(logit/T - prior_shift)
    where prior_shift = log_odds(p_train) - log_odds(p_real).
    For p_train=0.5 and p_real=0.01: prior_shift ≈ 4.595.

    Ranking is preserved exactly — both transformations are monotonic.
    """

    def __init__(self, T: float = 1.0, prior_shift: float = 0.0):
        """
        Args:
            T: Temperature parameter. T > 1 softens probabilities (reduces overconfidence).
               T < 1 sharpens probabilities. T = 1 is identity (no calibration).
            prior_shift: Logit shift applied AFTER temperature scaling to correct
                for train-vs-inference prior mismatch. shift > 0 reduces probabilities
                (assumes real prior is lower than training prior). 0 = no correction.
        """
        if T <= 0:
            raise ValueError(f"Temperature T must be > 0, got {T}")
        self.T = T
        self.prior_shift = float(prior_shift)

    @staticmethod
    def shift_for_prior(p_train: float = 0.5, p_real: float = 0.01) -> float:
        """Compute prior_shift from explicit train/real priors.

        Use when you know the actual class ratios and want a principled value
        rather than picking a number.
        """
        if not (0 < p_train < 1) or not (0 < p_real < 1):
            raise ValueError("priors must be strictly between 0 and 1")
        return float(np.log(p_train / (1 - p_train)) - np.log(p_real / (1 - p_real)))

    def fit(self, logits: np.ndarray, labels: np.ndarray, save_path: str = "data/temperature.json") -> float:
        """
        Fit temperature T by minimizing negative log-likelihood on validation set.

        Note: this only fits T. prior_shift is set via shift_for_prior() or directly.

        Args:
            logits: Raw model logits (N,)
            labels: Binary ground truth labels (N,)
            save_path: Where to save the fitted T as JSON

        Returns:
            Fitted temperature T
        """
        logits = np.asarray(logits, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.float64)

        def nll(T):
            scaled = logits / T
            # Numerically stable binary cross-entropy
            # BCE = -[y * log(sigma(z)) + (1-y) * log(1 - sigma(z))]
            probs = expit(scaled)
            probs = np.clip(probs, 1e-7, 1 - 1e-7)
            return -np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))

        result = minimize_scalar(nll, bounds=(0.01, 20.0), method="bounded")
        self.T = float(result.x)

        self.save(save_path)
        return self.T

    def save(self, path: str = "data/temperature.json") -> None:
        """Persist {T, prior_shift} to JSON."""
        with open(path, "w") as f:
            json.dump({"T": self.T, "prior_shift": self.prior_shift}, f, indent=2)

    def calibrated_proba(self, logit: float) -> float:
        """
        Apply temperature scaling + prior correction; return calibrated probability.

        Args:
            logit: Raw model logit

        Returns:
            Calibrated probability in (0, 1)
        """
        return float(expit(logit / self.T - self.prior_shift))

    def calibrated_proba_batch(self, logits: np.ndarray) -> np.ndarray:
        """
        Vectorized calibrated probabilities for an array of logits.

        Args:
            logits: Raw model logits (N,)

        Returns:
            Calibrated probabilities (N,)
        """
        scaled = np.asarray(logits, dtype=np.float64) / self.T - self.prior_shift
        return expit(scaled).astype(np.float32)

    @classmethod
    def load(cls, path: str = "data/temperature.json") -> "TemperatureScaler":
        """
        Load a previously fitted TemperatureScaler from JSON.

        Backward compatible: pre-prior-correction JSON files (only {"T": ...}) load
        with prior_shift=0, preserving original behavior.

        Args:
            path: Path to temperature.json

        Returns:
            TemperatureScaler with loaded T (and prior_shift if present)
        """
        with open(path, "r") as f:
            data = json.load(f)
        return cls(
            T=float(data["T"]),
            prior_shift=float(data.get("prior_shift", 0.0)),
        )

    def __repr__(self):
        return f"TemperatureScaler(T={self.T:.4f}, prior_shift={self.prior_shift:.4f})"


def confidence_tier(proba: float) -> str:
    """
    Map a calibrated probability to a human-readable confidence tier.

    Thresholds (post prior-correction, 2026-05-01):
        Strong:      proba >= 0.30  (~30x baseline rate of 1%)
        Moderate:    proba >= 0.10  (~10x baseline rate)
        Speculative: proba <  0.10  (near-baseline / noise)

    Original thresholds (0.80 / 0.50) assumed a saturated 1:1-trained
    distribution and labeled almost everything as Strong. After applying
    prior correction (TemperatureScaler.prior_shift), real probabilities
    span 0.0–0.99; tier thresholds adjusted to match.

    Args:
        proba: Calibrated probability in [0, 1]

    Returns:
        "Strong", "Moderate", or "Speculative"
    """
    if proba >= 0.30:
        return "Strong"
    elif proba >= 0.10:
        return "Moderate"
    else:
        return "Speculative"
