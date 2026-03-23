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
    """Post-hoc temperature scaling calibration."""

    def __init__(self, T: float = 1.0):
        """
        Args:
            T: Temperature parameter. T > 1 softens probabilities (reduces overconfidence).
               T < 1 sharpens probabilities. T = 1 is identity (no calibration).
        """
        if T <= 0:
            raise ValueError(f"Temperature T must be > 0, got {T}")
        self.T = T

    def fit(self, logits: np.ndarray, labels: np.ndarray, save_path: str = "data/temperature.json") -> float:
        """
        Fit temperature T by minimizing negative log-likelihood on validation set.

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

        # Save to JSON
        with open(save_path, "w") as f:
            json.dump({"T": self.T}, f, indent=2)

        return self.T

    def calibrated_proba(self, logit: float) -> float:
        """
        Apply temperature scaling and return calibrated probability.

        Args:
            logit: Raw model logit

        Returns:
            Calibrated probability in (0, 1)
        """
        return float(expit(logit / self.T))

    def calibrated_proba_batch(self, logits: np.ndarray) -> np.ndarray:
        """
        Vectorized calibrated probabilities for an array of logits.

        Args:
            logits: Raw model logits (N,)

        Returns:
            Calibrated probabilities (N,)
        """
        return expit(np.asarray(logits, dtype=np.float64) / self.T).astype(np.float32)

    @classmethod
    def load(cls, path: str = "data/temperature.json") -> "TemperatureScaler":
        """
        Load a previously fitted TemperatureScaler from JSON.

        Args:
            path: Path to temperature.json

        Returns:
            TemperatureScaler with loaded T
        """
        with open(path, "r") as f:
            data = json.load(f)
        return cls(T=float(data["T"]))

    def __repr__(self):
        return f"TemperatureScaler(T={self.T:.4f})"


def confidence_tier(proba: float) -> str:
    """
    Map a calibrated probability to a human-readable confidence tier.

    Tiers:
        Strong:      proba >= 0.80  (green)
        Moderate:    proba >= 0.50  (yellow)
        Speculative: proba <  0.50  (gray)

    Args:
        proba: Calibrated probability in [0, 1]

    Returns:
        "Strong", "Moderate", or "Speculative"
    """
    if proba >= 0.80:
        return "Strong"
    elif proba >= 0.50:
        return "Moderate"
    else:
        return "Speculative"
