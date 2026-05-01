"""
Unit tests for core/calibration.py.

Tests:
  - TemperatureScaler.calibrated_proba() output is in [0, 1]
  - Temperature T must be > 0 (constructor validation)
  - calibrated_proba() math: sigmoid(logit / T)
  - Fitted T > 0 after .fit()
  - confidence_tier() boundary conditions per SPEC §3.2 (revised 2026-05-01):
      Strong      >= 0.30   (post prior-correction; ~30x baseline rate of 1%)
      Moderate    >= 0.10
      Speculative <  0.10
      (Original thresholds were 0.80 / 0.50, calibrated for the saturated
      pre-prior-correction distribution. See core.calibration.confidence_tier
      docstring for rationale.)
  - TemperatureScaler.load() round-trips through JSON
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest
from scipy.special import expit

# Ensure the repo root is importable regardless of working directory
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import the module directly to avoid core/__init__.py which eagerly imports
# model.py (requires torch — not available in the test environment).
import importlib.util as _ilu
import pathlib as _pl

def _import_from_file(name, rel_path):
    spec = _ilu.spec_from_file_location(
        name,
        str(_pl.Path(__file__).parent.parent / rel_path),
    )
    mod = _ilu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_calibration = _import_from_file("core.calibration", "core/calibration.py")
TemperatureScaler = _calibration.TemperatureScaler
confidence_tier = _calibration.confidence_tier


# ---------------------------------------------------------------------------
# TemperatureScaler — constructor
# ---------------------------------------------------------------------------

class TestTemperatureScalerConstructor:
    def test_default_T_is_one(self):
        scaler = TemperatureScaler()
        assert scaler.T == 1.0

    def test_custom_T(self):
        scaler = TemperatureScaler(T=2.5)
        assert scaler.T == pytest.approx(2.5)

    def test_invalid_T_zero_raises(self):
        with pytest.raises(ValueError):
            TemperatureScaler(T=0.0)

    def test_invalid_T_negative_raises(self):
        with pytest.raises(ValueError):
            TemperatureScaler(T=-1.0)


# ---------------------------------------------------------------------------
# TemperatureScaler — calibrated_proba()
# ---------------------------------------------------------------------------

class TestCalibratedProba:
    """calibrated_proba(logit) = sigmoid(logit / T)"""

    def test_output_in_unit_interval_positive_logit(self):
        scaler = TemperatureScaler(T=1.0)
        p = scaler.calibrated_proba(2.5)
        assert 0.0 < p < 1.0

    def test_output_in_unit_interval_negative_logit(self):
        scaler = TemperatureScaler(T=1.0)
        p = scaler.calibrated_proba(-3.0)
        assert 0.0 < p < 1.0

    def test_output_in_unit_interval_zero_logit(self):
        scaler = TemperatureScaler(T=1.0)
        p = scaler.calibrated_proba(0.0)
        assert p == pytest.approx(0.5)

    def test_matches_sigmoid_formula_T1(self):
        scaler = TemperatureScaler(T=1.0)
        logit = 1.5
        expected = float(expit(logit / 1.0))
        assert scaler.calibrated_proba(logit) == pytest.approx(expected, abs=1e-6)

    def test_matches_sigmoid_formula_T2(self):
        """Higher T should soften (move toward 0.5) compared to T=1."""
        logit = 3.0
        scaler_t1 = TemperatureScaler(T=1.0)
        scaler_t2 = TemperatureScaler(T=2.0)
        p1 = scaler_t1.calibrated_proba(logit)
        p2 = scaler_t2.calibrated_proba(logit)
        expected_t2 = float(expit(logit / 2.0))
        assert scaler_t2.calibrated_proba(logit) == pytest.approx(expected_t2, abs=1e-6)
        # Higher T → softer probability (closer to 0.5)
        assert p2 < p1

    def test_large_positive_logit_approaches_one(self):
        scaler = TemperatureScaler(T=1.0)
        assert scaler.calibrated_proba(100.0) > 0.999

    def test_large_negative_logit_approaches_zero(self):
        scaler = TemperatureScaler(T=1.0)
        assert scaler.calibrated_proba(-100.0) < 0.001

    def test_returns_float(self):
        scaler = TemperatureScaler(T=1.5)
        assert isinstance(scaler.calibrated_proba(1.0), float)


# ---------------------------------------------------------------------------
# TemperatureScaler — fit()
# ---------------------------------------------------------------------------

class TestFit:
    def _make_val_data(self, n: int = 200, seed: int = 42):
        rng = np.random.default_rng(seed)
        labels = rng.integers(0, 2, size=n).astype(float)
        # Logits correlated with labels so fitting is meaningful
        logits = labels * 2.0 + rng.normal(0, 1, size=n)
        return logits, labels

    def test_fitted_T_is_positive(self, tmp_path):
        logits, labels = self._make_val_data()
        scaler = TemperatureScaler()
        save_path = str(tmp_path / "temperature.json")
        T = scaler.fit(logits, labels, save_path=save_path)
        assert T > 0.0

    def test_fitted_T_stored_on_instance(self, tmp_path):
        logits, labels = self._make_val_data()
        scaler = TemperatureScaler()
        save_path = str(tmp_path / "temperature.json")
        T = scaler.fit(logits, labels, save_path=save_path)
        assert scaler.T == pytest.approx(T)

    def test_fit_writes_json(self, tmp_path):
        logits, labels = self._make_val_data()
        scaler = TemperatureScaler()
        save_path = str(tmp_path / "temperature.json")
        scaler.fit(logits, labels, save_path=save_path)
        assert os.path.exists(save_path)
        with open(save_path) as f:
            data = json.load(f)
        assert "T" in data
        assert data["T"] > 0.0


# ---------------------------------------------------------------------------
# TemperatureScaler — load() (JSON round-trip)
# ---------------------------------------------------------------------------

class TestLoad:
    def test_load_round_trip(self, tmp_path):
        save_path = str(tmp_path / "temperature.json")
        with open(save_path, "w") as f:
            json.dump({"T": 1.7}, f)
        scaler = TemperatureScaler.load(save_path)
        assert scaler.T == pytest.approx(1.7)

    def test_load_preserves_calibration(self, tmp_path):
        T_value = 2.3
        save_path = str(tmp_path / "temperature.json")
        with open(save_path, "w") as f:
            json.dump({"T": T_value}, f)
        scaler = TemperatureScaler.load(save_path)
        logit = 1.0
        expected = float(expit(logit / T_value))
        assert scaler.calibrated_proba(logit) == pytest.approx(expected, abs=1e-6)

    def test_load_legacy_json_no_prior_shift(self, tmp_path):
        """Backward compat: pre-prior-correction JSON files only have {"T": ...}."""
        save_path = str(tmp_path / "temperature.json")
        with open(save_path, "w") as f:
            json.dump({"T": 1.5}, f)  # No prior_shift key
        scaler = TemperatureScaler.load(save_path)
        assert scaler.T == pytest.approx(1.5)
        assert scaler.prior_shift == 0.0

    def test_load_with_prior_shift(self, tmp_path):
        save_path = str(tmp_path / "temperature.json")
        with open(save_path, "w") as f:
            json.dump({"T": 1.5, "prior_shift": 4.6}, f)
        scaler = TemperatureScaler.load(save_path)
        assert scaler.T == pytest.approx(1.5)
        assert scaler.prior_shift == pytest.approx(4.6)


# ---------------------------------------------------------------------------
# TemperatureScaler — prior_shift correction
# ---------------------------------------------------------------------------

class TestPriorShift:
    def test_default_prior_shift_is_zero(self):
        """Backward compat: existing callers see no behavior change."""
        scaler = TemperatureScaler(T=1.0)
        assert scaler.prior_shift == 0.0
        # Without shift, calibrated_proba(0) = sigmoid(0) = 0.5 (unchanged)
        assert scaler.calibrated_proba(0.0) == pytest.approx(0.5)

    def test_positive_shift_reduces_probability(self):
        """A positive shift represents lower real prior → lower calibrated probs."""
        unshifted = TemperatureScaler(T=1.0, prior_shift=0.0)
        shifted = TemperatureScaler(T=1.0, prior_shift=4.6)
        logit = 2.0
        assert shifted.calibrated_proba(logit) < unshifted.calibrated_proba(logit)

    def test_shift_preserves_ranking(self):
        """Monotonic transform — ranking by probability must equal ranking by logit."""
        scaler = TemperatureScaler(T=1.384, prior_shift=4.595)
        logits = [5.0, 3.5, 1.0, -2.0, 8.0]
        probs = [scaler.calibrated_proba(z) for z in logits]
        # Argsort of probs descending should equal argsort of logits descending
        prob_order = sorted(range(len(probs)), key=lambda i: -probs[i])
        logit_order = sorted(range(len(logits)), key=lambda i: -logits[i])
        assert prob_order == logit_order

    def test_shift_for_prior_known_values(self):
        """log_odds(0.5) - log_odds(0.01) ≈ 4.5951"""
        shift = TemperatureScaler.shift_for_prior(p_train=0.5, p_real=0.01)
        assert shift == pytest.approx(4.5951, abs=1e-3)

    def test_shift_for_prior_zero_when_priors_match(self):
        shift = TemperatureScaler.shift_for_prior(p_train=0.3, p_real=0.3)
        assert shift == pytest.approx(0.0, abs=1e-10)

    def test_shift_for_prior_rejects_invalid(self):
        with pytest.raises(ValueError):
            TemperatureScaler.shift_for_prior(p_train=0.5, p_real=0.0)
        with pytest.raises(ValueError):
            TemperatureScaler.shift_for_prior(p_train=1.0, p_real=0.5)

    def test_save_persists_prior_shift(self, tmp_path):
        save_path = str(tmp_path / "temperature.json")
        scaler = TemperatureScaler(T=1.5, prior_shift=4.6)
        scaler.save(save_path)
        with open(save_path) as f:
            data = json.load(f)
        assert data["T"] == pytest.approx(1.5)
        assert data["prior_shift"] == pytest.approx(4.6)

    def test_calibrated_proba_batch_applies_shift(self):
        scaler = TemperatureScaler(T=1.0, prior_shift=4.6)
        logits = np.array([2.0, 3.0, 5.0])
        probs = scaler.calibrated_proba_batch(logits)
        # Each element should match scalar version
        for i, z in enumerate(logits):
            assert probs[i] == pytest.approx(scaler.calibrated_proba(float(z)), abs=1e-5)


# ---------------------------------------------------------------------------
# confidence_tier() — boundary conditions (SPEC §3.2)
# ---------------------------------------------------------------------------

class TestConfidenceTier:
    # --- Strong boundary (>= 0.30) ---
    def test_strong_at_0_30(self):
        assert confidence_tier(0.30) == "Strong"

    def test_strong_above_0_30(self):
        assert confidence_tier(0.50) == "Strong"
        assert confidence_tier(0.95) == "Strong"
        assert confidence_tier(1.0) == "Strong"

    def test_not_strong_just_below_0_30(self):
        assert confidence_tier(0.299) != "Strong"

    # --- Moderate boundary (>= 0.10) ---
    def test_moderate_at_0_10(self):
        assert confidence_tier(0.10) == "Moderate"

    def test_moderate_between_0_10_and_0_30(self):
        assert confidence_tier(0.15) == "Moderate"
        assert confidence_tier(0.299) == "Moderate"

    def test_not_moderate_just_below_0_10(self):
        assert confidence_tier(0.099) != "Moderate"

    # --- Speculative boundary (< 0.10) ---
    def test_speculative_below_0_10(self):
        assert confidence_tier(0.099) == "Speculative"
        assert confidence_tier(0.0) == "Speculative"
        assert confidence_tier(0.05) == "Speculative"

    def test_not_speculative_at_0_10(self):
        assert confidence_tier(0.10) != "Speculative"

    # --- Exhaustive spot checks ---
    @pytest.mark.parametrize("proba,expected", [
        (0.0,   "Speculative"),
        (0.05,  "Speculative"),
        (0.099, "Speculative"),
        (0.10,  "Moderate"),
        (0.20,  "Moderate"),
        (0.299, "Moderate"),
        (0.30,  "Strong"),
        (0.50,  "Strong"),
        (0.9,   "Strong"),
        (1.0,   "Strong"),
    ])
    def test_parametrized_tiers(self, proba, expected):
        assert confidence_tier(proba) == expected