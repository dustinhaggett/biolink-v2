"""
Unit tests for core/calibration.py.

Tests:
  - TemperatureScaler.calibrated_proba() output is in [0, 1]
  - Temperature T must be > 0 (constructor validation)
  - calibrated_proba() math: sigmoid(logit / T)
  - Fitted T > 0 after .fit()
  - confidence_tier() boundary conditions per SPEC §3.2:
      Strong      >= 0.80
      Moderate    >= 0.50
      Speculative < 0.50
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


# ---------------------------------------------------------------------------
# confidence_tier() — boundary conditions (SPEC §3.2)
# ---------------------------------------------------------------------------

class TestConfidenceTier:
    # --- Strong boundary ---
    def test_strong_at_0_80(self):
        assert confidence_tier(0.80) == "Strong"

    def test_strong_above_0_80(self):
        assert confidence_tier(0.95) == "Strong"
        assert confidence_tier(1.0) == "Strong"

    def test_not_strong_just_below_0_80(self):
        assert confidence_tier(0.799) != "Strong"

    # --- Moderate boundary ---
    def test_moderate_at_0_50(self):
        assert confidence_tier(0.50) == "Moderate"

    def test_moderate_between_0_50_and_0_80(self):
        assert confidence_tier(0.65) == "Moderate"
        assert confidence_tier(0.799) == "Moderate"

    def test_not_moderate_just_below_0_50(self):
        assert confidence_tier(0.499) != "Moderate"

    # --- Speculative boundary ---
    def test_speculative_below_0_50(self):
        assert confidence_tier(0.499) == "Speculative"
        assert confidence_tier(0.0) == "Speculative"
        assert confidence_tier(0.3) == "Speculative"

    def test_not_speculative_at_0_50(self):
        assert confidence_tier(0.50) != "Speculative"

    # --- Exhaustive spot checks ---
    @pytest.mark.parametrize("proba,expected", [
        (0.0,   "Speculative"),
        (0.1,   "Speculative"),
        (0.499, "Speculative"),
        (0.50,  "Moderate"),
        (0.75,  "Moderate"),
        (0.799, "Moderate"),
        (0.80,  "Strong"),
        (0.9,   "Strong"),
        (1.0,   "Strong"),
    ])
    def test_parametrized_tiers(self, proba, expected):
        assert confidence_tier(proba) == expected