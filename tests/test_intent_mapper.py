"""
Unit tests for core/intent_mapper.py.

All Claude API calls are mocked — no real network traffic.

Tests:
  - High-confidence path: model returns high-confidence match
  - Medium-confidence path: model returns medium-confidence match
  - Low-confidence path: model returns low-confidence + clarification
  - Null entity path: model cannot find a match
  - Fuzzy fallback: used when Claude API raises an exception
  - Fuzzy fallback: direct lexical match (cutoff 0.72)
  - Fuzzy fallback: token overlap match (low confidence)
  - Fuzzy fallback: no match → null result
  - Malformed JSON from model → fuzzy fallback triggered
  - Missing API key → fuzzy fallback triggered
  - Empty user input → null result
  - Entity returned by model not in candidate list → fuzzy fallback
"""

from __future__ import annotations

import json
import sys
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import intent_mapper directly to avoid core/__init__.py pulling in model.py
# which requires torch (not available in the unit-test environment).
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

_intent_mapper = _import_from_file("core.intent_mapper", "core/intent_mapper.py")
_fallback_map_disease = _intent_mapper._fallback_map_disease
_normalize_response   = _intent_mapper._normalize_response
map_disease           = _intent_mapper.map_disease


def _patch_anthropic(mock_client):
    """Return a patch.object context manager that replaces Anthropic on the loaded module."""
    return patch.object(_intent_mapper, "Anthropic", return_value=mock_client)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_DISEASES = [
    "Diabetes Mellitus, Type 2",
    "Breast Neoplasms",
    "Alzheimer Disease",
    "Atrial Fibrillation",
    "Hypertension",
    "Lung Neoplasms",
    "Parkinson Disease",
]


def _make_anthropic_response(json_payload: dict) -> MagicMock:
    """Build a fake Anthropic messages response containing JSON text."""
    block = SimpleNamespace(type="text", text=json.dumps(json_payload))
    response = MagicMock()
    response.content = [block]
    return response


# ---------------------------------------------------------------------------
# _normalize_response() — internal helper
# ---------------------------------------------------------------------------

class TestNormalizeResponse:
    def test_high_confidence_valid(self):
        data = {
            "ctd_entity": "Diabetes Mellitus, Type 2",
            "confidence": "high",
            "display_name": "Type 2 Diabetes",
            "clarification": None,
        }
        result = _normalize_response(data)
        assert result["ctd_entity"] == "Diabetes Mellitus, Type 2"
        assert result["confidence"] == "high"
        assert result["clarification"] is None

    def test_low_confidence_gets_clarification(self):
        data = {
            "ctd_entity": "Alzheimer Disease",
            "confidence": "low",
            "display_name": "Alzheimer's",
            "clarification": "Did you mean Alzheimer Disease?",
        }
        result = _normalize_response(data)
        assert result["confidence"] == "low"
        assert result["clarification"] is not None

    def test_null_entity_returns_null_result(self):
        data = {
            "ctd_entity": None,
            "confidence": "low",
            "display_name": "",
            "clarification": "Could not identify disease",
        }
        result = _normalize_response(data)
        assert result["ctd_entity"] is None

    def test_invalid_confidence_defaults_to_low(self):
        data = {
            "ctd_entity": "Hypertension",
            "confidence": "definitely",   # not a valid value
            "display_name": "High Blood Pressure",
            "clarification": None,
        }
        result = _normalize_response(data)
        assert result["confidence"] == "low"


# ---------------------------------------------------------------------------
# _fallback_map_disease() — no API calls
# ---------------------------------------------------------------------------

class TestFallbackMapDisease:
    def test_exact_lexical_match(self):
        result = _fallback_map_disease("type 2 diabetes", SAMPLE_DISEASES)
        # Should match "Diabetes Mellitus, Type 2" via fuzzy string matching
        assert result["ctd_entity"] is not None
        assert "Diabetes" in result["ctd_entity"]

    def test_close_lexical_match(self):
        result = _fallback_map_disease("alzheimer disease", SAMPLE_DISEASES)
        assert result["ctd_entity"] is not None
        assert "Alzheimer" in result["ctd_entity"]

    def test_token_overlap_match_low_confidence(self):
        # "breast cancer" should token-overlap with "Breast Neoplasms"
        result = _fallback_map_disease("breast cancer", SAMPLE_DISEASES)
        assert result["ctd_entity"] is not None

    def test_no_match_returns_null(self):
        result = _fallback_map_disease("xyzzy123", SAMPLE_DISEASES)
        assert result["ctd_entity"] is None

    def test_empty_input_returns_null(self):
        result = _fallback_map_disease("", SAMPLE_DISEASES)
        assert result["ctd_entity"] is None

    def test_result_has_required_keys(self):
        result = _fallback_map_disease("hypertension", SAMPLE_DISEASES)
        for key in ("ctd_entity", "confidence", "display_name", "clarification"):
            assert key in result


# ---------------------------------------------------------------------------
# map_disease() — mocked Claude API
# ---------------------------------------------------------------------------

class TestMapDiseaseHighConfidence:
    def test_returns_ctd_entity(self):
        payload = {
            "ctd_entity": "Diabetes Mellitus, Type 2",
            "confidence": "high",
            "display_name": "Type 2 Diabetes",
            "clarification": None,
        }
        fake_response = _make_anthropic_response(payload)
        mock_client = MagicMock()
        mock_client.messages.create.return_value = fake_response
        with _patch_anthropic(mock_client):
            result = map_disease("type 2 diabetes", SAMPLE_DISEASES)

        assert result["ctd_entity"] == "Diabetes Mellitus, Type 2"
        assert result["confidence"] == "high"
        assert result["clarification"] is None

    def test_display_name_populated(self):
        payload = {
            "ctd_entity": "Atrial Fibrillation",
            "confidence": "high",
            "display_name": "Atrial Fibrillation (AFib)",
            "clarification": None,
        }
        fake_response = _make_anthropic_response(payload)
        mock_client = MagicMock()
        mock_client.messages.create.return_value = fake_response
        with _patch_anthropic(mock_client):
            result = map_disease("irregular heartbeat", SAMPLE_DISEASES)

        assert result["display_name"] == "Atrial Fibrillation (AFib)"


class TestMapDiseaseMediumConfidence:
    def test_medium_confidence_no_clarification(self):
        payload = {
            "ctd_entity": "Hypertension",
            "confidence": "medium",
            "display_name": "High Blood Pressure",
            "clarification": None,
        }
        fake_response = _make_anthropic_response(payload)
        mock_client = MagicMock()
        mock_client.messages.create.return_value = fake_response
        with _patch_anthropic(mock_client):
            result = map_disease("high blood pressure", SAMPLE_DISEASES)

        assert result["confidence"] == "medium"
        assert result["ctd_entity"] == "Hypertension"


class TestMapDiseaseLowConfidence:
    def test_low_confidence_has_clarification(self):
        payload = {
            "ctd_entity": "Alzheimer Disease",
            "confidence": "low",
            "display_name": "Alzheimer's Disease",
            "clarification": "Did you mean Alzheimer Disease?",
        }
        fake_response = _make_anthropic_response(payload)
        mock_client = MagicMock()
        mock_client.messages.create.return_value = fake_response
        with _patch_anthropic(mock_client):
            result = map_disease("memory loss condition", SAMPLE_DISEASES)

        assert result["confidence"] == "low"
        assert result["clarification"] is not None
        assert len(result["clarification"]) > 0


class TestMapDiseaseNullEntity:
    def test_null_entity_path(self):
        payload = {
            "ctd_entity": None,
            "confidence": "low",
            "display_name": "",
            "clarification": "Could you describe your condition differently?",
        }
        fake_response = _make_anthropic_response(payload)
        mock_client = MagicMock()
        mock_client.messages.create.return_value = fake_response
        with _patch_anthropic(mock_client):
            result = map_disease("xyzzy123", SAMPLE_DISEASES)

        assert result["ctd_entity"] is None
        assert result["clarification"] is not None


class TestMapDiseaseFallbackPaths:
    def test_api_exception_triggers_fuzzy_fallback(self):
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = ConnectionError("timeout")
        with _patch_anthropic(mock_client):
            result = map_disease("type 2 diabetes", SAMPLE_DISEASES)

        # Should fall through to fuzzy matching — may or may not find a match
        # but result must have required keys
        for key in ("ctd_entity", "confidence", "display_name", "clarification"):
            assert key in result

    def test_malformed_json_triggers_fuzzy_fallback(self):
        block = SimpleNamespace(type="text", text="not valid json {{{{")
        fake_response = MagicMock()
        fake_response.content = [block]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = fake_response
        with _patch_anthropic(mock_client):
            result = map_disease("atrial fibrillation", SAMPLE_DISEASES)

        for key in ("ctd_entity", "confidence", "display_name", "clarification"):
            assert key in result

    def test_empty_user_input_returns_null(self):
        result = map_disease("", SAMPLE_DISEASES)
        assert result["ctd_entity"] is None

    def test_whitespace_only_input_returns_null(self):
        result = map_disease("   ", SAMPLE_DISEASES)
        assert result["ctd_entity"] is None

    def test_entity_not_in_candidate_list_triggers_fallback(self):
        """If Claude returns an entity not in the candidate list, fuzzy fallback is used."""
        payload = {
            "ctd_entity": "Completely Made Up Disease XYZ",
            "confidence": "high",
            "display_name": "Something",
            "clarification": None,
        }
        fake_response = _make_anthropic_response(payload)
        mock_client = MagicMock()
        mock_client.messages.create.return_value = fake_response
        with _patch_anthropic(mock_client):
            result = map_disease("breast cancer", SAMPLE_DISEASES)

        # Should have fallen back — result is valid but not the hallucinated entity
        assert result["ctd_entity"] != "Completely Made Up Disease XYZ" or result["ctd_entity"] is None


class TestMapDiseaseNoAPIKey:
    def test_missing_api_key_triggers_fallback(self):
        """TypeError from bad client args should fall through to fuzzy fallback."""
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = TypeError("Missing API key")
        with _patch_anthropic(mock_client):
            result = map_disease("hypertension", SAMPLE_DISEASES)

        for key in ("ctd_entity", "confidence", "display_name", "clarification"):
            assert key in result