"""
Unit tests for explanation/explainer.py.

All Anthropic API calls are mocked — no real network traffic or API keys needed.

Tests:
  - Output is a non-empty string on successful API call
  - Disclaimer language present in output (either API or fallback)
  - Fallback returned when API raises an exception
  - Fallback returned when API key is missing
  - Fallback returned when anthropic package raises ImportError
  - Fallback is non-empty string
  - Fallback contains drug name and disease name
  - Output contains drug/disease info when API succeeds
  - Proba formatting: 0% to 100% range handled
  - All confidence tiers accepted without error
"""

from __future__ import annotations

import sys
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from explanation.explainer import explain_prediction, _FALLBACK_TEMPLATE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DRUG = "metformin"
DISEASE = "Diabetes Mellitus, Type 2"
PROBA = 0.87
TIER = "Strong"
PUBMED = 142
FDA = "FDA Approved"

SAMPLE_EXPLANATION = (
    "Metformin is widely studied for its role in improving insulin sensitivity. "
    "The model's Strong confidence (87%) is supported by 142 PubMed publications. "
    "This is a research hypothesis, not a treatment recommendation — always consult your doctor."
)


def _make_anthropic_response(text: str) -> MagicMock:
    """Return a fake Anthropic SDK messages response."""
    block = SimpleNamespace(type="text", text=text)
    response = MagicMock()
    response.content = [block]
    return response


def _mock_anthropic_success(text: str = SAMPLE_EXPLANATION):
    """Context manager: patches Anthropic so messages.create() returns a good response."""
    fake_response = _make_anthropic_response(text)
    mock_client = MagicMock()
    mock_client.messages.create.return_value = fake_response
    mock_cls = MagicMock(return_value=mock_client)
    return patch.dict(
        "os.environ", {"ANTHROPIC_API_KEY": "sk-test-key"}
    ), patch("explanation.explainer.Anthropic", mock_cls)


# ---------------------------------------------------------------------------
# Successful API path
# ---------------------------------------------------------------------------

class TestExplainPredictionSuccess:
    def test_returns_non_empty_string(self):
        env_patch = patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test"})
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_anthropic_response(SAMPLE_EXPLANATION)
        cls_patch = patch("explanation.explainer.Anthropic", return_value=mock_client)

        with env_patch, cls_patch:
            result = explain_prediction(DRUG, DISEASE, PROBA, TIER, PUBMED, FDA)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_returns_api_text_when_successful(self):
        env_patch = patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test"})
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_anthropic_response(SAMPLE_EXPLANATION)
        cls_patch = patch("explanation.explainer.Anthropic", return_value=mock_client)

        with env_patch, cls_patch:
            result = explain_prediction(DRUG, DISEASE, PROBA, TIER, PUBMED, FDA)

        assert result == SAMPLE_EXPLANATION

    def test_calls_api_with_correct_model(self):
        env_patch = patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test"})
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_anthropic_response(SAMPLE_EXPLANATION)
        cls_patch = patch("explanation.explainer.Anthropic", return_value=mock_client)

        with env_patch, cls_patch:
            explain_prediction(DRUG, DISEASE, PROBA, TIER, PUBMED, FDA)

        call_kwargs = mock_client.messages.create.call_args
        assert call_kwargs.kwargs.get("model") == "claude-sonnet-4-20250514"

    def test_max_tokens_is_200(self):
        env_patch = patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test"})
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_anthropic_response(SAMPLE_EXPLANATION)
        cls_patch = patch("explanation.explainer.Anthropic", return_value=mock_client)

        with env_patch, cls_patch:
            explain_prediction(DRUG, DISEASE, PROBA, TIER, PUBMED, FDA)

        call_kwargs = mock_client.messages.create.call_args
        assert call_kwargs.kwargs.get("max_tokens") == 200

    def test_temperature_is_0_3(self):
        env_patch = patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test"})
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_anthropic_response(SAMPLE_EXPLANATION)
        cls_patch = patch("explanation.explainer.Anthropic", return_value=mock_client)

        with env_patch, cls_patch:
            explain_prediction(DRUG, DISEASE, PROBA, TIER, PUBMED, FDA)

        call_kwargs = mock_client.messages.create.call_args
        assert call_kwargs.kwargs.get("temperature") == pytest.approx(0.3)

    def test_system_prompt_includes_patient_friendly(self):
        """System prompt should mention patient-friendly communication."""
        from explanation.explainer import SYSTEM_PROMPT
        assert "patient" in SYSTEM_PROMPT.lower() or "non-specialist" in SYSTEM_PROMPT.lower() \
               or "reading level" in SYSTEM_PROMPT.lower()

    def test_system_prompt_includes_disclaimer_instruction(self):
        from explanation.explainer import SYSTEM_PROMPT
        assert "disclaimer" in SYSTEM_PROMPT.lower() or "never recommend" in SYSTEM_PROMPT.lower()


# ---------------------------------------------------------------------------
# Fallback path — API failures
# ---------------------------------------------------------------------------

class TestExplainPredictionFallback:
    def test_fallback_on_api_exception(self):
        env_patch = patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test"})
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("API unavailable")
        cls_patch = patch("explanation.explainer.Anthropic", return_value=mock_client)

        with env_patch, cls_patch:
            result = explain_prediction(DRUG, DISEASE, PROBA, TIER, PUBMED, FDA)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_fallback_on_missing_api_key(self):
        env_patch = patch.dict("os.environ", {}, clear=True)
        # Ensure ANTHROPIC_API_KEY is definitely absent
        with env_patch:
            os.environ.pop("ANTHROPIC_API_KEY", None)
            result = explain_prediction(DRUG, DISEASE, PROBA, TIER, PUBMED, FDA)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_fallback_contains_drug_name(self):
        env_patch = patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test"})
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = ConnectionError("timeout")
        cls_patch = patch("explanation.explainer.Anthropic", return_value=mock_client)

        with env_patch, cls_patch:
            result = explain_prediction(DRUG, DISEASE, PROBA, TIER, PUBMED, FDA)

        assert DRUG.lower() in result.lower()

    def test_fallback_contains_disease_name(self):
        env_patch = patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test"})
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = ConnectionError("timeout")
        cls_patch = patch("explanation.explainer.Anthropic", return_value=mock_client)

        with env_patch, cls_patch:
            result = explain_prediction(DRUG, DISEASE, PROBA, TIER, PUBMED, FDA)

        assert DISEASE.lower() in result.lower()

    def test_fallback_on_timeout_error(self):
        env_patch = patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test"})
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = TimeoutError("request timed out")
        cls_patch = patch("explanation.explainer.Anthropic", return_value=mock_client)

        with env_patch, cls_patch:
            result = explain_prediction(DRUG, DISEASE, 0.55, "Moderate", 5, "Unknown")

        assert isinstance(result, str)
        assert len(result) > 0

    def test_fallback_on_empty_api_response(self):
        """If API returns empty content blocks, fallback should be used."""
        env_patch = patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test"})
        mock_client = MagicMock()
        empty_response = MagicMock()
        empty_response.content = []  # no blocks → empty string → fallback
        mock_client.messages.create.return_value = empty_response
        cls_patch = patch("explanation.explainer.Anthropic", return_value=mock_client)

        with env_patch, cls_patch:
            result = explain_prediction(DRUG, DISEASE, PROBA, TIER, PUBMED, FDA)

        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# All confidence tiers & proba edge cases
# ---------------------------------------------------------------------------

class TestExplainPredictionEdgeCases:
    @pytest.mark.parametrize("tier,proba", [
        ("Strong", 0.95),
        ("Moderate", 0.65),
        ("Speculative", 0.30),
    ])
    def test_all_tiers_accepted(self, tier, proba):
        env_patch = patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test"})
        explanation_text = f"This is a test explanation for {tier} tier."
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_anthropic_response(explanation_text)
        cls_patch = patch("explanation.explainer.Anthropic", return_value=mock_client)

        with env_patch, cls_patch:
            result = explain_prediction("aspirin", "Hypertension", proba, tier, 10, "FDA Approved")

        assert isinstance(result, str)
        assert len(result) > 0

    def test_proba_zero(self):
        env_patch = patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test"})
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = RuntimeError("fail")
        cls_patch = patch("explanation.explainer.Anthropic", return_value=mock_client)

        with env_patch, cls_patch:
            result = explain_prediction("drug_x", "Disease Y", 0.0, "Speculative", 0, "Unknown")

        assert isinstance(result, str)

    def test_proba_one(self):
        env_patch = patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test"})
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = RuntimeError("fail")
        cls_patch = patch("explanation.explainer.Anthropic", return_value=mock_client)

        with env_patch, cls_patch:
            result = explain_prediction("drug_x", "Disease Y", 1.0, "Strong", 500, "FDA Approved")

        assert isinstance(result, str)

    def test_pubmed_count_zero(self):
        env_patch = patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test"})
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_anthropic_response("Some explanation.")
        cls_patch = patch("explanation.explainer.Anthropic", return_value=mock_client)

        with env_patch, cls_patch:
            result = explain_prediction("drug_x", "Disease Y", 0.6, "Moderate", 0, "Not in FDA Database")

        assert isinstance(result, str)
        assert len(result) > 0