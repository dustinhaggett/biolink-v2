"""
Plain-English explanation generator for BioLink v2 drug repurposing predictions.

Uses the Claude API to generate 2-4 sentence patient-friendly explanations of
why a drug might treat a disease, grounded in model confidence and PubMed evidence.

SPEC §5.1
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

try:
    from anthropic import Anthropic
except ImportError:  # pragma: no cover
    Anthropic = None  # type: ignore[assignment,misc]

# --- Prompt templates (SPEC §5.1) ---

SYSTEM_PROMPT = (
    "You are a biomedical science communicator helping patients and clinicians understand "
    "drug repurposing predictions. Write clear, honest, non-alarmist explanations at a "
    "patient-friendly reading level. Always include an appropriate medical disclaimer. "
    "Never recommend treatment."
)

USER_PROMPT_TEMPLATE = """\
Drug: {drug}
Disease: {disease}
Model confidence: {tier} ({proba:.0%})
PubMed evidence: {pubmed_count} publications
FDA status: {fda_status}

Write a 2-4 sentence explanation of:
1. The known or plausible mechanism linking this drug to this disease
2. What the evidence level means
3. A brief disclaimer

Keep it accessible to a non-specialist. Be honest about uncertainty."""

_FALLBACK_TEMPLATE = (
    "Our model suggests {drug} may have potential relevance for {disease}, "
    "with a {tier} confidence score of {proba:.0%}. "
    "This is a computational prediction based on known drug and disease biology; "
    "it is not a clinical recommendation. "
    "Please consult a qualified healthcare provider before considering any treatment."
)


def explain_prediction(
    drug: str,
    disease: str,
    proba: float,
    tier: str,
    pubmed_count: int,
    fda_status: str,
) -> str:
    """
    Generate a plain-English explanation of a drug-disease prediction.

    Calls the Claude API (claude-opus-4-6) with a patient-friendly system prompt.
    Falls back to a generic templated message if the API call fails.

    Args:
        drug:         Drug name (e.g. "metformin")
        disease:      CTD disease name (e.g. "Diabetes Mellitus, Type 2")
        proba:        Calibrated probability in [0, 1]
        tier:         Confidence tier string — "Strong", "Moderate", or "Speculative"
        pubmed_count: Number of co-occurrence publications found on PubMed
        fda_status:   FDA approval status string (e.g. "FDA Approved")

    Returns:
        2-4 sentence plain-English explanation string. Returns a generic fallback
        message if the Claude API is unavailable or returns an error.
    """
    user_prompt = USER_PROMPT_TEMPLATE.format(
        drug=drug,
        disease=disease,
        tier=tier,
        proba=proba,
        pubmed_count=pubmed_count,
        fda_status=fda_status,
    )

    try:
        if Anthropic is None:
            raise ImportError("anthropic package is not installed")

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            try:
                import streamlit as st
                api_key = st.secrets.get("ANTHROPIC_API_KEY")
            except Exception:
                pass
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")

        client = Anthropic(api_key=api_key)

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            temperature=0.3,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )

        # Extract text from response content blocks
        blocks = getattr(response, "content", [])
        text_parts = [
            block.text
            for block in blocks
            if getattr(block, "type", None) == "text"
        ]
        explanation = "\n".join(text_parts).strip()

        if not explanation:
            raise ValueError("Empty response from Claude API")

        return explanation

    except ImportError:
        logger.warning("anthropic package not installed — using fallback explanation")
    except RuntimeError as exc:
        logger.warning("Config error in explain_prediction: %s — using fallback", exc)
    except Exception as exc:
        logger.warning("Claude API call failed in explain_prediction: %s — using fallback", exc)

    # Graceful fallback: generic template that still conveys key facts
    return _FALLBACK_TEMPLATE.format(
        drug=drug,
        disease=disease,
        tier=tier,
        proba=proba,
    )