"""Disease intent mapping for BioLink v2."""

from __future__ import annotations

import json
import difflib
import re
from pathlib import Path
from typing import Any, Dict, List

try:
    from anthropic import Anthropic
except ImportError:  # pragma: no cover - depends on runtime environment
    Anthropic = None


DEFAULT_DISEASES_PATH = "data/diseases_list.txt"

SYSTEM_PROMPT = (
    "You are a biomedical entity resolver. Given a user's plain-English disease "
    "description, identify the single best matching disease entity from the "
    "provided list of CTD disease names. Return JSON only."
)


def load_candidate_diseases(path: str = DEFAULT_DISEASES_PATH) -> List[str]:
    """Load CTD disease names (one per line) from disk."""
    disease_path = Path(path)
    if not disease_path.exists():
        raise FileNotFoundError(f"Disease list not found: {path}")
    return [line.strip() for line in disease_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _default_null_result() -> Dict[str, Any]:
    return {
        "ctd_entity": None,
        "confidence": "low",
        "display_name": "",
        "clarification": "Could you describe your condition differently?",
    }


def _fallback_map_disease(user_input: str, candidate_diseases: List[str]) -> Dict[str, Any]:
    """Fuzzy fallback when Anthropic API is unavailable."""
    result = _default_null_result()
    query = user_input.strip().lower()
    if not query:
        return result

    def normalize(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()

    # Pass 1: lexical fuzzy match on normalized strings.
    norm_query = normalize(query)
    norm_to_original = {normalize(d): d for d in candidate_diseases}
    norm_candidates = list(norm_to_original.keys())
    norm_match = difflib.get_close_matches(norm_query, norm_candidates, n=1, cutoff=0.72)
    if norm_match:
        ctd_entity = norm_to_original[norm_match[0]]
        return {
            "ctd_entity": ctd_entity,
            "confidence": "medium",
            "display_name": ctd_entity,
            "clarification": None,
        }

    # Pass 2: token overlap for phrase-style queries.
    query_tokens = set(norm_query.split())
    if not query_tokens:
        return result

    best_entity = None
    best_overlap = 0.0
    for disease in candidate_diseases:
        disease_tokens = set(normalize(disease).split())
        if not disease_tokens:
            continue
        overlap = len(query_tokens & disease_tokens) / len(query_tokens | disease_tokens)
        if overlap > best_overlap:
            best_overlap = overlap
            best_entity = disease

    if best_entity is not None and best_overlap >= 0.30:
        return {
            "ctd_entity": best_entity,
            "confidence": "low",
            "display_name": best_entity,
            "clarification": "I matched this with low confidence. Is this the condition you meant?",
        }

    return result


def _extract_text(response: Any) -> str:
    """Extract assistant text from Anthropic response blocks."""
    blocks = getattr(response, "content", [])
    text_parts = [block.text for block in blocks if getattr(block, "type", None) == "text"]
    return "\n".join(text_parts).strip()


def _normalize_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate/normalize model JSON into the expected return schema."""
    result = _default_null_result()

    ctd_entity = data.get("ctd_entity")
    confidence = data.get("confidence")
    display_name = data.get("display_name")
    clarification = data.get("clarification")

    if confidence not in {"high", "medium", "low"}:
        confidence = "low"

    if ctd_entity in (None, "", "null"):
        result["confidence"] = confidence
        # Keep default null-entity clarification if model omitted one.
        if isinstance(clarification, str) and clarification.strip():
            result["clarification"] = clarification.strip()
        return result

    ctd_entity = str(ctd_entity).strip()
    result["ctd_entity"] = ctd_entity
    result["confidence"] = confidence
    result["display_name"] = (
        str(display_name).strip() if isinstance(display_name, str) and display_name.strip() else ctd_entity
    )

    if confidence == "low":
        if isinstance(clarification, str) and clarification.strip():
            result["clarification"] = clarification.strip()
        else:
            result["clarification"] = (
                "I might have matched the wrong condition. Could you clarify your disease name?"
            )
    else:
        result["clarification"] = None

    return result


def map_disease(user_input: str, candidate_diseases: List[str]) -> Dict[str, Any]:
    """
    Map free-text disease input to the best matching CTD disease.

    Returns:
        {
            "ctd_entity": str | None,
            "confidence": "high" | "medium" | "low",
            "display_name": str,
            "clarification": str | None
        }
    """
    if not isinstance(user_input, str) or not user_input.strip():
        return _default_null_result()
    if not candidate_diseases:
        return _default_null_result()

    formatted_candidates = "\n".join(candidate_diseases)
    user_prompt = (
        f"User input: '{user_input.strip()}'\n\n"
        "CTD disease list (select one):\n"
        f"{formatted_candidates}\n\n"
        'Return strictly valid JSON with keys: "ctd_entity", "confidence", '
        '"display_name", "clarification".'
    )

    import os
    from dotenv import load_dotenv
    load_dotenv()

    try:
        if Anthropic is None:
            raise RuntimeError("anthropic package is not installed")
        client = Anthropic()
        response = client.messages.create(
            model="claude-opus-4-6",
            system=SYSTEM_PROMPT,
            temperature=0,
            max_tokens=300,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw_text = _extract_text(response)

        # Accept plain JSON response and tolerate markdown-fenced JSON.
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()

        parsed = json.loads(cleaned)
        normalized = _normalize_response(parsed)

        # Fuzzy-match guard: if model returns a valid entity not exactly in list,
        # try case-insensitive match before giving up.
        if normalized["ctd_entity"] and normalized["ctd_entity"] not in candidate_diseases:
            lower_map = {d.lower(): d for d in candidate_diseases}
            matched = lower_map.get(normalized["ctd_entity"].lower())
            if matched:
                normalized["ctd_entity"] = matched
                normalized["display_name"] = matched
            else:
                # Genuinely not in list — fall back to difflib
                return _fallback_map_disease(user_input=user_input.strip(), candidate_diseases=candidate_diseases)

        return normalized

    except TypeError:
        # Missing API key / client misconfiguration — same as other API failures (SPEC: difflib fallback)
        return _fallback_map_disease(user_input=user_input.strip(), candidate_diseases=candidate_diseases)
    except (OSError, ConnectionError, TimeoutError):
        # Network/timeout errors — safe to fall back to difflib
        return _fallback_map_disease(user_input=user_input.strip(), candidate_diseases=candidate_diseases)
    except (json.JSONDecodeError, KeyError):
        # Malformed model response — fall back
        return _fallback_map_disease(user_input=user_input.strip(), candidate_diseases=candidate_diseases)


if __name__ == "__main__":
    diseases = load_candidate_diseases(DEFAULT_DISEASES_PATH)
    print(f"Loaded {len(diseases)} candidate diseases from {DEFAULT_DISEASES_PATH}")

    test_inputs = [
        "my heart beats too fast",
        "type 2 diabetes",
        "can't remember things",
        "breast cancer",
        "xyz123",
    ]

    for text in test_inputs:
        result = map_disease(text, diseases)
        print(f"\nInput: {text}")
        print(json.dumps(result, indent=2))
