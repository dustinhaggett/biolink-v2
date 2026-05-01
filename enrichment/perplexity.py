"""
Perplexity Sonar search for drug-disease evidence.

Queries the Perplexity API to find relevant studies, mechanisms,
clinical trials, and citations linking a drug to a disease.
"""

from __future__ import annotations

import logging
import os
import re
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

_ENDPOINT = "https://api.perplexity.ai/chat/completions"

_SEARCH_PROMPT = """\
I'm researching whether the drug **{drug}** could be repurposed to treat **{disease}**.

Start your response with these structured lines (one per line):

VERDICT: [pick ONE: SUPPORTS | STANDARD-OF-CARE | CONFLICTS | INSUFFICIENT]
HARM_FOR_INDICATION: [pick ONE: HARMFUL | NOT_HARMFUL | UNKNOWN]
TLDR: [one-sentence summary of the key finding]
EVIDENCE_QUALITY: [pick ONE: RCT | Human Study | Preclinical | Case Report | Theoretical]
PATHWAY: {drug} -> [molecular target] -> [pathway/mechanism] -> [disease effect]
INTERACTIONS: [YES or NO — does {drug} have known dangerous interactions with standard treatments for {disease}?]

Important: HARM_FOR_INDICATION asks a clinical-practice question, not a literature-evidence question:
"If a clinician administered {drug} to a patient WHO HAS {disease}, would the patient likely be harmed,
or their condition worsened, based on the drug's known mechanism, pharmacology, or contraindications?"

  - HARMFUL = yes, the patient would likely be harmed. This includes:
      * Drug pharmacologically worsens the disease (e.g., nicotine vasoconstriction worsens migraine).
      * Drug INDUCES the disease in research/clinical use (e.g., scopolamine induces amnesia,
        streptozocin induces diabetes — these are HARMFUL even if no clinical trial tested them
        as a TREATMENT for the disease they cause).
      * Drug is contraindicated for the disease class (e.g., immunosuppressants like cyclosporine
        in active bacterial infection — would worsen Lyme, even if no direct trial exists).
      * Drug interacts dangerously with standard treatment for {disease}.
  - NOT_HARMFUL = no — the drug would not specifically harm a patient with {disease}, based on
      mechanism and known evidence. (General drug toxicity at high doses is NOT_HARMFUL if there's
      no indication-specific concern.)
  - UNKNOWN = insufficient data to reason about patient outcomes for this combination.

If the drug's KNOWN MECHANISM would predictably worsen {disease}, classify HARMFUL even without a
direct clinical trial confirming harm.

Then provide:
1. **Mechanism**: How might {drug} work against {disease}? What biological pathways are involved?
2. **Key studies**: Summarize the most relevant published studies (include authors, year, and journal if possible).
3. **Clinical trials**: Are there any ongoing or completed clinical trials of {drug} for {disease}?
4. **Evidence strength**: How strong is the overall evidence? (preclinical only, small human studies, large RCTs, etc.)
5. **Interactions**: If YES above, briefly list the specific interactions with standard-of-care drugs for {disease}.

Be concise and factual. If evidence is limited or absent, say so clearly.\
"""

_FOLLOWUP_PROMPT = """\
Context: The drug **{drug}** was predicted as a repurposing candidate for **{disease}**.

Prior evidence summary:
{prior_evidence}

User's follow-up question: {question}

Answer the question concisely and factually, grounded in published biomedical evidence. \
If the answer is unknown or uncertain, say so. Include a brief medical disclaimer.\
"""

# Domains considered credible for biomedical evidence
_ALLOWED_DOMAINS = {
    "ncbi.nlm.nih.gov",
    "pubmed.ncbi.nlm.nih.gov",
    "pmc.ncbi.nlm.nih.gov",
    "clinicaltrials.gov",
    "cdc.gov",
    "fda.gov",
    "who.int",
    "nih.gov",
    "drugbank.com",
    "go.drugbank.com",
    "doi.org",
    "nature.com",
    "nejm.org",
    "thelancet.com",
    "bmj.com",
    "jamanetwork.com",
    "wiley.com",
    "onlinelibrary.wiley.com",
    "springer.com",
    "link.springer.com",
    "sciencedirect.com",
    "cell.com",
    "frontiersin.org",
    "mdpi.com",
    "journals.asm.org",
    "academic.oup.com",
    "hopkinsmedicine.org",
    "hopkinslyme.org",
    "hopkinsarthritis.org",
    "mayoclinic.org",
    "medscape.com",
    "uptodate.com",
    "cochranelibrary.com",
    "europepmc.org",
}


def _filter_citations(citations: list[str]) -> list[str]:
    """Keep only citations from credible biomedical sources."""
    filtered = []
    for url in citations:
        try:
            domain = urlparse(url).netloc.lower()
            # Strip www. prefix
            if domain.startswith("www."):
                domain = domain[4:]
            # Check if domain or any parent domain is allowed
            if any(domain == d or domain.endswith("." + d) for d in _ALLOWED_DOMAINS):
                filtered.append(url)
        except Exception:
            continue
    return filtered


def _parse_verdict(text: str) -> str:
    """Extract the verdict from the response text."""
    match = re.search(r"VERDICT:\s*(SUPPORTS|STANDARD-OF-CARE|CONFLICTS|INSUFFICIENT)", text)
    if match:
        return match.group(1).lower()
    return "insufficient"


def _parse_tldr(text: str) -> str | None:
    """Extract the TL;DR line from the response text."""
    match = re.search(r"TLDR:\s*(.+)", text)
    if match:
        # Strip markdown bold markers and trailing punctuation artifacts
        tldr = match.group(1).strip().strip("*").strip()
        return tldr
    return None


_QUALITY_NORMALIZE = {
    "rct": "RCT",
    "human study": "Human Study",
    "preclinical": "Preclinical",
    "case report": "Case Report",
    "theoretical": "Theoretical",
}


def _parse_evidence_quality(text: str) -> str:
    """Extract evidence quality classification from the response."""
    match = re.search(r"EVIDENCE_QUALITY:\s*(RCT|Human Study|Preclinical|Case Report|Theoretical)", text, re.IGNORECASE)
    if match:
        return _QUALITY_NORMALIZE.get(match.group(1).lower(), "Unknown")
    return "Unknown"


def _parse_pathway(text: str) -> str | None:
    """Extract the pathway chain from the response."""
    match = re.search(r"PATHWAY:\s*(.+)", text)
    if match:
        pathway = match.group(1).strip().strip("*").strip()
        return pathway if pathway else None
    return None


def _parse_interactions(text: str) -> bool:
    """Check if the response flags drug interactions."""
    match = re.search(r"INTERACTIONS:\s*(YES|NO)", text, re.IGNORECASE)
    if match:
        return match.group(1).upper() == "YES"
    return False


_HARM_NORMALIZE = {
    "harmful": "harmful",
    "not_harmful": "not_harmful",
    "not-harmful": "not_harmful",
    "not harmful": "not_harmful",
    "unknown": "unknown",
}


def _parse_harm_for_indication(text: str) -> str:
    """Extract indication-specific harm flag.

    Used by core.reranking to demote candidates known to harm THIS disease.
    Defaults to 'unknown' (no demotion) on missing/unparseable values to
    preserve the discovery-vs-harm principle: never demote on absence of
    evidence.
    """
    match = re.search(
        r"HARM_FOR_INDICATION:\s*(HARMFUL|NOT_HARMFUL|NOT-HARMFUL|NOT HARMFUL|UNKNOWN)",
        text,
        re.IGNORECASE,
    )
    if match:
        return _HARM_NORMALIZE.get(match.group(1).lower(), "unknown")
    return "unknown"


_STRUCTURED_PREFIXES = (
    "VERDICT:", "HARM_FOR_INDICATION:", "TLDR:", "EVIDENCE_QUALITY:", "PATHWAY:", "INTERACTIONS:",
)


def _clean_summary(text: str) -> str:
    """Remove structured header lines from the summary body."""
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip().lstrip("*").strip()
        if any(stripped.startswith(p) for p in _STRUCTURED_PREFIXES):
            continue
        cleaned.append(line)
    result = "\n".join(cleaned).strip()
    return result


def _get_api_key() -> str | None:
    """Retrieve the Perplexity API key from st.secrets or environment."""
    try:
        import streamlit as st
        key = st.secrets.get("PERPLEXITY_API_KEY")
        if key:
            return key
    except Exception:
        pass
    return os.environ.get("PERPLEXITY_API_KEY")


def search_drug_disease(drug: str, disease: str) -> dict:
    """
    Search Perplexity for evidence linking a drug to a disease.

    Returns:
        dict with keys:
            - "summary": str — the detailed evidence breakdown
            - "verdict": str — one of: supports, standard-of-care, conflicts, insufficient
            - "tldr": str | None — one-sentence summary
            - "citations": list[str] — filtered source URLs (scholarly only)
            - "error": str | None — error message if the search failed
    """
    api_key = _get_api_key()
    if not api_key:
        return _empty_evidence("PERPLEXITY_API_KEY is not configured")

    prompt = _SEARCH_PROMPT.format(drug=drug, disease=disease)

    try:
        resp = requests.post(
            _ENDPOINT,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "sonar",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 800,
                "temperature": 0.3,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        raw_text = ""
        if data.get("choices"):
            raw_text = data["choices"][0]["message"]["content"]

        citations = _filter_citations(data.get("citations", []))
        verdict = _parse_verdict(raw_text)
        harm_for_indication = _parse_harm_for_indication(raw_text)
        tldr = _parse_tldr(raw_text)
        evidence_quality = _parse_evidence_quality(raw_text)
        pathway = _parse_pathway(raw_text)
        has_interactions = _parse_interactions(raw_text)
        summary = _clean_summary(raw_text) if raw_text else None

        return {
            "summary": summary or None,
            "verdict": verdict,
            "harm_for_indication": harm_for_indication,
            "tldr": tldr,
            "evidence_quality": evidence_quality,
            "pathway": pathway,
            "has_interactions": has_interactions,
            "citations": citations,
            "error": None,
        }

    except requests.exceptions.Timeout:
        logger.warning("Perplexity API timeout for %s / %s", drug, disease)
        return _empty_evidence("Search timed out")
    except Exception as exc:
        logger.warning("Perplexity API error for %s / %s: %s", drug, disease, exc)
        return _empty_evidence(str(exc))


def _empty_evidence(error: str | None = None) -> dict:
    """Return an empty evidence dict (insufficient + unknown harm = preserve)."""
    return {
        "summary": None, "verdict": "insufficient",
        "harm_for_indication": "unknown",
        "tldr": None, "evidence_quality": "Unknown", "pathway": None,
        "has_interactions": False, "citations": [], "error": error,
    }


def ask_followup(drug: str, disease: str, question: str, prior_evidence: str) -> dict:
    """
    Ask a follow-up question about a drug-disease pair, grounded in prior evidence.

    Returns dict with "answer" and "citations" keys.
    """
    api_key = _get_api_key()
    if not api_key:
        return {"answer": None, "citations": [], "error": "PERPLEXITY_API_KEY is not configured"}

    prompt = _FOLLOWUP_PROMPT.format(
        drug=drug, disease=disease, question=question,
        prior_evidence=prior_evidence[:2000],  # Truncate to stay within limits
    )

    try:
        resp = requests.post(
            _ENDPOINT,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "sonar",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 500,
                "temperature": 0.3,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        answer = ""
        if data.get("choices"):
            answer = data["choices"][0]["message"]["content"]

        citations = _filter_citations(data.get("citations", []))
        return {"answer": answer or None, "citations": citations, "error": None}

    except Exception as exc:
        logger.warning("Perplexity followup error: %s", exc)
        return {"answer": None, "citations": [], "error": str(exc)}
