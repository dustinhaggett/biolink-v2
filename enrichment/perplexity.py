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

Start your response with EXACTLY one of these verdict lines (pick the most accurate):
VERDICT: SUPPORTS — if evidence suggests this drug may help treat this disease
VERDICT: STANDARD-OF-CARE — if this drug is already an established treatment for this disease
VERDICT: CONFLICTS — if evidence suggests this drug would be ineffective or harmful for this disease
VERDICT: INSUFFICIENT — if there is little or no evidence linking this drug to this disease

Then on the next line, write a single-sentence TL;DR starting with "TLDR:" that summarizes the key finding.

Then provide:
1. **Mechanism**: How might {drug} work against {disease}? What biological pathways are involved?
2. **Key studies**: Summarize the most relevant published studies (include authors, year, and journal if possible).
3. **Clinical trials**: Are there any ongoing or completed clinical trials of {drug} for {disease}?
4. **Evidence strength**: How strong is the overall evidence? (preclinical only, small human studies, large RCTs, etc.)

Be concise and factual. If evidence is limited or absent, say so clearly.\
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


def _clean_summary(text: str) -> str:
    """Remove the verdict and TLDR lines from the summary body."""
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip().lstrip("*").strip()
        if stripped.startswith("VERDICT:") or stripped.startswith("TLDR:"):
            continue
        cleaned.append(line)
    # Strip leading blank lines
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
        return {
            "summary": None,
            "verdict": "insufficient",
            "tldr": None,
            "citations": [],
            "error": "PERPLEXITY_API_KEY is not configured",
        }

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
        tldr = _parse_tldr(raw_text)
        summary = _clean_summary(raw_text) if raw_text else None

        return {
            "summary": summary or None,
            "verdict": verdict,
            "tldr": tldr,
            "citations": citations,
            "error": None,
        }

    except requests.exceptions.Timeout:
        logger.warning("Perplexity API timeout for %s / %s", drug, disease)
        return {"summary": None, "verdict": "insufficient", "tldr": None, "citations": [], "error": "Search timed out"}
    except Exception as exc:
        logger.warning("Perplexity API error for %s / %s: %s", drug, disease, exc)
        return {"summary": None, "verdict": "insufficient", "tldr": None, "citations": [], "error": str(exc)}
