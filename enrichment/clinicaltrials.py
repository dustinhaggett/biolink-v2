"""
ClinicalTrials.gov v2 API client.

Searches for clinical trials linking a drug (intervention) to a disease (condition).
Free API, no key required.
"""

from __future__ import annotations

import logging
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)

_BASE_URL = "https://clinicaltrials.gov/api/v2/studies"
_STUDY_URL = "https://clinicaltrials.gov/study"


async def find_trials(
    drug: str,
    disease: str,
    session: aiohttp.ClientSession,
    max_results: int = 3,
) -> list[dict[str, Any]]:
    """
    Search ClinicalTrials.gov for trials of a drug for a disease.

    Returns list of dicts with keys: nct_id, title, status, phase, url
    """
    params = {
        "query.cond": disease,
        "query.intr": drug,
        "pageSize": max_results,
        "fields": "NCTId,BriefTitle,OverallStatus,Phase",
    }

    try:
        async with session.get(_BASE_URL, params=params, timeout=aiohttp.ClientTimeout(total=8)) as resp:
            if resp.status != 200:
                logger.warning("ClinicalTrials.gov returned %d for %s/%s", resp.status, drug, disease)
                return []

            data = await resp.json()
            studies = data.get("studies", [])

            trials = []
            for study in studies[:max_results]:
                proto = study.get("protocolSection", {})
                id_mod = proto.get("identificationModule", {})
                status_mod = proto.get("statusModule", {})
                design_mod = proto.get("designModule", {})

                nct_id = id_mod.get("nctId", "")
                if not nct_id:
                    continue

                trials.append({
                    "nct_id": nct_id,
                    "title": id_mod.get("briefTitle", "Untitled"),
                    "status": status_mod.get("overallStatus", "Unknown"),
                    "phase": ", ".join(design_mod.get("phases", [])) if design_mod.get("phases") else None,
                    "url": f"{_STUDY_URL}/{nct_id}",
                })

            return trials

    except Exception as exc:
        logger.warning("ClinicalTrials.gov error for %s/%s: %s", drug, disease, exc)
        return []
