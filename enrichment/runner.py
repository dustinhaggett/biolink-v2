import asyncio
import aiohttp
from enrichment.pubmed import evidence_count
from enrichment.openfda import fda_status
from enrichment.clinicaltrials import find_trials

"""
    Runs PubMed + OpenFDA + ClinicalTrials.gov lookups in parallel for all results.
    Mutates and returns results with pubmed_count, fda_status, and clinical_trials filled in.
"""
async def enrich_results(results: list[dict], ctd_entity: str) -> list[dict]:
    async with aiohttp.ClientSession() as session:
        tasks = []
        for r in results:
            tasks.append(evidence_count(r["drug"], ctd_entity, session))
            tasks.append(fda_status(r["drug"], session))
            tasks.append(find_trials(r["drug"], ctd_entity, session, max_results=3))

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        for i, r in enumerate(results):
            pubmed = responses[i * 3]
            fda = responses[i * 3 + 1]
            trials = responses[i * 3 + 2]

            r["pubmed_count"] = pubmed if isinstance(pubmed, int) else None
            r["fda_status"] = fda if isinstance(fda, str) else "Unknown"
            r["clinical_trials"] = trials if isinstance(trials, list) else []

    return results
