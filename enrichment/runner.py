import asyncio
import aiohttp
from enrichment.pubmed import evidence_count
from enrichment.openfda import fda_status

"""
    Runs PubMed + OpenFDA lookups in parallel for all results.
    Mutates and returns results with pubmed_count and fda_status filled in.
"""
async def enrich_results(results: list[dict], ctd_entity: str) -> list[dict]:
    async with aiohttp.ClientSession() as session:
        tasks = []
        for r in results:
            tasks.append(evidence_count(r["drug"], ctd_entity, session))
            tasks.append(fda_status(r["drug"], session))

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        for i, r in enumerate(results):       # responses is a flat list: [pubmed_0, fda_0, pubmed_1, fda_1, ...]
            pubmed = responses[i * 2]
            fda = responses[i * 2 + 1]

            r["pubmed_count"] = pubmed if isinstance(pubmed, int) else None
            r["fda_status"] = fda if isinstance(fda, str) else "Unknown"

    return results