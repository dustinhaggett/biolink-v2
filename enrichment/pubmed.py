import aiohttp

_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

"""
    Query PubMed E-utilities for articles co-mentioning drug and disease.
    Returns count (int), or 0 on timeout/error.
"""
async def evidence_count(drug: str, disease: str, session: aiohttp.ClientSession) -> int:
    params = {
        "db": "pubmed",
        "term": f"{drug}[Title/Abstract] AND {disease}[Title/Abstract]",
        "retmode": "json",
        "retmax": 0,  # we only need the count
    }

    try:
        timeout = aiohttp.ClientTimeout(total=5)
        async with session.get(
            f"{_BASE}/esearch.fcgi",
            params=params,
            timeout=timeout
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return int(data["esearchresult"]["count"])

    except Exception:
        return 0