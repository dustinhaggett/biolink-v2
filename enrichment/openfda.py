import aiohttp

_BASE = "https://api.fda.gov/drug/label.json"

"""
    Query OpenFDA drug label API for a drug name.
    Returns: "FDA Approved" | "Not in FDA Database" | "Unknown"
"""
async def fda_status(drug_name: str, session: aiohttp.ClientSession) -> str:
    params = {
        "search": f'openfda.brand_name:"{drug_name}" OR openfda.generic_name:"{drug_name}"',
        "limit": 1,
    }

    try:
        timeout = aiohttp.ClientTimeout(total=5)
        async with session.get(_BASE, params=params, timeout=timeout) as resp:
            if resp.status == 404:
                return "Not in FDA Database"
            resp.raise_for_status()
            data = await resp.json()
            if data.get("results"):
                return "FDA Approved"
            return "Not in FDA Database"

    except aiohttp.ClientResponseError:
        return "Unknown"
    except Exception:
        return "Unknown"