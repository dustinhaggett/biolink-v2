"""End-to-end disease → ranked drugs pipeline for BioLink v2."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
# Allow `python core/inference.py` (script) as well as `python -m core.inference`.
if __package__ is None and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.calibration import TemperatureScaler, confidence_tier
from core.intent_mapper import load_candidate_diseases, map_disease
from core.model import BioLinkModel

_DEFAULT_WEIGHTS = _REPO_ROOT / "models" / "biolink_v1.pt"
_DEFAULT_BIOWORDVEC = _REPO_ROOT / "data" / "BioWordVec_PubMed_MIMICIII_d200.vec.bin"
_DEFAULT_DRUGS = _REPO_ROOT / "data" / "drugs_list.txt"
_DEFAULT_DISEASES = _REPO_ROOT / "data" / "diseases_list.txt"
_DEFAULT_TEMPERATURE = _REPO_ROOT / "data" / "temperature.json"


def _default_model() -> BioLinkModel:
    return BioLinkModel(
        weights_path=_DEFAULT_WEIGHTS,
        biowordvec_path=_DEFAULT_BIOWORDVEC,
        drugs_list_path=_DEFAULT_DRUGS,
    )


def _default_scaler() -> TemperatureScaler:
    if _DEFAULT_TEMPERATURE.exists():
        return TemperatureScaler.load(str(_DEFAULT_TEMPERATURE))
    return TemperatureScaler(T=1.0)


def disease_to_drugs(
    user_input: str,
    top_n: int = 20,
    model: Optional[BioLinkModel] = None,
    scaler: Optional[TemperatureScaler] = None,
    diseases_list: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Map free-text disease input to top-N drug candidates with calibrated probabilities.

    Returns:
        On success:
            {
                "query": str,
                "ctd_entity": str,
                "display_name": str,
                "clarification": None,
                "results": [
                    {
                        "drug": str,
                        "logit": float,
                        "proba": float,
                        "tier": str,
                        "pubmed_count": None,
                        "fda_status": None,
                        "explanation": None,
                    },
                    ...
                ],
            }
        On null entity or low-confidence match (block until clarified):
            {
                "query": str,
                "ctd_entity": None | str,
                "display_name": str,
                "clarification": str,
                "results": [],
            }
    """
    if model is None:
        model = _default_model()
    if scaler is None:
        scaler = _default_scaler()
    if diseases_list is None:
        diseases_list = load_candidate_diseases(str(_DEFAULT_DISEASES))

    mapped = map_disease(user_input, diseases_list)
    ctd_entity = mapped.get("ctd_entity")
    confidence = mapped.get("confidence", "low")
    display_name = mapped.get("display_name") or ""
    clarification = mapped.get("clarification")

    # 1–2: null entity → early return with clarification
    if ctd_entity is None:
        return {
            "query": user_input,
            "ctd_entity": None,
            "display_name": display_name,
            "clarification": clarification
            if isinstance(clarification, str) and clarification.strip()
            else "Could you describe your condition differently?",
            "results": [],
        }

    # Low confidence with no valid entity: surface clarification
    if confidence == "low" and clarification:
        # If the entity is valid (in the diseases list), proceed anyway
        if ctd_entity and diseases_list and str(ctd_entity) in diseases_list:
            pass  # Valid entity — skip clarification, proceed to scoring
        else:
            msg = (
                clarification.strip()
                if isinstance(clarification, str) and clarification.strip()
                else "I might have matched the wrong condition. Could you confirm or rephrase?"
            )
            return {
                "query": user_input,
                "ctd_entity": str(ctd_entity),
                "display_name": display_name or str(ctd_entity),
                "clarification": msg,
                "results": [],
            }

    # 3–7: encode disease, score all drugs, calibrate, tier
    disease_vec = model.encode_disease(str(ctd_entity))
    scored = model.score_all_drugs(disease_vec)
    top = scored[: max(0, int(top_n))]

    results: List[Dict[str, Any]] = []
    for drug_name, logit in top:
        proba = scaler.calibrated_proba(float(logit))
        results.append(
            {
                "drug": drug_name,
                "logit": float(logit),
                "proba": float(proba),
                "tier": confidence_tier(float(proba)),
                "pubmed_count": None,
                "fda_status": None,
                "explanation": None,
            }
        )

    return {
        "query": user_input,
        "ctd_entity": str(ctd_entity),
        "display_name": display_name or str(ctd_entity),
        "clarification": None,
        "results": results,
    }


def drug_to_diseases(
    drug_input: str,
    top_n: int = 20,
    model: Optional[BioLinkModel] = None,
    scaler: Optional[TemperatureScaler] = None,
    drugs_list: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Reverse search: given a drug name, find the top diseases it may treat.

    Returns dict with keys: query, drug_entity, display_name, clarification, results
    """
    if model is None:
        model = _default_model()
    if scaler is None:
        scaler = _default_scaler()
    if drugs_list is None:
        drugs_list = model.drug_names

    # Fuzzy match drug name against known drugs
    from core.intent_mapper import map_drug
    mapped = map_drug(drug_input, drugs_list)
    drug_entity = mapped.get("drug_entity")
    display_name = mapped.get("display_name") or ""
    clarification = mapped.get("clarification")

    if drug_entity is None:
        return {
            "query": drug_input,
            "drug_entity": None,
            "display_name": display_name,
            "clarification": clarification or "Could you enter a more specific drug name?",
            "results": [],
        }

    # Encode drug and score all diseases
    drug_vec = model.encode_drug(str(drug_entity))
    scored = model.score_all_diseases(drug_vec)
    top = scored[: max(0, int(top_n))]

    results: List[Dict[str, Any]] = []
    for disease_name, logit in top:
        proba = scaler.calibrated_proba(float(logit))
        results.append(
            {
                "disease": disease_name,
                "logit": float(logit),
                "proba": float(proba),
                "tier": confidence_tier(float(proba)),
            }
        )

    return {
        "query": drug_input,
        "drug_entity": str(drug_entity),
        "display_name": display_name or str(drug_entity),
        "clarification": None,
        "results": results,
    }


if __name__ == "__main__":
    # Smoke test: high-confidence path uses exact CTD name when API works;
    # otherwise intent mapper may return low confidence and empty results.
    out = disease_to_drugs("Hypertension", top_n=5)
    print(out["query"])
    print("ctd_entity:", out.get("ctd_entity"))
    print("clarification:", out.get("clarification"))
    for r in out.get("results", []):
        print(f"  {r['drug']}: logit={r['logit']:.4f} proba={r['proba']:.4f} tier={r['tier']}")
