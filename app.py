"""
BioLink v2 — Streamlit Application
Drug repurposing discovery tool powered by knowledge graph AI.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="BioLink",
    page_icon="🧬",
    layout="wide",
)

from ui.styles import inject_css
from ui.components import (
    render_disclaimer,
    render_search_section,
    render_clarification,
    render_result_card,
    render_sidebar_filters,
    render_export_button,
    filter_results,
)

inject_css()

# ── Paths ─────────────────────────────────────────────────────

_ROOT = Path(__file__).resolve().parent
_WEIGHTS = _ROOT / "models" / "biolink_v1.pt"
_BIOWORDVEC = _ROOT / "data" / "BioWordVec_PubMed_MIMICIII_d200.vec.bin"
_DRUGS = _ROOT / "data" / "drugs_list.txt"
_DISEASES = _ROOT / "data" / "diseases_list.txt"
_TEMPERATURE = _ROOT / "data" / "temperature.json"


# ── Cached Resources ──────────────────────────────────────────

@st.cache_resource(show_spinner="Loading BioLink model (this may take a minute)...")
def load_model():
    from core.model import BioLinkModel
    return BioLinkModel(
        weights_path=_WEIGHTS,
        biowordvec_path=_BIOWORDVEC,
        drugs_list_path=_DRUGS,
    )


@st.cache_resource
def load_scaler():
    from core.calibration import TemperatureScaler
    if _TEMPERATURE.exists():
        return TemperatureScaler.load(str(_TEMPERATURE))
    return TemperatureScaler(T=1.0)


@st.cache_resource
def load_diseases():
    from core.intent_mapper import load_candidate_diseases
    return load_candidate_diseases(str(_DISEASES))


# ── Pipeline Functions ────────────────────────────────────────

def run_inference(query: str) -> dict:
    """Run the core inference pipeline."""
    from core.inference import disease_to_drugs
    model = load_model()
    scaler = load_scaler()
    diseases = load_diseases()
    return disease_to_drugs(query, top_n=20, model=model, scaler=scaler, diseases_list=diseases)


def run_enrichment(results: list[dict], ctd_entity: str) -> list[dict]:
    """Run async enrichment (PubMed + OpenFDA) synchronously."""
    from enrichment.runner import enrich_results
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(enrich_results(results, ctd_entity))
    finally:
        loop.close()


def run_explanations(results: list[dict], disease: str, top_n: int = 10) -> list[dict]:
    """Generate explanations for the top N results."""
    from explanation.explainer import explain_prediction
    for r in results[:top_n]:
        r["explanation"] = explain_prediction(
            drug=r["drug"],
            disease=disease,
            proba=r.get("proba", 0.0),
            tier=r.get("tier", "Speculative"),
            pubmed_count=r.get("pubmed_count", 0) or 0,
            fda_status=r.get("fda_status", "Unknown") or "Unknown",
        )
    return results


def run_evidence_search(results: list[dict], disease: str, top_n: int = 5) -> list[dict]:
    """Search Perplexity for evidence on the top N drug-disease pairs."""
    from enrichment.perplexity import search_drug_disease
    for r in results[:top_n]:
        r["evidence"] = search_drug_disease(drug=r["drug"], disease=disease)
    return results


# ── Session State Init ────────────────────────────────────────

if "app_state" not in st.session_state:
    st.session_state.app_state = "idle"
if "results" not in st.session_state:
    st.session_state.results = None
if "disease_name" not in st.session_state:
    st.session_state.disease_name = ""
if "ctd_entity" not in st.session_state:
    st.session_state.ctd_entity = ""
if "clarification_question" not in st.session_state:
    st.session_state.clarification_question = ""


# ── Main App ──────────────────────────────────────────────────

def main():
    render_disclaimer()

    # ── Sidebar (only show filters when we have results) ──
    min_conf = 0.0
    fda_filters = {"FDA Approved", "Not in FDA Database", "Unknown"}
    if st.session_state.results:
        min_conf, fda_filters = render_sidebar_filters()
        with st.sidebar:
            st.markdown("---")
            render_export_button(
                filter_results(st.session_state.results, min_conf, fda_filters),
                st.session_state.disease_name,
            )

    # ── Idle State: Search ──
    if st.session_state.app_state in ("idle", "clarifying"):

        if st.session_state.app_state == "idle":
            query = render_search_section(all_diseases=load_diseases())
            if query:
                _run_pipeline(query)

        elif st.session_state.app_state == "clarifying":
            st.markdown(
                f"### Searching for: *{st.session_state.disease_name}*"
            )
            response = render_clarification(st.session_state.clarification_question)
            if response:
                _run_pipeline(response)

    # ── Results State ──
    if st.session_state.app_state == "results" and st.session_state.results:
        _render_results(min_conf, fda_filters)

    # Bottom disclaimer
    st.markdown("---")
    render_disclaimer()


def _run_pipeline(query: str):
    """Execute the full pipeline with status updates."""

    # Step 1: Intent mapping + model scoring
    with st.status("Analyzing...", expanded=True) as status:
        st.write("Identifying disease entity...")
        output = run_inference(query)

        # Check for clarification
        if output.get("clarification"):
            st.session_state.app_state = "clarifying"
            st.session_state.clarification_question = output["clarification"]
            st.session_state.disease_name = output.get("display_name", query)
            status.update(label="Clarification needed", state="complete")
            st.rerun()
            return

        st.session_state.ctd_entity = output["ctd_entity"]
        st.session_state.disease_name = output.get("display_name", output["ctd_entity"])
        results = output["results"]
        st.write(f"Found {len(results)} drug candidates")

        # Step 2: Enrichment
        st.write("Gathering PubMed evidence and FDA status...")
        results = run_enrichment(results, output["ctd_entity"])

        # Step 3: Explanations (top 10)
        st.write("Generating plain-English explanations...")
        results = run_explanations(results, output["ctd_entity"], top_n=10)

        # Step 4: Evidence search (top 5)
        st.write("Searching for published evidence...")
        results = run_evidence_search(results, output["ctd_entity"], top_n=5)

        status.update(label="Analysis complete", state="complete", expanded=False)

    # Store results and switch state
    st.session_state.results = results
    st.session_state.app_state = "results"
    st.rerun()


def _render_results(min_conf: float, fda_filters: set):
    """Render the results dashboard."""
    disease = st.session_state.disease_name

    # Header
    col1, col2 = st.columns([6, 2])
    with col1:
        st.markdown(
            f'<h1>Results for: <span class="primary-text">{disease}</span></h1>',
            unsafe_allow_html=True,
        )
    with col2:
        if st.button("New Search", use_container_width=True):
            st.session_state.app_state = "idle"
            st.session_state.results = None
            st.rerun()

    # Apply filters
    filtered = filter_results(st.session_state.results, min_conf, fda_filters)

    if not filtered:
        st.info("No results match your current filters. Try adjusting the confidence threshold or FDA status filters.")
        return

    st.markdown(f"Showing **{len(filtered)}** candidates")

    # Render cards
    for i, result in enumerate(filtered, 1):
        render_result_card(result, rank=i)


if __name__ == "__main__":
    main()
