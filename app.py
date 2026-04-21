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
    confidence_badge_html,
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
        diseases_list_path=_DISEASES,
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


@st.cache_resource
def load_drugs():
    path = _DRUGS
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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


def run_reverse_inference(query: str) -> dict:
    """Run the reverse inference pipeline (drug -> diseases)."""
    from core.inference import drug_to_diseases
    model = load_model()
    scaler = load_scaler()
    drugs = load_drugs()
    return drug_to_diseases(query, top_n=20, model=model, scaler=scaler, drugs_list=drugs)


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
if "compare_set" not in st.session_state:
    st.session_state.compare_set = set()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}
if "search_mode" not in st.session_state:
    st.session_state.search_mode = "disease"
if "drug_name" not in st.session_state:
    st.session_state.drug_name = ""


# ── Main App ──────────────────────────────────────────────────

def main():
    render_disclaimer()

    # ── Sidebar ──
    with st.sidebar:
        st.markdown(
            '<div style="margin-bottom:1.5rem;">'
            '<span style="font-family:Manrope,sans-serif; font-weight:800; font-size:1.3rem; color:#00606d;">BioLink</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        if st.session_state.app_state != "idle":
            if st.button("New Search", use_container_width=True, type="primary"):
                st.session_state.app_state = "idle"
                st.session_state.results = None
                st.session_state.compare_set = set()
                st.session_state.chat_history = {}
                st.rerun()
        st.markdown("---")

    min_conf = 0.0
    fda_filters = {"FDA Approved", "Not in FDA Database", "Unknown"}
    if st.session_state.results and st.session_state.app_state == "results":
        min_conf, fda_filters = render_sidebar_filters()
        with st.sidebar:
            st.markdown("---")
            filtered_for_export = filter_results(st.session_state.results, min_conf, fda_filters)
            render_export_button(filtered_for_export, st.session_state.disease_name)
            # PDF export
            try:
                from ui.pdf_export import generate_pdf
                pdf_bytes = generate_pdf(filtered_for_export, st.session_state.disease_name)
                st.download_button(
                    label="Export PDF",
                    data=pdf_bytes,
                    file_name=f"biolink_{st.session_state.disease_name.replace(' ', '_')}.pdf",
                    mime="application/pdf",
                )
            except Exception:
                pass  # PDF generation is optional, don't crash the app

    # ── Batch mode in sidebar ──
    if st.session_state.app_state == "idle":
        with st.sidebar:
            with st.expander("Batch Mode"):
                st.markdown("Upload a CSV with a `disease` column to run predictions for multiple diseases at once.")
                uploaded = st.file_uploader("Upload CSV", type=["csv"], key="batch_upload")
                if uploaded and st.button("Run Batch", type="primary"):
                    _run_batch(uploaded)

    # ── Idle State: Search ──
    if st.session_state.app_state in ("idle", "clarifying"):

        if st.session_state.app_state == "idle":
            query = render_search_section(
                all_diseases=load_diseases(),
                all_drugs=load_drugs(),
            )
            if query:
                if st.session_state.search_mode == "drug":
                    _run_reverse_pipeline(query)
                else:
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

    # ── Reverse Results State ──
    if st.session_state.app_state == "reverse_results" and st.session_state.results:
        _render_reverse_results()

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


def _run_reverse_pipeline(query: str):
    """Execute the reverse pipeline (drug -> diseases)."""
    with st.status("Analyzing...", expanded=True) as status:
        st.write("Identifying drug entity...")
        output = run_reverse_inference(query)

        if output.get("clarification"):
            st.session_state.app_state = "clarifying"
            st.session_state.clarification_question = output["clarification"]
            st.session_state.disease_name = output.get("display_name", query)
            status.update(label="Clarification needed", state="complete")
            st.rerun()
            return

        st.session_state.drug_name = output.get("display_name", output["drug_entity"])
        results = output["results"]
        st.write(f"Found {len(results)} potential conditions")

        status.update(label="Analysis complete", state="complete", expanded=False)

    st.session_state.results = results
    st.session_state.app_state = "reverse_results"
    st.rerun()


def _render_results(min_conf: float, fda_filters: set):
    """Render the results dashboard."""
    disease = st.session_state.disease_name

    # Header
    st.markdown(
        f'<h1>Results for: <span class="primary-text">{disease}</span></h1>',
        unsafe_allow_html=True,
    )

    # Apply filters
    filtered = filter_results(st.session_state.results, min_conf, fda_filters)

    if not filtered:
        st.info("No results match your current filters. Try adjusting the confidence threshold or FDA status filters.")
        return

    st.markdown(f"Showing **{len(filtered)}** candidates")

    # Compare mode controls
    compare_set = st.session_state.compare_set
    if len(compare_set) >= 2:
        if st.button(f"Compare {len(compare_set)} selected drugs", type="primary"):
            _render_comparison(filtered, compare_set)

    # Render cards
    for i, result in enumerate(filtered, 1):
        # Compare checkbox
        drug = result["drug"]
        checked = st.checkbox(
            f"Compare {drug}",
            value=drug in compare_set,
            key=f"cmp_{drug}",
            label_visibility="collapsed",
        )
        if checked:
            compare_set.add(drug)
        elif drug in compare_set:
            compare_set.discard(drug)

        render_result_card(result, rank=i, disease_name=disease)


def _render_reverse_results():
    """Render reverse search results (drug -> diseases)."""
    drug = st.session_state.drug_name
    results = st.session_state.results

    st.markdown(
        f'<h1>Conditions for: <span class="primary-text">{drug}</span></h1>',
        unsafe_allow_html=True,
    )

    st.markdown(f"Showing **{len(results)}** potential conditions")

    for i, r in enumerate(results, 1):
        disease = r["disease"]
        tier = r.get("tier", "Speculative")
        proba = r.get("proba", 0.0)
        tier_class = tier.lower()

        st.markdown(
            f'<div class="result-card result-card-{tier_class}">'
            f'<div style="display:flex; justify-content:space-between; align-items:center;">'
            f'<div><span class="drug-name">{i}. {disease}</span> '
            f'{confidence_badge_html(tier, proba)}</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )


def _render_comparison(filtered: list[dict], compare_set: set):
    """Render side-by-side comparison table."""
    import pandas as pd

    selected = [r for r in filtered if r["drug"] in compare_set]
    if not selected:
        return

    rows = []
    for r in selected:
        evidence = r.get("evidence", {}) or {}
        rows.append({
            "Drug": r["drug"],
            "Confidence": f"{r.get('proba', 0):.0%} {r.get('tier', '')}",
            "FDA Status": r.get("fda_status", "Unknown"),
            "Verdict": evidence.get("verdict", "—").replace("-", " ").title(),
            "Evidence Quality": evidence.get("evidence_quality", "—"),
            "PubMed": r.get("pubmed_count", "—"),
            "Trials": len(r.get("clinical_trials", [])),
            "TL;DR": evidence.get("tldr", "—") or "—",
        })

    df = pd.DataFrame(rows)
    st.markdown("### Comparison")
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.markdown("---")


def _run_batch(uploaded_file):
    """Run batch predictions for multiple diseases from a CSV."""
    import csv as csv_mod
    import io

    content = uploaded_file.read().decode("utf-8")
    reader = csv_mod.DictReader(io.StringIO(content))

    diseases = []
    for row in reader:
        disease = row.get("disease") or row.get("Disease") or row.get("DISEASE")
        if disease and disease.strip():
            diseases.append(disease.strip())

    if not diseases:
        st.error("No 'disease' column found in CSV.")
        return

    # Limit to 5
    diseases = diseases[:5]

    all_results = []
    progress = st.progress(0, text="Running batch predictions...")
    for i, disease in enumerate(diseases):
        progress.progress((i + 1) / len(diseases), text=f"Processing {disease}...")
        try:
            output = run_inference(disease)
            if output.get("results"):
                for r in output["results"][:10]:
                    all_results.append({
                        "disease": output.get("display_name", disease),
                        "rank": len([x for x in all_results if x["disease"] == output.get("display_name", disease)]) + 1,
                        "drug": r["drug"],
                        "confidence": f"{r.get('proba', 0):.4f}",
                        "tier": r.get("tier", ""),
                    })
        except Exception as exc:
            st.warning(f"Failed for {disease}: {exc}")

    progress.empty()

    if all_results:
        import pandas as pd
        df = pd.DataFrame(all_results)
        st.success(f"Batch complete: {len(diseases)} diseases, {len(all_results)} predictions")
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Download
        csv_output = df.to_csv(index=False)
        st.download_button(
            "Download Batch Results",
            data=csv_output,
            file_name="biolink_batch_results.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
