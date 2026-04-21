"""
Reusable Streamlit UI components for BioLink v2.
Renders result cards, badges, search box, disclaimers, etc.
"""

from __future__ import annotations

import csv
import io

import streamlit as st


# ── Disclaimer ────────────────────────────────────────────────

def render_disclaimer():
    st.markdown(
        '<div class="disclaimer-banner">'
        "<strong>Not medical advice.</strong> BioLink is a hypothesis-generation "
        "tool for research purposes only. Always consult a qualified physician "
        "before considering any treatment."
        "</div>",
        unsafe_allow_html=True,
    )


# ── Search ────────────────────────────────────────────────────


def render_search_section(all_diseases: list[str] | None = None):
    """Render the hero search section. Returns the submitted query or None."""
    st.markdown(
        '<div class="search-hero">'
        "<h1>Discover potential drug repurposing candidates "
        'using <span class="primary-text">knowledge graph AI</span></h1>'
        '<p class="search-subtitle">Enter a disease to find ranked drug candidates '
        "with calibrated confidence scores, FDA status, and evidence.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([5, 1])
    with col1:
        query = st.text_input(
            "Disease search",
            placeholder="Enter a disease (e.g., Alzheimer's Disease, Type 2 Diabetes)",
            label_visibility="collapsed",
            key="search_input",
        )
    with col2:
        search_clicked = st.button("Search", type="primary", use_container_width=True)

    # Dropdown to browse all diseases
    if all_diseases:
        def _on_dropdown_change():
            v = st.session_state.get("disease_dropdown", "")
            if v:
                st.session_state.submit_query = v
                # Reset dropdown so selecting the same disease again works
                st.session_state.disease_dropdown = ""

        st.markdown("**Or select a disease from dropdown**")
        st.selectbox(
            "Or select a disease from dropdown",
            options=[""] + all_diseases,
            index=0,
            key="disease_dropdown",
            label_visibility="collapsed",
            on_change=_on_dropdown_change,
            help="Select a disease from the full list of 2,500+ supported conditions",
        )

    if search_clicked and query.strip():
        return query.strip()
    if st.session_state.get("submit_query"):
        q = st.session_state.pop("submit_query")
        return q
    return None


# ── Clarification ─────────────────────────────────────────────

def render_clarification(question: str) -> str | None:
    """Show clarification prompt. Returns user's response or None."""
    st.markdown("### Clarification Required")
    st.info(question)
    col1, col2 = st.columns([5, 1])
    with col1:
        response = st.text_input(
            "Your clarification",
            placeholder="Type a more specific disease name...",
            label_visibility="collapsed",
            key="clarification_input",
        )
    with col2:
        if st.button("Confirm", type="primary", use_container_width=True):
            if response.strip():
                return response.strip()
    return None


# ── Confidence Badge ──────────────────────────────────────────

def confidence_badge_html(tier: str, proba: float) -> str:
    tier_lower = tier.lower()
    pct = f"{proba:.0%}"
    return f'<span class="badge badge-{tier_lower}">{pct} {tier}</span>'


_VERDICT_CONFIG = {
    "supports": ("Evidence Supports", "#1b7a3d", "#e6f4ea"),
    "standard-of-care": ("Standard of Care", "#1a56db", "#e1effe"),
    "conflicts": ("Evidence Conflicts", "#c4320a", "#fef2f2"),
    "insufficient": ("Insufficient Evidence", "#6b7280", "#f3f4f6"),
}


def _verdict_badge_html(verdict: str) -> str:
    label, color, bg = _VERDICT_CONFIG.get(verdict, _VERDICT_CONFIG["insufficient"])
    return (
        f'<span style="display:inline-block; padding:0.25rem 0.75rem; border-radius:999px; '
        f'font-size:0.8rem; font-weight:600; color:{color}; background:{bg}; '
        f'border:1px solid {color}22;">{label}</span>'
    )


def fda_badge_html(fda_status: str | None) -> str:
    if fda_status == "FDA Approved":
        return '<span class="fda-badge fda-approved">FDA Approved</span>'
    elif fda_status == "Not in FDA Database":
        return '<span class="fda-badge fda-not-found">Not in FDA DB</span>'
    else:
        return '<span class="fda-badge fda-unknown">Unknown</span>'


# ── Result Card ───────────────────────────────────────────────

def render_result_card(result: dict, rank: int):
    """Render a single drug result card."""
    tier = result.get("tier", "Speculative")
    tier_class = tier.lower()
    drug = result["drug"]
    proba = result.get("proba", 0.0)
    fda = result.get("fda_status")
    pubmed = result.get("pubmed_count")
    explanation = result.get("explanation")

    # Card header
    st.markdown(
        f'<div class="result-card result-card-{tier_class}">'
        f'<div style="display:flex; justify-content:space-between; align-items:center;">'
        f'<div><span class="drug-name">{rank}. {drug}</span> '
        f"{confidence_badge_html(tier, proba)} "
        f"{fda_badge_html(fda)}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Stats row
    pubmed_text = f"{pubmed} publications" if pubmed is not None else "Evidence data unavailable"
    st.markdown(
        f'<div class="stat-row">'
        f'<span class="stat-item">PubMed: <span class="stat-value">{pubmed_text}</span></span>'
        f"</div>",
        unsafe_allow_html=True,
    )

    # Evidence from Perplexity search
    evidence = result.get("evidence")
    if evidence and evidence.get("summary"):
        # Verdict badge
        verdict = evidence.get("verdict", "insufficient")
        verdict_html = _verdict_badge_html(verdict)
        tldr = evidence.get("tldr")

        # Show verdict + TL;DR inline
        tldr_html = f'<span style="margin-left:0.5rem; font-size:0.9rem; color:#3e494a;">{tldr}</span>' if tldr else ""
        st.markdown(
            f'{verdict_html}{tldr_html}',
            unsafe_allow_html=True,
        )

        with st.expander("View detailed evidence"):
            st.markdown(evidence["summary"])
            citations = evidence.get("citations", [])
            if citations:
                st.markdown("**Sources:**")
                for url in citations:
                    # Show domain name instead of raw URL
                    from urllib.parse import urlparse
                    try:
                        domain = urlparse(url).netloc
                        if domain.startswith("www."):
                            domain = domain[4:]
                        # Trim path for display
                        path = urlparse(url).path.rstrip("/")
                        label = domain + (path if len(path) < 40 else path[:37] + "...")
                    except Exception:
                        label = url
                    st.markdown(f"- [{label}]({url})")
    elif explanation:
        # Fallback to Claude explanation if no evidence search
        with st.expander("View explanation"):
            st.markdown(
                f'<div class="explanation-text">{explanation}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)


# ── Sidebar Filters ───────────────────────────────────────────

def render_sidebar_filters():
    """Render sidebar filters. Returns (min_confidence, fda_filters)."""
    with st.sidebar:
        st.markdown(
            '<div style="margin-bottom:1.5rem;">'
            '<span style="font-family:Manrope,sans-serif; font-weight:800; font-size:1.3rem; color:#00606d;">BioLink</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.markdown("### Filters")

        st.markdown("**Confidence Threshold**")
        min_conf = st.slider(
            "Minimum confidence",
            min_value=0,
            max_value=100,
            value=0,
            format="%d%%",
            label_visibility="collapsed",
        )

        st.markdown("**FDA Status**")
        show_approved = st.checkbox("FDA Approved", value=True)
        show_not_found = st.checkbox("Not in FDA Database", value=True)
        show_unknown = st.checkbox("Unknown", value=True)

        fda_filters = set()
        if show_approved:
            fda_filters.add("FDA Approved")
        if show_not_found:
            fda_filters.add("Not in FDA Database")
        if show_unknown:
            fda_filters.add("Unknown")

        return min_conf / 100.0, fda_filters


def filter_results(results: list[dict], min_conf: float, fda_filters: set) -> list[dict]:
    """Apply confidence and FDA filters to results."""
    filtered = []
    for r in results:
        if r.get("proba", 0) < min_conf:
            continue
        fda = r.get("fda_status", "Unknown")
        if fda not in fda_filters:
            continue
        filtered.append(r)
    return filtered


# ── CSV Export ────────────────────────────────────────────────

def render_export_button(results: list[dict], disease_name: str):
    """Render a CSV download button for the results."""
    if not results:
        return

    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=["rank", "drug", "proba", "tier", "fda_status", "pubmed_count"],
    )
    writer.writeheader()
    for i, r in enumerate(results, 1):
        writer.writerow({
            "rank": i,
            "drug": r["drug"],
            "proba": f"{r.get('proba', 0):.4f}",
            "tier": r.get("tier", ""),
            "fda_status": r.get("fda_status", ""),
            "pubmed_count": r.get("pubmed_count", ""),
        })

    st.download_button(
        label="Export CSV",
        data=output.getvalue(),
        file_name=f"biolink_{disease_name.replace(' ', '_')}.csv",
        mime="text/csv",
    )
