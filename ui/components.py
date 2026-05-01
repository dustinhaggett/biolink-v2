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


def render_search_section(
    all_diseases: list[str] | None = None,
    all_drugs: list[str] | None = None,
):
    """Render the hero search section. Returns the submitted query or None."""
    st.markdown(
        '<div class="search-hero">'
        "<h1>Discover potential drug repurposing candidates "
        'using <span class="primary-text">knowledge graph AI</span></h1>'
        '<p class="search-subtitle">Search by disease to find drug candidates, '
        "or by drug to find conditions it may treat.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Search mode toggle
    mode = st.radio(
        "Search mode",
        options=["disease", "drug"],
        format_func=lambda x: "Search by Disease" if x == "disease" else "Search by Drug",
        horizontal=True,
        key="search_mode",
        label_visibility="collapsed",
    )

    if mode == "disease":
        placeholder = "Enter a disease (e.g., Alzheimer's Disease, Type 2 Diabetes)"
    else:
        placeholder = "Enter a drug (e.g., Metformin, Doxycycline, Aspirin)"

    col1, col2 = st.columns([5, 1])
    with col1:
        query = st.text_input(
            "Search",
            placeholder=placeholder,
            label_visibility="collapsed",
            key="search_input",
        )
    with col2:
        search_clicked = st.button("Search", type="primary", use_container_width=True)

    # Dropdown to browse
    if mode == "disease" and all_diseases:
        def _on_dropdown_change():
            v = st.session_state.get("disease_dropdown", "")
            if v:
                st.session_state.submit_query = v
                st.session_state.disease_dropdown = ""

        st.markdown("**Or select from dropdown**")
        st.selectbox(
            "Select a disease",
            options=[""] + all_diseases,
            index=0,
            key="disease_dropdown",
            label_visibility="collapsed",
            on_change=_on_dropdown_change,
            help="Browse 2,500+ supported conditions",
        )
    elif mode == "drug" and all_drugs:
        def _on_drug_dropdown_change():
            v = st.session_state.get("drug_dropdown", "")
            if v:
                st.session_state.submit_query = v
                st.session_state.drug_dropdown = ""

        st.markdown("**Or select from dropdown**")
        st.selectbox(
            "Select a drug",
            options=[""] + all_drugs,
            index=0,
            key="drug_dropdown",
            label_visibility="collapsed",
            on_change=_on_drug_dropdown_change,
            help="Browse 7,000+ drugs in the database",
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


# ── Evidence Quality Badge ────────────────────────────────────

_QUALITY_CONFIG = {
    "RCT": ("RCT", "#1b7a3d", "#e6f4ea"),
    "Rct": ("RCT", "#1b7a3d", "#e6f4ea"),
    "Human Study": ("Human Study", "#1a56db", "#e1effe"),
    "Preclinical": ("Preclinical", "#b45309", "#fef3c7"),
    "Case Report": ("Case Report", "#6b7280", "#f3f4f6"),
    "Theoretical": ("Theoretical", "#9ca3af", "#f9fafb"),
}


def _evidence_quality_badge_html(quality: str) -> str:
    label, color, bg = _QUALITY_CONFIG.get(quality, ("Unknown", "#9ca3af", "#f9fafb"))
    return (
        f'<span style="display:inline-block; padding:0.2rem 0.6rem; border-radius:999px; '
        f'font-size:0.7rem; font-weight:600; color:{color}; background:{bg}; '
        f'border:1px solid {color}22; margin-left:0.4rem;">{label}</span>'
    )


# ── Citation Formatter ───────────────────────────────────────

def _format_citation_label(url: str) -> str:
    from urllib.parse import urlparse
    try:
        domain = urlparse(url).netloc
        if domain.startswith("www."):
            domain = domain[4:]
        path = urlparse(url).path.rstrip("/")
        return domain + (path if len(path) < 40 else path[:37] + "...")
    except Exception:
        return url


# ── Tier Summary Header ───────────────────────────────────────

def render_tier_summary(
    tier_counts: dict | None,
    total_candidates: int | None,
    showing_count: int,
) -> None:
    """Render an honest summary of the candidate distribution.

    Shows: "12 Strong · 87 Moderate · 540 Speculative — showing top 20 of 7,164 candidates"

    This makes the underlying distribution visible. Some diseases have only a few
    Strong candidates (top 20 covers most signal). Others — like diseases with
    rich therapeutic literature OR cluster-mismatch failures — have hundreds of
    Strong candidates that the top-20 cutoff hides. See "Candidate-count finding"
    in docs/POST_PRESENTATION_TODO.md.

    Backwards compatible: silently no-ops if inference didn't return tier_counts
    (e.g., older cached results).
    """
    if not tier_counts or total_candidates is None:
        return

    strong = int(tier_counts.get("Strong", 0))
    moderate = int(tier_counts.get("Moderate", 0))
    speculative = int(tier_counts.get("Speculative", 0))

    pill = (
        '<span style="display:inline-block; padding:0.15rem 0.55rem; border-radius:999px; '
        'font-size:0.75rem; font-weight:600; color:{color}; background:{bg}; '
        'border:1px solid {color}22; margin-right:0.4rem;">{count:,} {label}</span>'
    )

    # Pluralization handled inline (Streamlit markdown is fine with this)
    pills_html = (
        pill.format(color="#1b7a3d", bg="#e6f4ea", count=strong,      label="Strong")
        + pill.format(color="#b45309", bg="#fef3c7", count=moderate,    label="Moderate")
        + pill.format(color="#6b7280", bg="#f3f4f6", count=speculative, label="Speculative")
    )

    suffix = f"showing top {showing_count} of {total_candidates:,} candidates"

    st.markdown(
        f'<div style="margin: 0.5rem 0 1rem; padding: 0.5rem 0.75rem; '
        f'background: #fafbfc; border-radius: 0.5rem; border: 1px solid #e5e7eb;">'
        f'<div style="margin-bottom: 0.35rem;">{pills_html}</div>'
        f'<div style="font-size: 0.78rem; color: #6b7280;">{suffix}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── Result Card ───────────────────────────────────────────────

def render_result_card(result: dict, rank: int, disease_name: str = ""):
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

    # Clinical trials
    trials = result.get("clinical_trials", [])
    if trials:
        _render_clinical_trials(trials)

    # Evidence from Perplexity search
    evidence = result.get("evidence")
    if evidence and evidence.get("summary"):
        # Verdict + evidence quality + TL;DR
        verdict = evidence.get("verdict", "insufficient")
        verdict_html = _verdict_badge_html(verdict)
        quality = evidence.get("evidence_quality", "Unknown")
        quality_html = _evidence_quality_badge_html(quality) if quality != "Unknown" else ""
        tldr = evidence.get("tldr")

        # Interaction warning badge
        interaction_html = ""
        if evidence.get("has_interactions"):
            interaction_html = (
                '<span style="display:inline-block; padding:0.2rem 0.6rem; border-radius:999px; '
                'font-size:0.7rem; font-weight:600; color:#c4320a; background:#fef2f2; '
                'border:1px solid #c4320a22; margin-left:0.4rem;">Drug Interactions</span>'
            )

        tldr_html = f'<div style="margin-top:0.4rem; font-size:0.9rem; color:#3e494a;">{tldr}</div>' if tldr else ""
        st.markdown(
            f'{verdict_html}{quality_html}{interaction_html}{tldr_html}',
            unsafe_allow_html=True,
        )

        # Pathway chain
        pathway = evidence.get("pathway")
        if pathway:
            st.markdown(
                f'<div style="margin:0.5rem 0; padding:0.5rem 0.75rem; background:#f0f7f8; '
                f'border-radius:0.5rem; font-size:0.85rem; color:#00606d; font-family:monospace;">'
                f'{pathway}</div>',
                unsafe_allow_html=True,
            )

        with st.expander("View detailed evidence"):
            st.markdown(evidence["summary"])
            citations = evidence.get("citations", [])
            if citations:
                st.markdown("**Sources:**")
                for url in citations:
                    label = _format_citation_label(url)
                    st.markdown(f"- [{label}]({url})")
    elif explanation:
        with st.expander("View explanation"):
            st.markdown(
                f'<div class="explanation-text">{explanation}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)


# ── Clinical Trials Section ──────────────────────────────────

_TRIAL_STATUS_COLORS = {
    "RECRUITING": "#1b7a3d",
    "ACTIVE_NOT_RECRUITING": "#b45309",
    "COMPLETED": "#1a56db",
    "TERMINATED": "#6b7280",
    "WITHDRAWN": "#6b7280",
}


def _render_clinical_trials(trials: list[dict]):
    """Render clinical trial links."""
    st.markdown(
        f'<div style="margin:0.3rem 0; font-size:0.85rem;">'
        f'<strong>Clinical Trials:</strong> {len(trials)} found</div>',
        unsafe_allow_html=True,
    )
    for trial in trials:
        status = trial.get("status", "Unknown")
        color = _TRIAL_STATUS_COLORS.get(status, "#6b7280")
        status_display = status.replace("_", " ").title()
        phase = f" | {trial['phase']}" if trial.get("phase") else ""
        title = trial.get("title", "")
        url = trial.get("url", "")
        nct = trial.get("nct_id", "")
        st.markdown(
            f'<div style="font-size:0.8rem; margin-left:0.5rem; margin-bottom:0.3rem;">'
            f'<a href="{url}" target="_blank" style="color:#00606d; text-decoration:none;">{nct}</a> '
            f'<span style="color:{color}; font-weight:600;">{status_display}</span>{phase} '
            f'— {title[:80]}{"..." if len(title) > 80 else ""}</div>',
            unsafe_allow_html=True,
        )


# ── Sidebar Filters ───────────────────────────────────────────

def render_sidebar_filters():
    """Render sidebar filters. Returns (min_confidence, fda_filters)."""
    with st.sidebar:
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


def render_inline_filters():
    """Render filters inline in main content area. Returns (min_confidence, fda_filters)."""
    col_slider, col_fda1, col_fda2, col_fda3 = st.columns([3, 1, 1, 1])

    with col_slider:
        min_conf = st.slider(
            "Confidence Threshold",
            min_value=0,
            max_value=100,
            value=0,
            format="%d%%",
        )

    with col_fda1:
        show_approved = st.checkbox("FDA Approved", value=True)
    with col_fda2:
        show_not_found = st.checkbox("Not in FDA DB", value=True)
    with col_fda3:
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
