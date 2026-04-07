"""
CSS injection for BioLink v2 Streamlit app.
Implements the design system from DESIGN.md:
  - Manrope/Inter font pairing
  - Teal clinical color palette
  - No-border tonal layering
  - Confidence badge colors
"""

import streamlit as st

# Design system color tokens
COLORS = {
    "primary": "#00606d",
    "primary_container": "#267987",
    "background": "#f8fafb",
    "surface": "#f8fafb",
    "surface_container": "#eceeef",
    "surface_container_low": "#f2f4f5",
    "surface_container_lowest": "#ffffff",
    "on_surface": "#191c1d",
    "on_surface_variant": "#3e494a",
    "outline_variant": "#bdc9ca",
    "error": "#ba1a1a",
    "tertiary_fixed": "#d6e3ff",
    "secondary_container": "#cde7ed",
    "strong_green": "#06D6A0",
    "moderate_amber": "#FFD166",
    "speculative_gray": "#94A3B8",
}


def inject_css():
    """Inject the full design system CSS into the Streamlit app."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap');

        /* Global */
        .stApp {
            background-color: #f8fafb;
            font-family: 'Inter', sans-serif;
            color: #191c1d;
        }

        /* Hide default Streamlit header/footer */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* Headlines use Manrope */
        h1, h2, h3 {
            font-family: 'Manrope', sans-serif !important;
            color: #191c1d !important;
        }

        h1 {
            font-size: 2rem !important;
            font-weight: 800 !important;
            letter-spacing: -0.02em !important;
        }

        /* Primary color for highlighted text */
        .primary-text {
            color: #00606d;
        }

        /* Disclaimer banner */
        .disclaimer-banner {
            background-color: #f2f4f5;
            padding: 0.75rem 1rem;
            border-left: 3px solid #00606d;
            font-size: 0.8rem;
            color: #3e494a;
            margin-bottom: 1.5rem;
        }

        /* Search section */
        .search-hero {
            text-align: center;
            padding: 3rem 0 1rem 0;
        }

        .search-hero h1 {
            font-size: 2.2rem !important;
            margin-bottom: 0.25rem !important;
        }

        .search-subtitle {
            color: #3e494a;
            font-size: 0.95rem;
            margin-bottom: 2rem;
        }

        /* Example disease chips */
        .chip-container {
            display: flex;
            gap: 0.5rem;
            justify-content: center;
            flex-wrap: wrap;
            margin-top: 1rem;
        }

        .chip {
            background-color: #e6e8e9;
            color: #191c1d;
            padding: 0.35rem 0.85rem;
            border-radius: 1rem;
            font-size: 0.8rem;
            font-family: 'Inter', sans-serif;
            cursor: pointer;
            transition: background-color 0.2s;
        }

        .chip:hover {
            background-color: #cde7ed;
        }

        /* Result card */
        .result-card {
            background-color: #ffffff;
            padding: 1.25rem 1.5rem;
            margin-bottom: 0.75rem;
            border-left: 4px solid #bdc9ca;
            box-shadow: 0px 4px 12px rgba(25, 28, 29, 0.04);
        }

        .result-card-strong {
            border-left-color: #06D6A0;
        }

        .result-card-moderate {
            border-left-color: #FFD166;
        }

        .result-card-speculative {
            border-left-color: #94A3B8;
        }

        .result-card .drug-name {
            font-family: 'Manrope', sans-serif;
            font-size: 1.15rem;
            font-weight: 700;
            color: #191c1d;
            margin: 0;
        }

        /* Confidence badges */
        .badge {
            display: inline-block;
            padding: 0.2rem 0.65rem;
            border-radius: 1rem;
            font-size: 0.7rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .badge-strong {
            background-color: #d1fae5;
            color: #065f46;
        }

        .badge-moderate {
            background-color: #fef3c7;
            color: #92400e;
        }

        .badge-speculative {
            background-color: #e2e8f0;
            color: #475569;
        }

        /* FDA status badges */
        .fda-badge {
            display: inline-block;
            padding: 0.15rem 0.55rem;
            border-radius: 1rem;
            font-size: 0.65rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        .fda-approved {
            background-color: #d6e3ff;
            color: #001b3d;
        }

        .fda-not-found {
            background-color: #e2e8f0;
            color: #475569;
        }

        .fda-unknown {
            background-color: #f2f4f5;
            color: #6e797a;
        }

        /* Explanation section */
        .explanation-text {
            background-color: #f2f4f5;
            padding: 1rem 1.25rem;
            font-size: 0.85rem;
            color: #3e494a;
            line-height: 1.6;
            margin-top: 0.75rem;
        }

        /* Stat callout */
        .stat-row {
            display: flex;
            gap: 1rem;
            margin-top: 0.5rem;
        }

        .stat-item {
            font-size: 0.8rem;
            color: #3e494a;
        }

        .stat-value {
            font-weight: 600;
            color: #191c1d;
        }

        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #eceeef;
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            font-family: 'Manrope', sans-serif !important;
        }

        /* Streamlit input styling overrides */
        .stTextInput > div > div > input {
            font-family: 'Inter', sans-serif;
            border: 1px solid rgba(189, 201, 202, 0.3);
            border-radius: 0.5rem;
        }

        .stTextInput > div > div > input:focus {
            border-color: #00606d;
            box-shadow: 0 0 0 1px #00606d;
        }

        /* Button styling */
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #00606d, #267987);
            color: white;
            border: none;
            font-family: 'Inter', sans-serif;
            font-weight: 600;
        }

        /* Loading step */
        .loading-step {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0.5rem 0;
            font-size: 0.9rem;
        }

        .loading-step .step-done {
            color: #06D6A0;
        }

        .loading-step .step-active {
            color: #00606d;
        }

        .loading-step .step-pending {
            color: #bdc9ca;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
