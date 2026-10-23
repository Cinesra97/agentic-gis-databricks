"""Streamlit styling helpers (dark-mode friendly)."""

from __future__ import annotations

import streamlit as st

KIND_BADGE = {
    "observation": ("Observation", "blue"),
    "inference": ("Inference", "orange"),
    "verified": ("Verified", "green"),
}

NODE_LABELS = {
    "start": "Reading the question",
    "discover": "Searching authorized datasets",
    "plan": "Writing an analysis plan",
    "review_sql": "Validating SQL against the safety policy",
    "approve": "Waiting for SQL approval",
    "execute": "Running analysis steps",
    "reflect": "Checking results",
    "finalize": "Preparing map layers and citations",
    "respond": "Writing the answer",
}

# Categorical series colors for charts (fixed order; validated for adjacent-pair CVD separation).
SERIES_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
SERIES_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
OTHER_GRAY = "#898781"

CSS = """
<style>
.block-container {padding-top: 2.4rem; padding-bottom: 1rem; max-width: 100%;}
section[data-testid="stSidebar"] .block-container {padding-top: 1rem;}
.agis-title {font-size: 1.35rem; font-weight: 650; margin: 0; line-height: 1.2;}
.agis-sub {opacity: .72; font-size: .85rem; margin-top: 2px;}
.agis-headline {font-weight: 600; font-size: 1.0rem; margin-bottom: .25rem;}
.agis-muted {opacity: .7; font-size: .82rem;}
.agis-sql-note {font-size: .8rem; opacity: .8;}
div[data-testid="stChatMessage"] {padding: .5rem .6rem;}
iframe[title="streamlit_folium.st_folium"] {border-radius: 8px;}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
