"""OnTrack · Simulador de tiempos de entrega (diseño integrado).

Integra el diseño del prototipo `app` (dashboard OnTrack) con el modelo entrenado del
laboratorio. La app no consulta Databricks: carga `models/model.joblib` y
`models/input_schema.json` desde este repositorio.

Ejecutar en local (desde la raíz del repositorio):

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from olistbr.inference.predict import load_model  # noqa: E402
from olistbr.inference.schema import load_schema  # noqa: E402
from simulator import render_simulator  # noqa: E402
from theme import footer, inject_css  # noqa: E402
from views import render_dashboard, render_eda, render_methodology  # noqa: E402

MODEL_PATH = Path(os.environ.get("ONTRACK_MODEL_PATH", ROOT / "models" / "model.joblib"))
SCHEMA_PATH = Path(os.environ.get("ONTRACK_SCHEMA_PATH", ROOT / "models" / "input_schema.json"))

VIEWS = ["Dashboard General", "Datos & EDA", "Metodología & Flujo", "Simulador OnTrack"]


@st.cache_resource
def get_model(path: str):
    return load_model(path)


@st.cache_data
def get_schema(path: str) -> dict:
    return load_schema(path)


def header() -> None:
    """Marca y estado del modelo, con el mismo lenguaje visual del prototipo."""
    st.html(
        '<div class="ot-header">'
        '<div class="ot-brand">'
        '<div class="ot-logo"><span class="ot-logo-mark">OT</span></div>'
        '<div class="ot-brand-copy"><span class="ot-brand-name">OnTrack</span>'
        '<span class="ot-brand-tag">Logistics</span></div>'
        "</div>"
        '<div class="ot-header-meta"><span class="ot-live-dot"></span>'
        "Modelo en línea · LightGBM cuantil 0,75</div>"
        "</div>"
    )


def main() -> None:
    st.set_page_config(
        page_title="OnTrack · Simulador de entregas",
        page_icon="📦",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_css()

    if not MODEL_PATH.exists() or not SCHEMA_PATH.exists():
        with st.container(key="ontrack_card"):
            header()
            st.html(
                '<div class="ot-section-head"><div>'
                '<div class="ot-eyebrow"><span class="ot-eyebrow-dot"></span>OnTrack · Logistics</div>'
                '<h2 class="ot-title">Falta el modelo entrenado</h2>'
                '<p class="ot-subtitle">La app necesita los archivos exportados por el laboratorio.</p>'
                "</div></div>"
            )
            st.error(
                "Falta `models/model.joblib` o `models/input_schema.json`. "
                "Ejecuta el notebook `notebooks/09_inferencia.ipynb` para generarlos."
            )
        st.stop()

    model = get_model(str(MODEL_PATH))
    schema = get_schema(str(SCHEMA_PATH))

    with st.container(key="ontrack_card"):
        header()
        # Enlace directo a una vista: ?vista=simulador | eda | metodologia
        start = {"simulador": VIEWS[3], "eda": VIEWS[1], "metodologia": VIEWS[2]}.get(
            st.query_params.get("vista", ""), "Dashboard General"
        )
        view = st.segmented_control(
            "Vista",
            VIEWS,
            default=start,
            key="view",
            label_visibility="collapsed",
            width="stretch",
        )
        view = view or "Dashboard General"

        if view == "Dashboard General":
            render_dashboard(schema)
        elif view == "Datos & EDA":
            render_eda()
        elif view == "Metodología & Flujo":
            render_methodology()
        else:
            render_simulator(model, schema)

        st.html(footer())


main()
