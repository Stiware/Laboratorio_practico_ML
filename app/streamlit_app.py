"""OnTrack · Fecha de entrega recomendada (etapa 9).

Un formulario que pide los datos de un pedido y responde con la fecha de entrega que conviene prometer.

- Carga `models/model.joblib` y `models/input_schema.json` desde ESTE repositorio. **No consulta Databricks** ni lee `.env`.
- Valida los datos: si uno es imposible, explica qué corregir en vez de responder cualquier cosa; si uno es poco común, avisa.
- Toda la lógica está en `src/olistbr/inference/` (con pruebas); esta app solo dibuja.

Ejecutar en local (desde la raíz del repositorio):  streamlit run app/streamlit_app.py
Para probarla con otros archivos: variables de entorno `ONTRACK_MODEL_PATH` y `ONTRACK_SCHEMA_PATH`.
"""
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from olistbr.inference.predict import InputValidationError, load_model, predict_deadline  # noqa: E402
from olistbr.inference.schema import APP_TITLE, load_schema  # noqa: E402

MODEL_PATH = Path(os.environ.get("ONTRACK_MODEL_PATH", ROOT / "models" / "model.joblib"))
SCHEMA_PATH = Path(os.environ.get("ONTRACK_SCHEMA_PATH", ROOT / "models" / "input_schema.json"))
UNKNOWN = "(no sé)"
NUMERIC_ORDER = ["distance_km", "n_items", "main_seller_items", "price_sum", "freight_sum", "weight_g_sum", "volume_cm3_sum"]


@st.cache_resource
def get_model(path: str):
    return load_model(path)


@st.cache_data
def get_schema(path: str) -> dict:
    return load_schema(path)


def numeric_field(schema: dict, field: str):
    """Un campo numérico (con el valor típico del entrenamiento como punto de partida) y su casilla «No sé»."""
    label = schema["labels"][field]
    if field in schema["integer_columns"]:
        value = st.number_input(label, value=int(round(schema["median"][field])), step=1, key=field)
    else:
        value = st.number_input(label, value=float(schema["median"][field]), key=field)
    unknown = st.checkbox(f"No sé ({label.split(' (')[0].lower()})", key=f"unk_{field}")
    return np.nan if unknown else value


def build_form(schema: dict) -> dict:
    """Dibuja el formulario en tres columnas y devuelve el pedido como un diccionario."""
    labels, states = schema["labels"], schema["states"]
    order = {}
    left, middle, right = st.columns(3)
    with left:
        default_date = pd.Timestamp(schema["observed_period"][1]).date()
        order["order_purchase_timestamp"] = st.date_input(labels["order_purchase_timestamp"], value=default_date, key="order_purchase_timestamp")
        order["seller_state"] = st.selectbox(labels["seller_state"], states, index=states.index("SP") if "SP" in states else 0, key="seller_state")
        order["customer_state"] = st.selectbox(labels["customer_state"], states, index=states.index("SP") if "SP" in states else 0, key="customer_state")
        category = st.selectbox(labels["product_category_name"], schema["categories"] + [UNKNOWN], key="product_category_name")
        order["product_category_name"] = None if category == UNKNOWN else category
    for column, fields in ((middle, NUMERIC_ORDER[:4]), (right, NUMERIC_ORDER[4:])):
        with column:
            for field in fields:
                order[field] = numeric_field(schema, field)
    return order


def show_result(result: pd.DataFrame, schema: dict) -> None:
    row = result.iloc[0]
    left, right = st.columns(2)
    with left:
        st.metric("Fecha de entrega recomendada", row["fecha_recomendada"].strftime("%d/%m/%Y"))
    with right:
        st.metric("Plazo recomendado", f"{row['plazo_dias']:.1f} días".replace(".", ","))
    total = schema.get("late_share_total")
    region = schema.get("late_share_by_region", {}).get(row["region_cliente"])
    lines = []
    if total is not None:
        lines.append(f"En la prueba, el **{total:.0%}** de los pedidos llegó después del plazo del modelo (≈ 1 de cada 4).")
    if region is not None:
        lines.append(f"En la región del cliente (**{row['region_cliente']}**) fue el **{region:.0%}**.")
    if lines:
        st.info(" ".join(lines))
    for message in result.attrs.get("warnings", []):
        st.warning(message[0].upper() + message[1:])


def main() -> None:
    st.set_page_config(page_title="OnTrack", page_icon="📦", layout="wide")
    if not MODEL_PATH.exists() or not SCHEMA_PATH.exists():
        st.title(APP_TITLE)
        st.error("Falta `models/model.joblib` o `models/input_schema.json`. Ejecuta el notebook `notebooks/09_inferencia.ipynb` para generarlos.")
        st.stop()

    model, schema = get_model(str(MODEL_PATH)), get_schema(str(SCHEMA_PATH))
    st.title(schema["app_title"])
    st.caption("Herramienta de apoyo para planificar entregas. **No decide por sí sola.**")

    with st.form("pedido"):
        order = build_form(schema)
        submitted = st.form_submit_button("Recomendar fecha", type="primary")

    if submitted:
        if isinstance(order["order_purchase_timestamp"], date):
            order["order_purchase_timestamp"] = pd.Timestamp(order["order_purchase_timestamp"])
        try:
            show_result(predict_deadline(pd.DataFrame([order]), model, schema), schema)
        except InputValidationError as error:
            st.error(f"Revisa los datos: {error}")
    else:
        st.info("Completa los datos del pedido y pulsa **Recomendar fecha**. Marca *No sé* en lo que no conozcas.")

    with st.expander("Uso responsable", expanded=True):
        st.markdown(schema["responsible_use"])


main()
