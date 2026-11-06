"""Esquema de entrada de OnTrack (etapa 9): qué pregunta la app, qué valores son posibles y qué conoce el modelo.

El esquema es un archivo (`models/input_schema.json`) que viaja con el modelo. Lo usan `predict.py` (para validar un pedido) y la
app (para construir su formulario). Se arma **solo con el entrenamiento** y con los resultados ya guardados de la prueba
(`reports/07_metricas_test.json`, `07_segmentos.csv`); nunca lee la prueba.

Reemplaza la plantilla del curso, que era de clasificación: OnTrack es **regresión** y recomienda un plazo en días.
"""
import json
from pathlib import Path

import pandas as pd

from olistbr.config import UNKNOWN_CATEGORY

# Los 11 datos que se le piden a la persona (lo que se conoce al momento de la compra). El vendedor no se pregunta: su código
# es anónimo y nadie lo conoce; el modelo lo trata como «otros» (vendedores poco frecuentes).
FORM_FIELDS = [
    "order_purchase_timestamp", "seller_state", "customer_state", "distance_km", "product_category_name", "n_items",
    "main_seller_items", "price_sum", "freight_sum", "weight_g_sum", "volume_cm3_sum",
]
REQUIRED_FIELDS = ["order_purchase_timestamp", "seller_state", "customer_state"]  # sin ellos el plazo no tiene sentido
NUMERIC_FIELDS = ["distance_km", "n_items", "main_seller_items", "price_sum", "freight_sum", "weight_g_sum", "volume_cm3_sum"]
INTEGER_FIELDS = ["n_items", "main_seller_items"]

# Lo POSIBLE en el mundo real (decisión del grupo, Q3 de `especificaciones/09-inferencia-app.md`), no lo observado en los datos.
# Fuera de estos rangos la predicción se niega a responder.
VALID_RANGES = {
    "distance_km": (0, 4500),          # ≈ la mayor distancia dentro de Brasil (máximo observado: 3.399 km)
    "price_sum": (0, 20000),           # reales; máximo observado 7.160
    "freight_sum": (0, 2000),          # máximo observado 1.795
    "n_items": (1, 30),                # un pedido tiene al menos un ítem; máximo observado 20
    "main_seller_items": (1, 30),      # además, no puede superar el número de ítems
    "weight_g_sum": (0, 200000),       # gramos; máximo observado 184.400
    "volume_cm3_sum": (0, 1500000),    # cm³; máximo observado 1.076.544
}
MIN_PURCHASE_DATE = "2016-01-01"       # antes de esto Olist no tenía datos

LABELS = {
    "order_purchase_timestamp": "Fecha de compra",
    "seller_state": "Estado del vendedor",
    "customer_state": "Estado del cliente",
    "distance_km": "Distancia vendedor–cliente (km)",
    "product_category_name": "Categoría del producto",
    "n_items": "Número de ítems",
    "main_seller_items": "Ítems del vendedor principal",
    "price_sum": "Precio total (R$)",
    "freight_sum": "Flete total (R$)",
    "weight_g_sum": "Peso total (g)",
    "volume_cm3_sum": "Volumen total (cm³)",
}

APP_TITLE = "OnTrack · Fecha de entrega recomendada"
RESPONSIBLE_USE = (
    "- La fecha es una **estimación para planificar, no una promesa**: en la prueba, cerca de **1 de cada 4 pedidos** llegó después del plazo "
    "del modelo (es lo que implica que atrasarse cueste 3 veces más que sobrar). La fecha se cuenta desde el día de compra y se redondea al día, "
    "así que esa proporción es aproximada.\n"
    "- En los meses de alta demanda (fin de año e inicio de año) el plazo **quedó corto** en la validación: hay que vigilar la cobertura cada mes.\n"
    "- La **ubicación del cliente** cambia el plazo; revisa el % de pedidos tarde de su región.\n"
    "- El modelo se entrenó con pedidos de Olist de **enero de 2017 a agosto de 2018**: no se validó para otras fechas.\n"
    "- La decisión final es de una persona. Límites y cifras en `models/ficha_modelo.md`."
)


def build_schema(train: pd.DataFrame, test_results: dict | None = None, segments: pd.DataFrame | None = None, path=None) -> dict:
    """Arma (y opcionalmente guarda) el esquema a partir del ENTRENAMIENTO y de los resultados ya guardados de la prueba.

    - `train`: tabla con las columnas de `train.parquet`.
    - `test_results`: contenido de `reports/07_metricas_test.json` (para el % de pedidos tarde del modelo en la prueba).
    - `segments`: `reports/07_segmentos.csv` (para el % tarde por región del cliente en la prueba).
    No guarda identificadores ni filas: solo categorías, rangos, medianas y textos.
    """
    missing = [c for c in FORM_FIELDS if c not in train.columns]
    if missing:
        raise ValueError(f"faltan columnas en el entrenamiento: {', '.join(missing)}")
    states = sorted(set(train["seller_state"].dropna()) | set(train["customer_state"].dropna()))
    categories = sorted(c for c in train["product_category_name"].dropna().unique() if c != UNKNOWN_CATEGORY)
    dates = pd.to_datetime(train["order_purchase_timestamp"])
    schema = {
        "fields": list(FORM_FIELDS),
        "required": list(REQUIRED_FIELDS),
        "numeric": list(NUMERIC_FIELDS),
        "integer_columns": list(INTEGER_FIELDS),
        "labels": dict(LABELS),
        "states": states,
        "categories": categories,
        "valid_range": {k: [float(lo), float(hi)] for k, (lo, hi) in VALID_RANGES.items()},
        "observed_range": {c: [float(train[c].min()), float(train[c].max())] for c in NUMERIC_FIELDS},
        "median": {c: float(train[c].median()) for c in NUMERIC_FIELDS},
        "min_purchase_date": MIN_PURCHASE_DATE,
        "observed_period": [dates.min().date().isoformat(), dates.max().date().isoformat()],
        "late_share_total": None,
        "late_share_by_region": {},
        "app_title": APP_TITLE,
        "responsible_use": RESPONSIBLE_USE,
    }
    if test_results:
        model_row = next((r for r in test_results.get("resultado", []) if r["regla"].startswith("LightGBM")), None)
        if model_row is not None:
            schema["late_share_total"] = float(model_row["tarde"])
    if segments is not None:
        regions = segments[segments["segmento"] == "region"]
        schema["late_share_by_region"] = {str(g): float(t) for g, t in zip(regions["grupo"], regions["tarde"])}
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    return schema


def load_schema(path) -> dict:
    """Lee el esquema guardado por `build_schema`."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
