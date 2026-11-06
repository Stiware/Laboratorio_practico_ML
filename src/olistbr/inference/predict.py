"""Inferencia de OnTrack (etapa 9): de los datos de un pedido a una fecha de entrega recomendada.

1. `validate_orders`: un dato **imposible** (fuera del rango posible del esquema, un estado que no existe, más ítems del vendedor principal
   que ítems totales…) hace que la predicción **se niegue y explique qué corregir**; un dato poco común solo genera una advertencia.
2. `order_to_features`: completa lo que la persona no sabe (el vendedor se toma como «otros», `missing_coords` si falta la distancia,
   categoría «desconocida») y aplica `add_features`, igual que en el entrenamiento, para llegar a las 25 columnas del modelo.
3. `predict_deadline`: el modelo predice el plazo (percentil 75, en días); la fecha recomendada es la compra más el plazo, **redondeada hacia
   arriba al día**, para que la promesa nunca quede antes del plazo del modelo.

Reemplaza la plantilla del curso, que era de clasificación (clase y probabilidades).
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from olistbr.config import OTHER_LABEL, UNKNOWN_CATEGORY
from olistbr.features.engineering import add_features
from olistbr.inference.schema import FORM_FIELDS

DATE_FIELD = "order_purchase_timestamp"


class InputValidationError(ValueError):
    """Los datos del pedido no permiten dar un plazo; el mensaje dice qué corregir."""


def load_model(path):
    """Carga el modelo exportado (`models/model.joblib`). Solo cargar modelos del propio grupo: un pickle puede ejecutar código."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no existe {path}: ejecuta el notebook 09 para exportar el modelo")
    return joblib.load(path)


def _fmt(value) -> str:
    """Número con separador de miles y coma decimal, como en los informes."""
    text = f"{float(value):,.1f}".rstrip("0").rstrip(".")
    return text.replace(",", "_").replace(".", ",").replace("_", ".")


def _to_date(value):
    """La fecha de compra como fecha sin zona horaria (si trae zona, se usa su hora local); NaT si no es una fecha."""
    date = pd.to_datetime(value, errors="coerce")
    if pd.notna(date) and date.tzinfo is not None:
        date = date.tz_localize(None)
    return date


def _as_frame(orders) -> pd.DataFrame:
    frame = pd.DataFrame(orders).reset_index(drop=True)
    for field in FORM_FIELDS:
        if field not in frame.columns:
            frame[field] = np.nan
    return frame


def validate_orders(orders, schema: dict) -> list:
    """Revisa cada pedido. Si hay algún dato imposible lanza `InputValidationError` con **todos** los motivos; si no, devuelve
    una lista de advertencias por pedido (datos poco comunes, «No sé», fechas fuera del periodo de entrenamiento)."""
    frame = _as_frame(orders)
    labels = schema["labels"]
    errors, warnings = [], []
    for i, row in frame.iterrows():
        prefix = f"Pedido {i + 1}: " if len(frame) > 1 else ""
        row_errors, row_warnings = [], []
        for field in schema["required"]:
            value = row[field]
            if value is None or (not isinstance(value, str) and pd.isna(value)) or (isinstance(value, str) and not value.strip()):
                row_errors.append(f"«{labels[field]}» es obligatorio")
        date = _to_date(row[DATE_FIELD])
        if pd.notna(row[DATE_FIELD]) and pd.isna(date):
            row_errors.append(f"«{labels[DATE_FIELD]}» no es una fecha válida")
        elif pd.notna(date):
            if date < pd.Timestamp(schema["min_purchase_date"]):
                row_errors.append(f"«{labels[DATE_FIELD]}» es anterior a 2016 (no hay datos de Olist antes)")
            start, end = (pd.Timestamp(d) for d in schema["observed_period"])
            if not start <= date.normalize() <= end:
                row_warnings.append(
                    f"la fecha está fuera del periodo de entrenamiento ({schema['observed_period'][0]} a {schema['observed_period'][1]}): "
                    "el modelo no se validó para otras fechas"
                )
        for field in ("seller_state", "customer_state"):
            value = row[field]
            present = value is not None and not (not isinstance(value, str) and pd.isna(value)) and str(value).strip()
            if present and (not isinstance(value, str) or value not in schema["states"]):
                row_errors.append(f"«{labels[field]}» = {value!r} no es un estado de Brasil conocido")
        for field in schema["numeric"]:
            value = pd.to_numeric(row[field], errors="coerce")
            if pd.isna(value):
                if pd.notna(row[field]):
                    row_errors.append(f"«{labels[field]}» debe ser un número")
                elif field == "distance_km":
                    row_warnings.append("sin la distancia el plazo es menos preciso (el modelo usa un valor típico)")
                else:
                    row_warnings.append(f"sin «{labels[field]}» el modelo usa un valor típico")
                continue
            if field in schema["integer_columns"] and value != round(value):
                row_errors.append(f"«{labels[field]}» debe ser un número entero (se recibió {_fmt(value)})")
                continue
            low, high = schema["valid_range"][field]
            if not low <= value <= high:
                row_errors.append(f"«{labels[field]}» = {_fmt(value)} está fuera de lo posible ({_fmt(low)} a {_fmt(high)})")
                continue
            obs_low, obs_high = schema["observed_range"][field]
            if not obs_low <= value <= obs_high:
                row_warnings.append(f"«{labels[field]}» = {_fmt(value)} es poco común: el modelo vio pocos pedidos así")
        items, seller_items = (pd.to_numeric(row[f], errors="coerce") for f in ("n_items", "main_seller_items"))
        if pd.notna(items) and pd.notna(seller_items) and seller_items > items:
            row_errors.append("los ítems del vendedor principal no pueden ser más que el número de ítems del pedido")
        elif pd.isna(items) and pd.notna(seller_items):
            row_warnings.append("sin el número de ítems no se puede comprobar que los del vendedor principal no lo superen")
        category = row["product_category_name"]
        if category is None or (not isinstance(category, str) and pd.isna(category)):
            row_warnings.append("sin la categoría se usa «desconocida»")
        elif category not in schema["categories"]:
            row_warnings.append(f"la categoría {category!r} no aparece en el entrenamiento: el modelo la trata como poco frecuente")
        errors.extend(prefix + e for e in row_errors)
        warnings.append(row_warnings)
    if errors:
        raise InputValidationError("; ".join(errors))
    return warnings


def order_to_features(orders) -> pd.DataFrame:
    """Convierte los datos del formulario en las columnas que espera el modelo (las de `add_features`)."""
    frame = _as_frame(orders)
    table = pd.DataFrame({
        "order_purchase_timestamp": pd.to_datetime([_to_date(v) for v in frame[DATE_FIELD]]),
        "order_estimated_delivery_date": pd.NaT,  # solo alimenta `promised_days`, que el modelo nunca usa
        "seller_state": frame["seller_state"],
        "customer_state": frame["customer_state"],
        "product_category_name": frame["product_category_name"].where(frame["product_category_name"].notna(), UNKNOWN_CATEGORY),
        "seller_id": OTHER_LABEL,  # vendedor desconocido para la persona: como los vendedores poco frecuentes del entrenamiento
    })
    for field in ("distance_km", "n_items", "main_seller_items", "price_sum", "freight_sum", "weight_g_sum", "volume_cm3_sum"):
        table[field] = pd.to_numeric(frame[field], errors="coerce").astype(float)
    table["order_estimated_delivery_date"] = pd.to_datetime(table["order_estimated_delivery_date"])
    table["missing_coords"] = table["distance_km"].isna().astype(int)
    return add_features(table)


def predict_deadline(orders, model, schema: dict) -> pd.DataFrame:
    """Plazo recomendado (días), fecha recomendada, región del cliente, % tarde de esa región en la prueba y advertencias.

    Lanza `InputValidationError` si algún dato es imposible. Las advertencias van en la columna `advertencias` y en `attrs["warnings"]`.
    """
    warnings = validate_orders(orders, schema)
    features = order_to_features(orders)
    plan = np.asarray(model.predict(features), dtype=float)
    deadline = (features["order_purchase_timestamp"] + pd.to_timedelta(plan, unit="D")).dt.ceil("D").dt.date
    regions = features["customer_region"]
    result = pd.DataFrame({
        "plazo_dias": plan,
        "fecha_recomendada": deadline.to_numpy(),
        "region_cliente": regions.to_numpy(),
        "pct_tarde_region": [schema["late_share_by_region"].get(r, np.nan) for r in regions],
        "advertencias": ["; ".join(w) for w in warnings],
    })
    result.attrs["warnings"] = [w for row in warnings for w in row]
    return result
