"""Variables nuevas de la etapa 4: creación (sin estado), grupos de ideas y preprocesador.

Sigue la forma de `engineering.py` del ejemplo CreditSight:
- `add_features` crea las columnas derivadas. **No aprende de los datos** (cada fila se calcula con sus
  propios valores y con tablas fijas), así que se puede aplicar antes de partir sin filtrar información,
  y la app puede calcular lo mismo.
- `feature_columns` dice qué columnas usa el modelo según los grupos de ideas activos.
- `build_preprocessor` arma el `ColumnTransformer` (imputación, escalado, codificación e historial),
  que se ajusta solo con el entrenamiento y va dentro de un `Pipeline`.

Todas las columnas se conocen al comprar. El plazo que Olist ya prometió (`promised_days`) solo es una
referencia que se mide una vez: la app recomienda la fecha, así que no puede usarlo como entrada.
"""
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

from olistbr.config import MIN_CATEGORY_FREQUENCY, RANDOM_STATE

# --- Variables base (ya existen en train.parquet) -----------------------------------------------
BASE_NUMERIC = [
    "price_sum",
    "freight_sum",
    "n_items",
    "weight_g_sum",
    "volume_cm3_sum",
    "distance_km",
    "missing_coords",
]
BASE_CATEGORICAL = ["product_category_name", "seller_id", "seller_state", "customer_state"]

# Solo para medir una vez cuánto se pierde sin ella; nunca entra a un grupo ni al modelo final.
REFERENCE_ONLY_COLUMNS = ["promised_days"]

# --- Tabla fija estado -> región (IBGE). No depende de los datos, así que no filtra información. ----
REGION_BY_STATE = {
    **dict.fromkeys(["AC", "AP", "AM", "PA", "RO", "RR", "TO"], "Norte"),
    **dict.fromkeys(["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"], "Nordeste"),
    **dict.fromkeys(["DF", "GO", "MT", "MS"], "Centro-Oeste"),
    **dict.fromkeys(["ES", "MG", "RJ", "SP"], "Sudeste"),
    **dict.fromkeys(["PR", "RS", "SC"], "Sul"),
}

SLOW_WEEKDAYS = (3, 4, 5)  # jueves, viernes y sábado (0 = lunes): lo comprado esos días tarda más
SECONDS_PER_DAY = 86400


def black_friday(year: int) -> pd.Timestamp:
    """Cuarto viernes de noviembre del año dado (en 2017: 2017-11-24)."""
    first = pd.Timestamp(year=year, month=11, day=1)
    first_friday = first + pd.Timedelta(days=(4 - first.weekday()) % 7)
    return first_friday + pd.Timedelta(days=21)


def _log1p(series: pd.Series) -> pd.Series:
    """log(1 + x); un dato vacío, infinito o negativo da vacío (no se inventa un logaritmo)."""
    return np.log1p(series.where(np.isfinite(series) & (series >= 0)))


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """División segura: denominador cero, negativo, infinito o ausente -> vacío (nunca infinito)."""
    ratio = numerator / denominator.where(np.isfinite(denominator) & (denominator > 0))
    return ratio.where(np.isfinite(ratio))


def _flag(condition: pd.Series, *required: pd.Series) -> pd.Series:
    """Convierte una condición en 1.0 / 0.0, y en vacío si falta alguno de los datos de los que depende."""
    known = pd.Series(True, index=condition.index)
    for series in required:
        known &= series.notna()
    return condition.astype(float).where(known)


def _year_end_peak(purchase: pd.Series) -> pd.Series:
    """1.0 desde el lunes de la semana de Black Friday hasta el 31 de diciembre de ese año; 0.0 el resto."""
    day = purchase.dt.normalize()
    starts = {int(y): black_friday(int(y)) - pd.Timedelta(days=4) for y in purchase.dt.year.dropna().unique()}
    start = purchase.dt.year.map(starts)
    # Como `start` es de noviembre del mismo año de la compra, "día >= inicio" ya implica "<= 31 de diciembre".
    return _flag(day >= pd.to_datetime(start), purchase)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve una copia de `df` con las variables derivadas de la etapa 4 (ver `data-model.md`).

    Reglas: división por cero o dato ausente da vacío (NaN), nunca infinito; no modifica la tabla de entrada.
    Necesita las columnas de `train.parquet`.
    """
    out = df.copy()

    # F1: logaritmos (el modelo es lineal y la relación con los días es creciente pero curva)
    out["log_distance"] = _log1p(out["distance_km"])
    out["log_freight"] = _log1p(out["freight_sum"])
    out["log_price"] = _log1p(out["price_sum"])
    out["log_weight"] = _log1p(out["weight_g_sum"])
    out["log_volume"] = _log1p(out["volume_cm3_sum"])

    # F2 y F3: ¿mismo estado? y región (tabla fija)
    seller_state, customer_state = out["seller_state"], out["customer_state"]
    out["same_state"] = _flag(customer_state == seller_state, seller_state, customer_state)
    out["customer_region"] = customer_state.map(REGION_BY_STATE)
    out["seller_region"] = seller_state.map(REGION_BY_STATE)

    # F4 (solo referencia): días entre la compra y el plazo que Olist ya prometió
    purchase = out["order_purchase_timestamp"]
    promised = out["order_estimated_delivery_date"] - purchase
    out["promised_days"] = promised.dt.total_seconds() / SECONDS_PER_DAY

    # F5: calendario de la compra (día de la semana como categoría; jueves a sábado como marca)
    weekday = purchase.dt.dayofweek
    out["purchase_dow"] = weekday.map(lambda d: str(int(d)) if pd.notna(d) else np.nan)
    out["is_slow_weekday"] = _flag(weekday.isin(SLOW_WEEKDAYS), purchase)

    # F6: temporada alta de fin de año (calendario fijo)
    out["is_year_end_peak"] = _year_end_peak(purchase)

    # F7: la distancia pesa distinto dentro y fuera del estado (un modelo lineal no lo ve solo)
    out["log_distance_x_same_state"] = out["log_distance"] * out["same_state"]

    # F8: ruta vendedor -> cliente (para el historial; se codifica dentro del Pipeline)
    both_states = seller_state.notna() & customer_state.notna()
    out["route"] = (seller_state.fillna("") + "→" + customer_state.fillna("")).where(both_states)

    # F9: razones de flete y densidad
    out["freight_ratio"] = _ratio(out["freight_sum"], out["price_sum"])
    out["density"] = _ratio(out["weight_g_sum"], out["volume_cm3_sum"])

    # F10: el pedido tiene ítems de más de un vendedor
    out["multi_seller"] = _flag(out["n_items"] > out["main_seller_items"], out["n_items"], out["main_seller_items"])
    return out


# --- Grupos de ideas (los interruptores con los que se mide cada idea) -----------------------------
# Cada grupo es autocontenido: la idea 7 incluye sus componentes (log_distance y same_state) para no medir una
# interacción sin sus efectos principales. Si se activan varios grupos, las columnas repetidas cuentan una vez.
GROUPS = {
    "logs": {"numeric": ["log_distance", "log_freight", "log_price", "log_weight", "log_volume"]},  # F1
    "same_state": {"numeric": ["same_state"]},  # F2
    "regions": {"categorical": ["customer_region", "seller_region"]},  # F3
    "calendar": {"numeric": ["is_slow_weekday"], "categorical": ["purchase_dow"]},  # F5
    "peak_season": {"numeric": ["is_year_end_peak"]},  # F6
    "distance_interaction": {"numeric": ["log_distance", "same_state", "log_distance_x_same_state"]},  # F7
    "history": {"target_encoded": ["seller_id", "product_category_name", "route"]},  # F8
    "ratios": {"numeric": ["freight_ratio", "density"]},  # F9
    "multi_seller": {"numeric": ["multi_seller"]},  # F10
}


class FeatureColumns(NamedTuple):
    """Columnas que usa el modelo, según su tratamiento."""

    numeric: list
    categorical: list
    target_encoded: list


def feature_columns(groups=(), include_reference: bool = False) -> FeatureColumns:
    """Devuelve las columnas numéricas, categóricas y con historial de los grupos activos (sin repetir).

    - `groups`: nombres de `GROUPS`; uno desconocido lanza `ValueError`.
    - `include_reference=True` agrega `promised_days`: solo para medirlo una vez como referencia.
    """
    unknown = [g for g in groups if g not in GROUPS]
    if unknown:
        raise ValueError(f"grupo(s) de ideas desconocido(s): {unknown}; los válidos son {list(GROUPS)}")

    numeric = list(BASE_NUMERIC)
    categorical = list(BASE_CATEGORICAL)
    target_encoded: list = []
    for group in groups:
        for column in GROUPS[group].get("numeric", []):
            if column not in numeric:
                numeric.append(column)
        for column in GROUPS[group].get("categorical", []):
            if column not in categorical:
                categorical.append(column)
        for column in GROUPS[group].get("target_encoded", []):
            if column not in target_encoded:
                target_encoded.append(column)
    if include_reference:
        numeric += [c for c in REFERENCE_ONLY_COLUMNS if c not in numeric]
    return FeatureColumns(numeric, categorical, target_encoded)


def build_preprocessor(groups=(), include_reference: bool = False, scale: bool = True) -> ColumnTransformer:
    """Imputación, escalado, codificación e historial; se ajusta solo con entrenamiento (va dentro de un Pipeline).

    - Numéricas: mediana del entrenamiento con indicador de vacío, y escalado.
    - Categóricas: "Unknown" si falta y one-hot; las categorías poco frecuentes quedan agrupadas y las
      nuevas no fallan.
    - Historial (grupo `history`): promedio de `dias_entrega` por vendedor, categoría y ruta con
      `TargetEncoder`, que calcula cada fila de entrenamiento con las demás (validación cruzada interna).
      Una categoría nueva recibe el promedio global. El separador interno no agrupa por cliente, pero eso
      no contamina la validación: los pliegues exteriores sí son por cliente.
    """
    cols = feature_columns(groups, include_reference)
    numeric_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    categorical_steps = [
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="infrequent_if_exist", min_frequency=MIN_CATEGORY_FREQUENCY, sparse_output=False
            ),
        ),
    ]
    transformers = [
        ("num", Pipeline(numeric_steps), cols.numeric),
        ("cat", Pipeline(categorical_steps), cols.categorical),
    ]
    if cols.target_encoded:
        history_steps = [
            ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
            (
                "target",
                TargetEncoder(
                    target_type="continuous", cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
                ),
            ),
        ]
        transformers.append(("history", Pipeline(history_steps), cols.target_encoded))
    return ColumnTransformer(transformers, remainder="drop")
