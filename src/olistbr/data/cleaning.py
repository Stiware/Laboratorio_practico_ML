"""Limpieza del dataset de Olist: carga de los CSV, reglas del EDA y tabla de pedidos.

Cada decisión sale del EDA (`especificaciones/02-eda.md`, `reports/01_eda.md`) y de
`especificaciones/03-preprocesamiento.md`. Cada función implementa una regla, devuelve una copia y no
modifica la tabla que recibe:
  1. `build_target`: solo pedidos entregados con fecha de entrega; se crea `dias_entrega`.
  2. `drop_extreme_targets`: los pedidos de más de 100 días se separan y se reportan aparte.
  3. `aggregate_items`: una fila por pedido (suma de ítems, vendedor principal).
  4. `drop_leakage_columns`: se quita lo que se conoce después de comprar y los identificadores.
  6. La distancia vendedor-cliente se calcula en `olistbr.features.geo`.
  7. `clean_products`: categoría desconocida, traducción y peso 0 como vacío.
  9. `filter_stable_period`: compras entre 2017-01 y 2018-08.
 10. `load_raw` / `drop_personal_data`: no se cargan reseñas ni textos libres.
La regla 5 (partición por cliente) está en `olistbr.data.split` y la 8 (vendedores raros) en
`olistbr.features.categories`. `build_orders_table` encadena las reglas 1 a 7 en el orden aprobado.

Nada de aquí "aprende" de los datos: lo que depende del entrenamiento (medianas, vendedores frecuentes)
va en `features/` o en la etapa 4, dentro de un `Pipeline`.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from olistbr.config import (
    MAX_DELIVERY_DAYS,
    PERIOD_END,
    PERIOD_START,
    RAW_DIR,
    TARGET,
    UNKNOWN_CATEGORY,
)
from olistbr.features.geo import add_distance, zip_centroids

# nombre interno -> (archivo, columnas que se leen, columnas que son fechas)
RAW_TABLES = {
    "orders": (
        "olist_orders_dataset.csv",
        [
            "order_id",
            "customer_id",
            "order_status",
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ],
        [
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ],
    ),
    "items": (
        "olist_order_items_dataset.csv",
        ["order_id", "order_item_id", "product_id", "seller_id", "price", "freight_value"],
        [],
    ),
    "customers": (
        "olist_customers_dataset.csv",
        ["customer_id", "customer_unique_id", "customer_zip_code_prefix", "customer_state"],
        [],
    ),
    "products": (
        "olist_products_dataset.csv",
        [
            "product_id",
            "product_category_name",
            "product_weight_g",
            "product_length_cm",
            "product_height_cm",
            "product_width_cm",
        ],
        [],
    ),
    "sellers": (
        "olist_sellers_dataset.csv",
        ["seller_id", "seller_zip_code_prefix", "seller_state"],
        [],
    ),
    "geolocation": (
        "olist_geolocation_dataset.csv",
        [
            "geolocation_zip_code_prefix",
            "geolocation_lat",
            "geolocation_lng",
            "geolocation_city",
            "geolocation_state",
        ],
        [],
    ),
    "translation": (
        "product_category_name_translation.csv",
        ["product_category_name", "product_category_name_english"],
        [],
    ),
}


def drop_personal_data(df: pd.DataFrame) -> pd.DataFrame:
    """Quita los textos libres de reseñas y cualquier columna `review_*`.

    Son datos personales potenciales y además se conocen después de la compra (fuga).
    Devuelve una copia; no modifica la tabla que recibe.
    """
    personal = [c for c in df.columns if c.startswith("review_")]
    return df.drop(columns=personal)


def load_raw(raw_dir=RAW_DIR) -> dict:
    """Lee las 7 tablas permitidas de `raw_dir` y devuelve un diccionario de DataFrames.

    - Nunca lee las reseñas ni los pagos.
    - Convierte a fecha las columnas de fecha de los pedidos.
    - No escribe nada en `raw_dir`.
    """
    raw_dir = Path(raw_dir)
    tables = {}
    for name, (filename, columns, date_columns) in RAW_TABLES.items():
        # utf-8-sig: la tabla de traducción trae una marca BOM al inicio del archivo
        df = pd.read_csv(
            raw_dir / filename,
            usecols=columns,
            parse_dates=date_columns,
            encoding="utf-8-sig",
        )
        tables[name] = drop_personal_data(df)
    return tables


# Columnas que se conocen después de la compra o que identifican (decisiones 4, 8 y 10).
LEAKAGE_COLUMNS = [
    "order_status",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "customer_id",
    "customer_city",
    "customer_zip_code_prefix",
    "seller_zip_code_prefix",
    "seller_city",
]


def build_target(orders: pd.DataFrame) -> pd.DataFrame:
    """Decisión 1: deja solo los pedidos entregados con fecha de entrega y crea `dias_entrega`.

    `dias_entrega` = fecha de entrega al cliente - fecha de compra, en días (puede ser decimal).
    """
    delivered = orders[
        (orders["order_status"] == "delivered")
        & orders["order_delivered_customer_date"].notna()
    ].copy()
    elapsed = delivered["order_delivered_customer_date"] - delivered["order_purchase_timestamp"]
    delivered[TARGET] = elapsed.dt.total_seconds() / 86400
    return delivered


def filter_stable_period(
    df: pd.DataFrame, start: str = PERIOD_START, end: str = PERIOD_END
) -> pd.DataFrame:
    """Decisión 9: conserva compras entre `start` y `end`, ambos días completos incluidos."""
    purchase = df["order_purchase_timestamp"]
    inside = (purchase >= pd.Timestamp(start)) & (purchase < pd.Timestamp(end) + pd.Timedelta(days=1))
    return df[inside].copy()


def drop_extreme_targets(
    df: pd.DataFrame, max_days: float = MAX_DELIVERY_DAYS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Decisión 2: separa los pedidos que tardaron más de `max_days` días.

    Devuelve `(kept, extreme)`. Los extremos no se pierden: se guardan aparte para reportarlos.
    """
    is_extreme = df[TARGET] > max_days
    return df[~is_extreme].copy(), df[is_extreme].copy()


def clean_products(products: pd.DataFrame, translation: pd.DataFrame) -> pd.DataFrame:
    """Decisión 7: categoría vacía -> 'desconocida', categoría traducida y peso 0 -> vacío.

    Una categoría sin traducción conserva su nombre original. El peso no se imputa aquí:
    la mediana depende del entrenamiento y se calcula en la etapa 4.
    """
    result = products.copy()
    unique_translation = translation.drop_duplicates("product_category_name")  # por si una categoría viene repetida
    english = unique_translation.set_index("product_category_name")["product_category_name_english"]
    category = result["product_category_name"].map(english).fillna(result["product_category_name"])
    result["product_category_name"] = category.fillna(UNKNOWN_CATEGORY)
    result["product_weight_g"] = result["product_weight_g"].where(result["product_weight_g"] > 0)
    return result


def aggregate_items(
    items: pd.DataFrame, products: pd.DataFrame, sellers: pd.DataFrame
) -> pd.DataFrame:
    """Decisión 3: una fila por pedido a partir de sus ítems.

    - Suma de precio y flete, conteo de ítems.
    - Peso y volumen (largo x alto x ancho de cada ítem) sumados; si a algún ítem le falta el
      dato, el valor del pedido queda vacío en vez de sumarse a medias.
    - Vendedor principal: el que tiene más ítems en el pedido; en empate, el menor `seller_id`.
      Se agregan sus ítems (`main_seller_items`), su código postal y su estado.
    - Categoría: la del primer ítem (menor `order_item_id`) del vendedor principal.
    """
    product_cols = [
        "product_id", "product_category_name", "product_weight_g",
        "product_length_cm", "product_height_cm", "product_width_cm",
    ]
    df = items.merge(products[product_cols], on="product_id", how="left")
    df["volume_cm3"] = df["product_length_cm"] * df["product_height_cm"] * df["product_width_cm"]

    grouped = df.groupby("order_id")
    totals = grouped.agg(
        price_sum=("price", "sum"),
        freight_sum=("freight_value", "sum"),
        n_items=("order_item_id", "count"),
        weight_g_sum=("product_weight_g", "sum"),
        weight_known=("product_weight_g", "count"),
        volume_cm3_sum=("volume_cm3", "sum"),
        volume_known=("volume_cm3", "count"),
    )
    totals["weight_g_sum"] = totals["weight_g_sum"].where(totals["weight_known"] == totals["n_items"], np.nan)
    totals["volume_cm3_sum"] = totals["volume_cm3_sum"].where(totals["volume_known"] == totals["n_items"], np.nan)
    totals = totals.drop(columns=["weight_known", "volume_known"])

    # Vendedor principal: más ítems; en empate, menor seller_id.
    per_seller = df.groupby(["order_id", "seller_id"]).size().rename("main_seller_items").reset_index()
    main = (
        per_seller.sort_values(["order_id", "main_seller_items", "seller_id"], ascending=[True, False, True])
        .drop_duplicates("order_id")
    )

    # Categoría del primer ítem del vendedor principal.
    first_item = (
        df.merge(main[["order_id", "seller_id"]], on=["order_id", "seller_id"])
        .sort_values(["order_id", "order_item_id"])
        .drop_duplicates("order_id")[["order_id", "product_category_name"]]
    )

    result = (
        totals.reset_index()
        .merge(main, on="order_id")
        .merge(first_item, on="order_id")
        .merge(sellers[["seller_id", "seller_zip_code_prefix", "seller_state"]], on="seller_id", how="left")
    )
    result["product_category_name"] = result["product_category_name"].fillna(UNKNOWN_CATEGORY)
    return result


def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Decisión 4 (y 8 y 10): quita lo que se conoce después de comprar y los identificadores.

    Se quitan estado y fechas de entrega, reseñas, `customer_id`, ciudades y códigos postales.
    Se conservan `order_id` y `customer_unique_id` (para agrupar la partición, no para el modelo)
    y `order_approved_at`, que la etapa 4 mide con y sin ella.
    """
    to_drop = [c for c in df.columns if c in LEAKAGE_COLUMNS or c.startswith("review_")]
    return df.drop(columns=to_drop)


def _log(report, step: str, rule: str, before: int, after: int) -> None:
    """Anota cuántas filas había antes y después de una regla (si se pidió un reporte)."""
    if report is not None:
        report.append({"paso": step, "regla": rule, "filas_antes": before, "filas_despues": after})


def build_orders_table(raw_dir=RAW_DIR, report: list | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construye la tabla de pedidos (una fila por pedido entregado) desde los CSV originales.

    Devuelve `(orders, extreme_orders)`: la tabla para entrenar y los pedidos de más de 100 días
    (con las mismas columnas) que se excluyen y se reportan aparte.

    Si se pasa una lista vacía en `report`, se llena con las filas antes y después de cada regla
    (para el notebook y para `reports/02_preprocesamiento.md`).
    """
    tables = load_raw(raw_dir)

    # Pedido + persona (customer_unique_id agrupa la partición; el zip sirve para la distancia).
    customers = tables["customers"]
    orders = tables["orders"].merge(customers, on="customer_id", how="left")

    n = len(orders)
    orders = build_target(orders)
    _log(report, "1", "build_target: solo entregados con fecha de entrega", n, len(orders))
    n = len(orders)
    orders = filter_stable_period(orders)
    _log(report, "9", "filter_stable_period: compras 2017-01 a 2018-08", n, len(orders))
    n = len(orders)
    orders, extreme = drop_extreme_targets(orders)
    _log(report, "2", "drop_extreme_targets: más de 100 días (van aparte)", n, len(orders))

    products = clean_products(tables["products"], tables["translation"])
    items = aggregate_items(tables["items"], products, tables["sellers"])

    centroids = zip_centroids(tables["geolocation"])

    def finish(part: pd.DataFrame) -> pd.DataFrame:
        # inner: un pedido sin ítems no tiene qué predecir con precio, peso ni vendedor
        part = part.merge(items, on="order_id", how="inner")
        # la distancia usa los códigos postales, que se quitan justo después (drop_leakage_columns)
        part = add_distance(part, centroids)
        part = drop_leakage_columns(part)
        return part.reset_index(drop=True)

    n = len(orders)
    final_orders = finish(orders)
    _log(report, "3", "aggregate_items: una fila por pedido (los pedidos sin ítems salen)", n, len(final_orders))
    final_extreme = finish(extreme)
    return final_orders, final_extreme
