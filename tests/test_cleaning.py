"""Pruebas de la limpieza: carga sin datos personales, reglas del EDA y tabla de pedidos.

Una función, sus casos normal y sucio, hechos a mano (no leen `data/raw/`).
"""
import numpy as np
import pandas as pd

from olistbr.features.geo import haversine_km
from olistbr.data.cleaning import (
    aggregate_items,
    build_orders_table,
    build_target,
    clean_products,
    drop_extreme_targets,
    drop_leakage_columns,
    drop_personal_data,
    filter_stable_period,
    load_raw,
)


def test_build_target():
    """Decisión 1: solo pedidos entregados con fecha de entrega; dias_entrega en días."""
    orders = pd.DataFrame({
        "order_id": ["a", "b", "c"],
        "order_status": ["delivered", "delivered", "canceled"],
        "order_purchase_timestamp": pd.to_datetime(["2017-03-01 10:00", "2017-03-01 00:00", "2017-03-01 00:00"]),
        "order_delivered_customer_date": pd.to_datetime(["2017-03-11 10:00", None, None]),
    })
    result = build_target(orders)
    assert list(result["order_id"]) == ["a"]  # b (sin fecha) y c (cancelado) salen
    assert result["dias_entrega"].iloc[0] == 10.0


def test_build_target_counts_partial_days():
    orders = pd.DataFrame({
        "order_id": ["a"],
        "order_status": ["delivered"],
        "order_purchase_timestamp": pd.to_datetime(["2017-03-01 00:00"]),
        "order_delivered_customer_date": pd.to_datetime(["2017-03-01 12:00"]),
    })
    assert build_target(orders)["dias_entrega"].iloc[0] == 0.5


def test_filter_stable_period():
    """Decisión 9: 2017-01-01 y 2018-08-31 (todo el día) entran; antes y después, no."""
    df = pd.DataFrame({
        "order_id": ["antes", "inicio", "fin", "despues"],
        "order_purchase_timestamp": pd.to_datetime([
            "2016-12-31 23:59", "2017-01-01 00:00", "2018-08-31 23:00", "2018-09-01 00:00",
        ]),
    })
    result = filter_stable_period(df)
    assert list(result["order_id"]) == ["inicio", "fin"]


def test_drop_extreme_targets():
    """Decisión 2: 100,0 se queda; 100,1 sale y se guarda aparte."""
    df = pd.DataFrame({"order_id": ["a", "b", "c"], "dias_entrega": [100.0, 100.1, 5.0]})
    kept, extreme = drop_extreme_targets(df)
    assert list(kept["order_id"]) == ["a", "c"]
    assert list(extreme["order_id"]) == ["b"]


def test_clean_products():
    """Decisión 7: categoría vacía -> 'desconocida'; traducida; sin traducción se conserva; peso 0 -> vacío."""
    products = pd.DataFrame({
        "product_id": ["p1", "p2", "p3"],
        "product_category_name": ["beleza_saude", None, "categoria_sin_traduccion"],
        "product_weight_g": [500.0, 0.0, np.nan],
    })
    translation = pd.DataFrame({
        "product_category_name": ["beleza_saude"],
        "product_category_name_english": ["health_beauty"],
    })
    result = clean_products(products, translation).set_index("product_id")
    assert result.loc["p1", "product_category_name"] == "health_beauty"
    assert result.loc["p2", "product_category_name"] == "desconocida"
    assert result.loc["p3", "product_category_name"] == "categoria_sin_traduccion"
    assert result.loc["p1", "product_weight_g"] == 500.0
    assert np.isnan(result.loc["p2", "product_weight_g"])  # el 0 era imposible


def _products():
    return pd.DataFrame({
        "product_id": ["p1", "p2", "p3"],
        "product_category_name": ["health_beauty", "desconocida", "bed_bath_table"],
        "product_weight_g": [500.0, np.nan, 200.0],
        "product_length_cm": [10.0, 5.0, 20.0],
        "product_height_cm": [10.0, 5.0, 10.0],
        "product_width_cm": [10.0, 5.0, 5.0],
    })


def _sellers():
    return pd.DataFrame({
        "seller_id": ["s1", "s2"],
        "seller_zip_code_prefix": [3000, 2000],
        "seller_state": ["MG", "RJ"],
    })


def test_aggregate_items_sums_counts_and_picks_main_seller():
    """Decisión 3: una fila por pedido; vendedor principal = el de más ítems; empate = menor seller_id."""
    items = pd.DataFrame({
        "order_id": ["o1", "o1", "o3", "o3", "o3"],
        "order_item_id": [1, 2, 1, 2, 3],
        "product_id": ["p1", "p1", "p3", "p1", "p1"],
        "seller_id": ["s2", "s2", "s1", "s2", "s2"],
        "price": [100.0, 50.0, 30.0, 20.0, 10.0],
        "freight_value": [10.0, 5.0, 3.0, 2.0, 1.0],
    })
    result = aggregate_items(items, _products(), _sellers()).set_index("order_id")
    assert len(result) == 2
    # o1: dos ítems del mismo vendedor
    assert result.loc["o1", "price_sum"] == 150.0
    assert result.loc["o1", "freight_sum"] == 15.0
    assert result.loc["o1", "n_items"] == 2
    assert result.loc["o1", "weight_g_sum"] == 1000.0
    assert result.loc["o1", "volume_cm3_sum"] == 2000.0
    assert result.loc["o1", "seller_id"] == "s2"
    assert result.loc["o1", "main_seller_items"] == 2
    # o3: s2 tiene 2 ítems y s1 uno, gana s2; su zip y estado salen de sellers
    assert result.loc["o3", "seller_id"] == "s2"
    assert result.loc["o3", "main_seller_items"] == 2
    assert result.loc["o3", "seller_zip_code_prefix"] == 2000
    assert result.loc["o3", "seller_state"] == "RJ"
    assert result.loc["o3", "n_items"] == 3


def test_aggregate_items_tie_goes_to_smallest_seller_id():
    items = pd.DataFrame({
        "order_id": ["o1", "o1"],
        "order_item_id": [1, 2],
        "product_id": ["p1", "p3"],
        "seller_id": ["s2", "s1"],
        "price": [1.0, 1.0],
        "freight_value": [1.0, 1.0],
    })
    result = aggregate_items(items, _products(), _sellers()).iloc[0]
    assert result["seller_id"] == "s1"
    # la categoría sale del primer ítem (menor order_item_id) del vendedor principal
    assert result["product_category_name"] == "bed_bath_table"


def test_aggregate_items_missing_weight_makes_order_weight_empty():
    """Si algún ítem no tiene peso, el peso del pedido queda vacío (no se suma a medias)."""
    items = pd.DataFrame({
        "order_id": ["o1", "o1"],
        "order_item_id": [1, 2],
        "product_id": ["p1", "p2"],  # p2 no tiene peso
        "seller_id": ["s1", "s1"],
        "price": [1.0, 1.0],
        "freight_value": [1.0, 1.0],
    })
    result = aggregate_items(items, _products(), _sellers()).iloc[0]
    assert np.isnan(result["weight_g_sum"])
    assert result["volume_cm3_sum"] == 1125.0  # las dimensiones sí están: 1000 + 125
    assert result["n_items"] == 2


def test_aggregate_items_unknown_product_gets_unknown_category():
    items = pd.DataFrame({
        "order_id": ["o1"],
        "order_item_id": [1],
        "product_id": ["pX"],  # no existe en productos
        "seller_id": ["s1"],
        "price": [1.0],
        "freight_value": [1.0],
    })
    result = aggregate_items(items, _products(), _sellers()).iloc[0]
    assert result["product_category_name"] == "desconocida"
    assert np.isnan(result["weight_g_sum"])


def test_drop_leakage_columns():
    """Decisión 4: se quitan las columnas posteriores a la compra y los identificadores; se conserva order_approved_at."""
    df = pd.DataFrame({
        "order_id": ["o1"],
        "customer_unique_id": ["u1"],
        "order_approved_at": [pd.Timestamp("2017-03-01")],
        "order_status": ["delivered"],
        "order_delivered_customer_date": [pd.Timestamp("2017-03-11")],
        "order_delivered_carrier_date": [pd.Timestamp("2017-03-03")],
        "review_score": [5],
        "customer_id": ["c1"],
        "customer_zip_code_prefix": [1000],
        "seller_zip_code_prefix": [3000],
        "customer_city": ["sao paulo"],
        "price_sum": [150.0],
        "dias_entrega": [10.0],
    })
    result = drop_leakage_columns(df)
    assert set(result.columns) == {
        "order_id", "customer_unique_id", "order_approved_at", "price_sum", "dias_entrega",
    }


def test_clean_products_tolerates_repeated_translation():
    """Si la tabla de traducción trae una categoría repetida, no falla ni duplica productos."""
    products = pd.DataFrame({
        "product_id": ["p1"], "product_category_name": ["beleza_saude"], "product_weight_g": [500.0],
    })
    translation = pd.DataFrame({
        "product_category_name": ["beleza_saude", "beleza_saude"],
        "product_category_name_english": ["health_beauty", "health_beauty"],
    })
    result = clean_products(products, translation)
    assert len(result) == 1
    assert result["product_category_name"].iloc[0] == "health_beauty"


def test_empty_tables_do_not_break_filters():
    """Una tabla vacía pasa por los filtros sin errores."""
    empty = pd.DataFrame({
        "order_id": pd.Series(dtype="str"),
        "order_purchase_timestamp": pd.Series(dtype="datetime64[ns]"),
        "dias_entrega": pd.Series(dtype="float64"),
    })
    assert filter_stable_period(empty).empty
    kept, extreme = drop_extreme_targets(empty)
    assert kept.empty and extreme.empty


def test_build_target_keeps_negative_days_visible():
    """Si una entrega figurara antes de la compra, `dias_entrega` sale negativo (no se oculta).

    Hoy no ocurre en los datos reales (mínimo 0,53 días); el notebook lo comprueba con
    «dias_entrega entre 0 y 100» y fallaría si apareciera.
    """
    orders = pd.DataFrame({
        "order_id": ["a"],
        "order_status": ["delivered"],
        "order_purchase_timestamp": pd.to_datetime(["2017-03-10 00:00"]),
        "order_delivered_customer_date": pd.to_datetime(["2017-03-09 00:00"]),
    })
    assert build_target(orders)["dias_entrega"].iloc[0] == -1.0


def test_load_raw_has_no_reviews(raw_dir):
    """Las reseñas y los pagos no se cargan, aunque los archivos estén en la carpeta."""
    tables = load_raw(raw_dir)
    assert set(tables) == {
        "orders", "items", "customers", "products", "sellers", "geolocation", "translation",
    }
    for df in tables.values():
        assert not any(c.startswith("review_") for c in df.columns)
        assert "payment_value" not in df.columns


def test_load_raw_converts_dates_and_reads_translation_with_bom(raw_dir):
    tables = load_raw(raw_dir)
    assert pd.api.types.is_datetime64_any_dtype(tables["orders"]["order_purchase_timestamp"])
    assert "product_category_name" in tables["translation"].columns  # la marca BOM no la daña


def test_no_personal_columns():
    """Caso sucio: una tabla con textos libres de reseñas pierde todas las columnas review_*."""
    dirty = pd.DataFrame({
        "order_id": ["o1"],
        "review_id": ["r1"],
        "review_comment_title": ["hola"],
        "review_comment_message": ["mi nombre es ..."],
        "review_score": [5],
    })
    clean = drop_personal_data(dirty)
    assert list(clean.columns) == ["order_id"]
    assert "review_comment_message" in dirty.columns  # la entrada no se modifica


def test_no_personal_columns_leaves_normal_table_untouched():
    normal = pd.DataFrame({"order_id": ["o1"], "price": [10.0]})
    pd.testing.assert_frame_equal(drop_personal_data(normal), normal)


def test_load_raw_does_not_modify_raw(raw_dir):
    """Los archivos originales quedan con el mismo tamaño y la misma fecha de modificación."""
    before = {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in raw_dir.glob("*.csv")}
    load_raw(raw_dir)
    after = {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in raw_dir.glob("*.csv")}
    assert before == after


def test_build_orders_table_smoke(raw_dir):
    orders, extreme = build_orders_table(raw_dir)

    # o1, o2, o3 pasan; o4 (cancelado), o5 (2016), o6 (extremo) y o7 (sin fecha de entrega) no
    assert sorted(orders["order_id"]) == ["o1", "o2", "o3"]
    assert orders["order_id"].is_unique
    assert list(extreme["order_id"]) == ["o6"]  # el extremo se guarda aparte

    assert orders["dias_entrega"].between(0, 100).all()
    assert (extreme["dias_entrega"] > 100).all()
    assert orders["order_purchase_timestamp"].between("2017-01-01", "2018-08-31 23:59:59").all()

    # sin columnas de fuga ni identificadores
    forbidden = {"order_status", "order_delivered_customer_date", "customer_id", "customer_city"}
    assert not forbidden & set(orders.columns)
    assert not any(c.startswith("review_") for c in orders.columns)

    # extremos y entrenamiento tienen las mismas columnas
    assert list(orders.columns) == list(extreme.columns)

    # los valores salen bien: o1 tardó 10 días y tiene 2 ítems
    o1 = orders.set_index("order_id").loc["o1"]
    assert o1["dias_entrega"] == 10.0
    assert o1["n_items"] == 2
    assert o1["customer_unique_id"] == "u1"
    assert pd.isna(o1["weight_g_sum"])  # p2 pesa 0 -> vacío -> el pedido queda vacío

    # distancia: cliente en zip 1000 (promedio -23.6, -46.7), vendedor s1 en zip 3000 (-19.9, -43.9)
    expected = haversine_km(-23.6, -46.7, -19.9, -43.9)
    assert abs(o1["distance_km"] - expected) < 0.01
    assert o1["missing_coords"] == 0
    assert orders["missing_coords"].eq(0).all()
    # los códigos postales ya no están (solo servían para la distancia)
    assert "customer_zip_code_prefix" not in orders.columns
    assert "seller_zip_code_prefix" not in orders.columns


def test_build_orders_table_report_counts_rows_before_and_after(raw_dir):
    """Con `report` el pipeline anota las filas antes y después de cada regla."""
    report = []
    orders, _ = build_orders_table(raw_dir, report=report)
    steps = {row["paso"]: row for row in report}
    assert steps["1"]["filas_antes"] == 7 and steps["1"]["filas_despues"] == 5  # sin o4 (cancelado) ni o7 (sin fecha)
    assert steps["9"]["filas_despues"] == 4  # sin o5 (2016)
    assert steps["2"]["filas_despues"] == 3  # sin o6 (extremo)
    assert steps["3"]["filas_despues"] == len(orders) == 3
