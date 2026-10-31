"""Permite ejecutar `pytest tests` desde la raíz del repo sin instalar el paquete."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture
def raw_dir(tmp_path):
    """Carpeta con CSV mínimos hechos a mano, con las mismas columnas que los de Olist.

    Casos que contiene (todos inventados para probar, no son datos reales):
    - o1, o2: dos pedidos de la misma persona (u1); o3 y o5 son de u2; o4 y o6 son de u3.
    - o1 (10 días), o2 (7), o3 (20): entregados dentro del periodo.
    - o4: cancelado. o5: entregado en 2016 (fuera del periodo).
    - o6: tardó más de 100 días (extremo). o7: entregado sin fecha de entrega.
    - p2 no tiene categoría y pesa 0; s1 y s2 son los vendedores.
    - Una fila de geolocalización cae fuera de Brasil (zip 3000, ciudad belo horizonte).
    """
    def write(name, text):
        (tmp_path / name).write_text(text.strip() + "\n", encoding="utf-8")

    write("olist_orders_dataset.csv", """
order_id,customer_id,order_status,order_purchase_timestamp,order_approved_at,order_delivered_carrier_date,order_delivered_customer_date,order_estimated_delivery_date
o1,c1,delivered,2017-03-01 10:00:00,2017-03-01 11:00:00,2017-03-03 10:00:00,2017-03-11 10:00:00,2017-03-20 00:00:00
o2,c2,delivered,2017-05-01 10:00:00,2017-05-01 12:00:00,2017-05-02 10:00:00,2017-05-08 10:00:00,2017-05-15 00:00:00
o3,c3,delivered,2018-01-10 09:00:00,2018-01-10 10:00:00,2018-01-12 10:00:00,2018-01-30 09:00:00,2018-02-10 00:00:00
o4,c4,canceled,2017-07-01 10:00:00,,,,2017-07-20 00:00:00
o5,c5,delivered,2016-12-01 10:00:00,2016-12-01 11:00:00,2016-12-02 10:00:00,2016-12-10 10:00:00,2016-12-20 00:00:00
o6,c6,delivered,2017-06-01 10:00:00,2017-06-01 11:00:00,2017-06-02 10:00:00,2017-10-30 10:00:00,2017-06-25 00:00:00
o7,c1,delivered,2017-08-01 10:00:00,2017-08-01 11:00:00,2017-08-02 10:00:00,,2017-08-20 00:00:00
""")
    write("olist_order_items_dataset.csv", """
order_id,order_item_id,product_id,seller_id,shipping_limit_date,price,freight_value
o1,1,p1,s1,2017-03-05 00:00:00,100.0,10.0
o1,2,p2,s1,2017-03-05 00:00:00,50.0,5.0
o2,1,p1,s2,2017-05-04 00:00:00,80.0,8.0
o3,1,p3,s1,2018-01-14 00:00:00,30.0,3.0
o3,2,p1,s2,2018-01-14 00:00:00,20.0,2.0
o4,1,p1,s1,2017-07-05 00:00:00,10.0,1.0
o5,1,p1,s1,2016-12-05 00:00:00,10.0,1.0
o6,1,p1,s1,2017-06-05 00:00:00,10.0,1.0
o7,1,p1,s1,2017-08-05 00:00:00,10.0,1.0
""")
    write("olist_customers_dataset.csv", """
customer_id,customer_unique_id,customer_zip_code_prefix,customer_city,customer_state
c1,u1,1000,sao paulo,SP
c2,u1,1000,sao paulo,SP
c3,u2,2000,rio de janeiro,RJ
c4,u3,3000,belo horizonte,MG
c5,u2,2000,rio de janeiro,RJ
c6,u3,3000,belo horizonte,MG
""")
    write("olist_products_dataset.csv", """
product_id,product_category_name,product_name_lenght,product_description_lenght,product_photos_qty,product_weight_g,product_length_cm,product_height_cm,product_width_cm
p1,beleza_saude,40,100,1,500,10,10,10
p2,,,,,0,5,5,5
p3,cama_mesa_banho,40,100,1,200,20,10,5
""")
    write("olist_sellers_dataset.csv", """
seller_id,seller_zip_code_prefix,seller_city,seller_state
s1,3000,belo horizonte,MG
s2,2000,rio de janeiro,RJ
""")
    write("olist_geolocation_dataset.csv", """
geolocation_zip_code_prefix,geolocation_lat,geolocation_lng,geolocation_city,geolocation_state
1000,-23.5,-46.6,sao paulo,SP
1000,-23.7,-46.8,sao paulo,SP
2000,-22.9,-43.2,rio de janeiro,RJ
3000,-19.9,-43.9,belo horizonte,MG
3000,40.0,-8.0,belo horizonte,MG
""")
    write("product_category_name_translation.csv", """
﻿product_category_name,product_category_name_english
beleza_saude,health_beauty
cama_mesa_banho,bed_bath_table
""")
    # Archivos que NO deben leerse nunca:
    write("olist_order_reviews_dataset.csv", """
review_id,order_id,review_score,review_comment_title,review_comment_message,review_creation_date,review_answer_timestamp
r1,o1,5,titulo secreto,mensaje secreto,2017-03-12 00:00:00,2017-03-13 00:00:00
""")
    write("olist_order_payments_dataset.csv", """
order_id,payment_sequential,payment_type,payment_installments,payment_value
o1,1,credit_card,1,150.0
""")
    return tmp_path


@pytest.fixture
def features_frame():
    """Tabla pequeña hecha a mano con las columnas de `train.parquet` (entrada de `add_features`).

    Casos (todos inventados para probar, no son datos reales):
    - o0: lunes, SP -> RJ, todo normal.   - o1: jueves, mismo estado SP/SP, precio 0 y volumen 0.
    - o2: domingo 2017-11-19, sin fecha prometida, estado del cliente fuera de la tabla ("XX"), todo vacío.
    - o3: lunes 2017-11-20 (inicio de la temporada alta), AM -> AM.  - o4: domingo 2017-12-31 (último día).
    - o5: lunes 2018-01-01 (ya fuera de la temporada).  - o6: viernes, con precio y distancia negativos.
    - o7: sábado.
    """
    ts = pd.Timestamp
    rows = [
        # id, compra, prometida, price, freight, n_items, main_items, weight, volume, dist, miss, categoría, vendedor, est. vend., est. cliente
        ("o0", ts("2017-03-06 10:00"), ts("2017-03-30 10:00"), 100.0, 20.0, 3, 2, 1000.0, 2000.0, 500.0, 0, "health_beauty", "s1", "SP", "RJ"),
        ("o1", ts("2017-03-09 10:00"), ts("2017-03-20 10:00"), 0.0, 10.0, 3, 3, 500.0, 0.0, 0.0, 0, "bed_bath_table", "s2", "SP", "SP"),
        ("o2", ts("2017-11-19 23:00"), pd.NaT, np.nan, np.nan, 1, 1, np.nan, np.nan, np.nan, 1, "desconocida", "s3", "SP", "XX"),
        ("o3", ts("2017-11-20 00:00"), ts("2017-12-20 00:00"), 50.0, 50.0, 1, 1, 200.0, 400.0, 3000.0, 0, "computers", "s4", "AM", "AM"),
        ("o4", ts("2017-12-31 12:00"), ts("2018-01-15 12:00"), 10.0, 5.0, 1, 1, 100.0, 100.0, 10.0, 0, "health_beauty", "s1", "SP", "SP"),
        ("o5", ts("2018-01-01 12:00"), ts("2018-01-20 12:00"), 10.0, 5.0, 1, 1, 100.0, 100.0, 10.0, 0, "health_beauty", "s1", "SP", "SP"),
        ("o6", ts("2017-03-10 10:00"), ts("2017-03-25 10:00"), -1.0, 5.0, 1, 1, 100.0, 100.0, -5.0, 0, "toys", "s5", "MG", "MG"),
        ("o7", ts("2017-03-11 10:00"), ts("2017-03-26 10:00"), 30.0, 6.0, 2, 1, 300.0, 600.0, 120.0, 0, "toys", "s5", "MG", "RJ"),
    ]
    cols = [
        "order_id", "order_purchase_timestamp", "order_estimated_delivery_date", "price_sum", "freight_sum",
        "n_items", "main_seller_items", "weight_g_sum", "volume_cm3_sum", "distance_km", "missing_coords",
        "product_category_name", "seller_id", "seller_state", "customer_state",
    ]
    df = pd.DataFrame(rows, columns=cols)
    df["customer_unique_id"] = [f"u{i}" for i in range(len(df))]
    df["dias_entrega"] = [10.0, 7.0, 12.0, 14.0, 9.0, 8.0, 6.0, 11.0]
    return df


@pytest.fixture
def frame_factory():
    """Devuelve una función que arma una tabla más grande, con las mismas columnas que `train.parquet`.

    `make(n, seed)` crea `n` pedidos de fechas entre 2017-06 y 2018-04, con clientes que a veces se repiten,
    tres vendedores, tres estados y un objetivo que depende de la distancia (para que haya algo que aprender).
    """
    def make(n: int = 120, seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        purchase = pd.Timestamp("2017-06-05") + pd.to_timedelta(rng.integers(0, 330, n), unit="D")
        n_items = rng.integers(1, 4, n)
        distance = rng.uniform(5, 2000, n)
        df = pd.DataFrame({
            "order_id": [f"o{i}" for i in range(n)],
            "customer_unique_id": [f"u{i}" for i in rng.integers(0, max(n // 2, 1), n)],
            "order_purchase_timestamp": purchase,
            "order_estimated_delivery_date": purchase + pd.to_timedelta(rng.integers(8, 35, n), unit="D"),
            "price_sum": rng.uniform(10, 300, n),
            "freight_sum": rng.uniform(5, 60, n),
            "n_items": n_items,
            "main_seller_items": np.minimum(n_items, rng.integers(1, 4, n)),
            "weight_g_sum": rng.uniform(100, 5000, n),
            "volume_cm3_sum": rng.uniform(200, 20000, n),
            "distance_km": distance,
            "missing_coords": 0,
            "product_category_name": rng.choice(["health_beauty", "toys", "computers", "bed_bath_table"], n),
            "seller_id": rng.choice(["s1", "s2", "s3"], n),
            "seller_state": rng.choice(["SP", "RJ", "MG"], n),
            "customer_state": rng.choice(["SP", "RJ", "MG"], n),
        })
        df["dias_entrega"] = 4 + distance / 150 + rng.normal(0, 1, n)
        return df

    return make


@pytest.fixture
def quantile_frame(frame_factory):
    """Devuelve una función `make(n, seed)` que arma una tabla sintética con ruido ASIMÉTRICO.

    El objetivo es `4 + distancia / 200 + ruido gamma` (cola larga a la derecha), así que el percentil 75 de los
    días queda claramente por encima de la mediana: sirve para comprobar que el plazo recomendado cubre cerca
    del 75 % de los pedidos. Incluye las variables derivadas de `add_features`.
    """
    from olistbr.features.engineering import add_features

    def make(n: int = 600, seed: int = 0) -> pd.DataFrame:
        df = frame_factory(n, seed)
        rng = np.random.default_rng(seed + 1000)
        df["dias_entrega"] = 4 + df["distance_km"] / 200 + rng.gamma(shape=2.0, scale=2.0, size=n)
        return add_features(df)

    return make
