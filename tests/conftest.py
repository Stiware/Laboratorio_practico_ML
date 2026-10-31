"""Permite ejecutar `pytest tests` desde la raíz del repo sin instalar el paquete."""
import sys
from pathlib import Path

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
