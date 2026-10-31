# Modelo de datos: Preprocesamiento de OnTrack

## Entradas (`data/raw/`, solo lectura)

| Archivo | Clave | Columnas que se usan |
|---|---|---|
| `olist_orders_dataset.csv` | `order_id` | `customer_id`, `order_status`, `order_purchase_timestamp`, `order_approved_at`, `order_delivered_customer_date`, `order_estimated_delivery_date` |
| `olist_order_items_dataset.csv` | `order_id` + `order_item_id` | `product_id`, `seller_id`, `price`, `freight_value` |
| `olist_customers_dataset.csv` | `customer_id` | `customer_unique_id`, `customer_zip_code_prefix`, `customer_state` |
| `olist_products_dataset.csv` | `product_id` | `product_category_name`, `product_weight_g`, `product_length_cm`, `product_height_cm`, `product_width_cm` |
| `olist_sellers_dataset.csv` | `seller_id` | `seller_zip_code_prefix`, `seller_state` |
| `olist_geolocation_dataset.csv` | (sin clave) | `geolocation_zip_code_prefix`, `geolocation_lat`, `geolocation_lng`, `geolocation_city`, `geolocation_state` (estas dos solo para corregir puntos erróneos; no son entradas del modelo) |
| `product_category_name_translation.csv` | `product_category_name` | `product_category_name_english` |

**No se cargan:** `olist_order_reviews_dataset.csv` (reseñas: datos personales y fuga) ni `olist_order_payments_dataset.csv`
(no lo usa ninguna regla aprobada).

## Salida: una fila por pedido

| Columna | Tipo | Origen | Uso |
|---|---|---|---|
| `order_id` | texto | orders | Identificador; no es entrada del modelo |
| `customer_unique_id` | texto | customers | Agrupa la partición; no es entrada del modelo |
| `dias_entrega` | número | entrega − compra, en días | **Objetivo** |
| `order_purchase_timestamp` | fecha | orders | Entrada posible (mes/estacionalidad) y base del filtro de periodo |
| `order_estimated_delivery_date` | fecha | orders | Entrada posible (se conoce al comprar) |
| `order_approved_at` | fecha | orders | Aparte; se mide con y sin ella en la etapa 4 |
| `price_sum` | número | suma de `price` | Entrada |
| `freight_sum` | número | suma de `freight_value` | Entrada |
| `n_items` | entero | conteo de ítems | Entrada |
| `weight_g_sum` | número | suma de peso (vacío si falta alguno) | Entrada; se imputa en la etapa 4 |
| `volume_cm3_sum` | número | suma de largo×alto×ancho | Entrada; se imputa en la etapa 4 |
| `product_category_name` | texto | traducida; `"desconocida"` si falta | Entrada |
| `seller_id` | texto | vendedor principal; `"otros"` si es raro | Entrada |
| `main_seller_items` | entero | ítems del vendedor principal | Solo para contar vendedores raros |
| `seller_state` | texto | sellers | Entrada |
| `customer_state` | texto | customers | Entrada y segmento para revisar sesgos |
| `distance_km` | número | haversine | Entrada; vacía si faltan coordenadas |
| `missing_coords` | 0/1 | indicador | Entrada |

`train.parquet` y `test.parquet` tienen estas mismas columnas. `extreme_orders.parquet` tiene las mismas columnas de los pedidos
excluidos por `dias_entrega > 100`.

## Reglas de validez (se prueban)

- `order_id` es único en la salida.
- `0 ≤ dias_entrega ≤ 100`.
- `2017-01-01 ≤ order_purchase_timestamp ≤ 2018-08-31`.
- Ningún `customer_unique_id` está a la vez en `train` y `test`; `len(train) + len(test)` = filas de la tabla limpia.
- No existe ninguna columna `order_delivered_*`, `order_status`, `review_*`, `customer_id`, ciudad ni código postal.

## Relaciones

- Un cliente (`customer_unique_id`) tiene uno o varios `customer_id` (uno por pedido) y cada `customer_id` tiene un pedido.
- Un pedido tiene uno o varios ítems; cada ítem tiene un producto y un vendedor.
- El código postal une cliente y vendedor con la tabla de geolocalización (varias filas por código postal).
