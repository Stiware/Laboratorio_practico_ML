# Contrato de las funciones de `src/olistbr/` (data, features y models)

Todas reciben y devuelven `pandas.DataFrame` (salvo donde se indica), no modifican su entrada (trabajan sobre una copia) y
llevan docstring en español. Los nombres son los que se probarán.

| Módulo | Función | Entrada → salida | Regla (EDA) | Prueba |
|---|---|---|---|---|
| `data/cleaning.py` | `load_raw(raw_dir)` | ruta → `dict[str, DataFrame]` con las tablas permitidas; fechas convertidas; nunca incluye reseñas | 10 | `test_load_raw_has_no_reviews` |
| `data/cleaning.py` | `drop_personal_data(df)` | tabla → tabla sin `review_comment_title`, `review_comment_message` ni identificadores de reseña (si existen) | 10 | `test_no_personal_columns` |
| `data/cleaning.py` | `build_target(orders)` | pedidos → solo `delivered` con fecha de entrega, con `dias_entrega` | 1 | `test_build_target` |
| `data/cleaning.py` | `filter_stable_period(df, start, end)` | tabla → compras entre `2017-01-01` y `2018-08-31`, ambos incluidos | 9 | `test_filter_stable_period` |
| `data/cleaning.py` | `drop_extreme_targets(df, max_days=100)` | tabla → `(kept, extreme)`; `extreme` tiene `dias_entrega > 100` | 2 | `test_drop_extreme_targets` |
| `data/cleaning.py` | `clean_products(products, translation)` | productos → categoría traducida o `"desconocida"`, peso 0 como vacío | 7 | `test_clean_products` |
| `data/cleaning.py` | `aggregate_items(items, products, sellers)` | ítems → una fila por `order_id` (precio, flete, conteo, peso, volumen, vendedor principal, `main_seller_items`, categoría) | 3 | `test_aggregate_items` |
| `data/cleaning.py` | `drop_leakage_columns(df)` | tabla → sin `order_delivered_*`, `order_status`, `review_*`, `customer_id`, ciudad ni códigos postales; conserva `order_approved_at` | 4 | `test_drop_leakage_columns` |
| `features/geo.py` | `zip_centroids(geolocation)` | geolocalización → un punto (lat, lng) por código postal; los puntos fuera de Brasil se reemplazan por el promedio del mismo código postal o, si no hay, por la mediana de la misma ciudad y estado | 6 | `test_zip_centroids` |
| `features/geo.py` | `haversine_km(lat1, lng1, lat2, lng2)` | arreglos → distancia en km | 6 | `test_haversine_km` |
| `features/geo.py` | `add_distance(df, centroids)` | tabla → `distance_km` y `missing_coords` | 6 | `test_add_distance` |
| `features/categories.py` | `frequent_values(df, column, weight_col, min_count=10)` | **solo train** → lista de valores con al menos `min_count` | 8 | `test_frequent_values` |
| `features/categories.py` | `group_rare_categories(df, column, frequent, other_label="otros")` | tabla → valores fuera de `frequent` pasan a `"otros"` | 8 | `test_group_rare_categories` |
| `data/split.py` | `split_by_group(df, group_col="customer_unique_id", test_size=0.2, seed=42)` | tabla → `(train, test)` sin cliente repetido | 5 | `test_split_by_group_no_overlap`, `test_row_split_overlaps` |
| `data/split.py` | `prepare_train_test(orders, ...)` | tabla → `(train, test, frequent_sellers)`: parte por cliente, calcula los vendedores frecuentes **solo con train** y los aplica a ambos. Añadida tras la revisión de código para que el orden viva en `src/` | 5 y 8 | `test_prepare_train_test_learns_sellers_only_from_train`, `test_prepare_train_test_matches_manual_steps` |
| `data/cleaning.py` | `build_orders_table(raw_dir, report=None)` | ruta → `(orders_table, extreme_orders)` aplicando las reglas 1 a 7 en el orden aprobado | todas | `test_build_orders_table_smoke` (con tablas mínimas) |

## Comportamientos que se prueban a mano (casos normal y sucio)

- `build_target`: un pedido `delivered` con fecha, uno `delivered` sin fecha (sale), uno `canceled` (sale).
- `filter_stable_period`: 2016-12-31 sale; 2017-01-01 y 2018-08-31 entran; 2018-09-01 sale.
- `drop_extreme_targets`: 100,0 se queda; 100,1 sale y aparece en `extreme`.
- `aggregate_items`: pedido con 3 ítems y 2 vendedores (suma, conteo y empate por menor `seller_id`); un ítem sin peso deja el peso del pedido vacío.
- `clean_products`: categoría vacía → `"desconocida"`; categoría sin traducción conserva el nombre original; peso 0 → vacío.
- `zip_centroids`: tres filas del mismo código postal dan un solo promedio; un punto fuera de Brasil no cuenta si el código postal tiene otra fila buena; si no la tiene, toma la mediana de la ciudad y estado (con nombres escritos distinto en mayúsculas o tildes); si tampoco hay ciudad con puntos buenos, queda vacío.
- `haversine_km`: distancia conocida entre dos ciudades dentro de una tolerancia (valor de referencia calculado a mano y escrito en la prueba).
- `add_distance`: código postal sin centroide → `distance_km` vacío y `missing_coords = 1`.
- `frequent_values` / `group_rare_categories`: vendedor con 9 ítems → `"otros"`, con 10 → se conserva; vendedor solo presente en test → `"otros"`.
- `split_by_group`: un cliente con 5 pedidos queda todo en un lado; misma semilla, misma partición; partir por fila a propósito produce solapamiento.

## Funciones añadidas después de la revisión

| Módulo | Función | Entrada → salida | Prueba |
|---|---|---|---|
| `data/split.py` | `assert_disjoint_groups(*frames, group_col)` | falla (`AssertionError`) si un cliente está en más de una partición; la usa `split_by_group` por sí misma | `test_assert_disjoint_groups_raises_on_overlap` |
| `models/metrics.py` | `delivery_cost(real, predicted, under_cost, over_cost)` | costo por pedido; subestimar cuesta 3 veces más (config) | `test_delivery_cost_penalizes_underestimating_three_times_more` |
| `models/metrics.py` | `mean_delivery_cost(real, predicted, ratio)` | costo medio; con `ratio=1` es el MAE | `test_mean_delivery_cost_with_ratio_one_is_mae` |
| `models/metrics.py` | `mae_days(real, predicted)` | error absoluto medio en días | `test_mae_days` |
| `models/metrics.py` | `best_fixed_deadline(real, ratio)` | mejor plazo fijo: percentil `ratio / (ratio + 1)` | `test_best_fixed_deadline_minimizes_the_mean_cost` |
