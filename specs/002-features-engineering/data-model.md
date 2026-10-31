# Modelo de datos: variables de la etapa 4

## Entrada

`data/processed/train.parquet` (la deja el notebook 03): 76.924 filas, una por pedido, con estas columnas.

| Columna | Tipo | Uso en la etapa 4 |
|---|---|---|
| `order_id` | texto | identificador; no entra al modelo |
| `customer_unique_id` | texto | agrupa los pliegues; no entra al modelo |
| `dias_entrega` | número | **objetivo** |
| `order_purchase_timestamp` | fecha | base de las variables de calendario y de la prueba hacia adelante |
| `order_estimated_delivery_date` | fecha | solo para calcular `promised_days` (referencia) |
| `order_approved_at` | fecha | no entra al modelo (se mide aparte en otra etapa) |
| `price_sum`, `freight_sum`, `n_items`, `weight_g_sum`, `volume_cm3_sum`, `distance_km`, `missing_coords` | número | variables base |
| `product_category_name`, `seller_id`, `seller_state`, `customer_state` | texto | variables base (categóricas) |
| `main_seller_items` | entero | solo para la marca de varios vendedores |

## Variables derivadas (las crea `add_features`, sin aprender de los datos)

| Columna | Idea | Fórmula | Vacío cuando |
|---|---|---|---|
| `log_distance`, `log_freight`, `log_price`, `log_weight`, `log_volume` | F1 | `log1p(distance_km)`, `log1p(freight_sum)`, `log1p(price_sum)`, `log1p(weight_g_sum)`, `log1p(volume_cm3_sum)` | el dato original está vacío o es negativo |
| `same_state` | F2 | `1` si `customer_state == seller_state`, si no `0` | algún estado vacío |
| `customer_region`, `seller_region` | F3 | región según la tabla fija de `research.md` D3 | estado vacío o fuera de la tabla |
| `promised_days` | F4 (**solo referencia**) | `(order_estimated_delivery_date − order_purchase_timestamp)` en días | alguna fecha vacía |
| `purchase_dow` | F5 | día de la semana como texto (`"0"` lunes … `"6"` domingo) | fecha vacía |
| `is_slow_weekday` | F5 | `1` si es jueves, viernes o sábado | fecha vacía |
| `is_year_end_peak` | F6 | `1` entre el lunes de la semana de Black Friday y el 31 de diciembre | fecha vacía |
| `log_distance_x_same_state` | F7 | `log_distance × same_state` | alguno vacío |
| `route` | F8 | texto `seller_state→customer_state` (para el historial) | algún estado vacío |
| `freight_ratio` | F9 | `freight_sum / price_sum` | `price_sum` ≤ 0 o vacío |
| `density` | F9 | `weight_g_sum / volume_cm3_sum` | `volume_cm3_sum` ≤ 0 o vacío |
| `multi_seller` | F10 | `1` si `n_items > main_seller_items`, si no `0` | alguno vacío |

Reglas: división por cero o dato ausente da **vacío (NaN)**, nunca infinito; la tabla de entrada no se modifica.

## Grupos de ideas (los interruptores)

| Grupo | Idea | Numéricas | Categóricas | Con historial (`TargetEncoder`) |
|---|---|---|---|---|
| `logs` | F1 | `log_distance`, `log_freight`, `log_price`, `log_weight`, `log_volume` | | |
| `same_state` | F2 | `same_state` | | |
| `regions` | F3 | | `customer_region`, `seller_region` | |
| `calendar` | F5 | `is_slow_weekday` | `purchase_dow` | |
| `peak_season` | F6 | `is_year_end_peak` | | |
| `distance_interaction` | F7 | `log_distance`, `same_state`, `log_distance_x_same_state` | | |
| `history` | F8 | | | `seller_id`, `product_category_name`, `route` |
| `ratios` | F9 | `freight_ratio`, `density` | | |
| `multi_seller` | F10 | `multi_seller` | | |
| *(referencia, fuera de los grupos)* | F4 | `promised_days` | | |

Cuando se activan varios grupos, las columnas repetidas se cuentan una sola vez.

**Base (siempre presente):** numéricas `price_sum`, `freight_sum`, `n_items`, `weight_g_sum`, `volume_cm3_sum`, `distance_km`,
`missing_coords`; categóricas `product_category_name`, `seller_id`, `seller_state`, `customer_state`.

## Salidas

| Archivo | Contenido |
|---|---|
| `reports/04_features_tabla.csv` | una fila por medición (14): `idea`, `grupo`, `mae_media`, `mae_desv`, `mae_mejora`, `mae_ruido`, `mae_mejora_ruido`, `costo_media`, `costo_mejora`, `costo_mejora_ruido`, `adelante_mejora`, `adelante_mae` (error absoluto de la prueba hacia adelante), `decision`, `motivo` |
| `reports/04_features.md` | el texto con la decisión, la cifra y el motivo de cada idea, y las limitaciones |

## Reglas de validez (se prueban)

- Ningún cliente en la validación y el entrenamiento del mismo pliegue; los 5 pliegues cubren todas las filas una vez.
- La partición hacia adelante: entrenamiento con compras ≤ 2018-02-28, validación 2018-03-01 a 2018-08-31 y sin clientes compartidos.
- `promised_days` no está en las columnas de ninguna combinación de grupos.
- Las columnas de salida del preprocesador no tienen vacíos ni infinitos.
