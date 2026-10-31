# Especificación · Etapa 3 · Preprocesamiento (limpieza y partición)

> Se llena **antes** de pedirle código a Claude, a partir de la tabla de decisiones de
> `especificaciones/02-eda.md`. Claude la lee y la implementa; ustedes revisan que el código haga
> **exactamente** esto (`material-del-curso/kit/checklist-revision-ia.md`).
> Guía: `material-del-curso/kit/guia-eda-y-preprocesamiento.md`, sección 4 · Ejemplo: `material-del-curso/ejemplo-creditsight/src/creditsight/data/`.

**Grupo:** _______________   **Fecha:** 31 oct 2026   **Estado:** ☐ borrador  ☒ cerrada _(el grupo aprobó todos los puntos ⚠ el 31 oct 2026)_

## 1. Unidad de análisis y partición

- **Una fila es:** un pedido entregado a un cliente (después de agregar sus ítems).
- **Entidad que se repite (columna):** `customer_unique_id` (2.997 personas con más de un pedido, máx. 17). ☐ no se repite
- **Se parte por:** ☒ grupo (`GroupShuffleSplit`, agrupando por `customer_unique_id`) ☐ fila
- **Proporción de prueba:** 20 %   **Semilla:** 42  ⚠ _(propuesta; el grupo confirma)_
- **¿Hay orden temporal que respetar?** ☒ sí → solo se usa el periodo 2017-01 a 2018-08 (decisión 9). ⚠ _Pregunta abierta:_ la decisión 9 dice también "evaluar por fecha (últimos meses como prueba)". Propuesta: en esta etapa la prueba es el 20 % de **clientes** (decisión 5); la revisión por fecha se hace en la etapa 7 sobre ese mismo conjunto de prueba, sin crear una segunda partición.
- **Dónde se guarda la prueba:** `data/processed/` (cerrada: **no se mira** hasta la etapa 7)

## 2. Datos personales

| Columna | Qué es | Se elimina en | Prueba que lo verifica |
|---|---|---|---|
| `review_comment_title` | Texto libre de la reseña | `load_raw()` (primer paso al cargar) | `test_no_personal_columns` |
| `review_comment_message` | Texto libre de la reseña | `load_raw()` (primer paso al cargar) | `test_no_personal_columns` |
| `review_id` y el resto de `review_*` | Reseña posterior a la compra (también es fuga) | `load_raw()`: no se carga la tabla de reseñas | `test_no_personal_columns` |

> `customer_unique_id` se conserva **solo** para agrupar la partición y se saca de las entradas del modelo. `order_id` y `customer_id` ya vienen con hash.

## 3. Reglas de limpieza

Una fila por decisión del EDA. **Una función, una prueba.** Se marca como vacío lo imposible; no se borran filas sin criterio.

| # (de la tabla del EDA) | Columna(s) | Problema | Regla exacta (con el umbral o valor) | Función en `src/olistbr/data/cleaning.py` | Prueba en `tests/test_cleaning.py` |
|---|---|---|---|---|---|
| 1 | `order_status`, `order_delivered_customer_date`, `order_purchase_timestamp` | 2.965 pedidos sin fecha de entrega | Conservar `order_status == "delivered"` con fecha de entrega no vacía; `dias_entrega` = (entrega − compra) en días. Quedan 96.470 | `build_target` | `test_build_target` |
| 2 | `dias_entrega` | 64 pedidos > 100 días | Excluir `dias_entrega > 100` del entrenamiento y guardarlos aparte para reportarlos | `drop_extreme_targets` | `test_drop_extreme_targets` |
| 3 | `order_items` (precio, flete, peso, volumen, vendedor) | Varios ítems/vendedores por pedido | Una fila por `order_id`: suma de `price` y `freight_value`, conteo de ítems, suma de peso y volumen, vendedor con más ítems (empate: el de menor `seller_id`) | `aggregate_items` | `test_aggregate_items` |
| 4 | `order_delivered_*`, `order_status`, `review_*`, `order_approved_at` | Se conocen después de comprar | Quitar de las entradas. `order_approved_at` se mide con y sin ella en la etapa 4, así que se conserva en una columna aparte y no entra al modelo por defecto | `drop_leakage_columns` | `test_drop_leakage_columns` |
| 5 | `customer_unique_id` | Cliente repetido | Partir por grupo; ningún cliente en dos particiones | `split_by_group` (en `split.py`) | `test_split_by_group` (en `tests/test_split.py`) |
| 6 | `geolocation_*`, códigos postales | 261.831 duplicados; 478 pedidos sin coordenadas | Promediar latitud/longitud por código postal; distancia vendedor–cliente en km (haversine); columna indicadora `missing_coords` si falta | `zip_centroids`, `haversine_km` | `test_zip_centroids`, `test_haversine_km` |
| 7 | `product_category_name`, `product_weight_g` | 610 sin categoría; 4 con peso 0; 13 sin traducción | Categoría vacía → `"desconocida"`; traducir al inglés; peso 0 → vacío (la imputación va en la etapa 4) | `clean_products` | `test_clean_products` |
| 8 | `customer_city`, zip del cliente, `seller_id` | Demasiadas categorías | Excluir ciudad y zip como variables; vendedores con < 10 ítems → `"otros"` ⚠ _(el conteo se calcula solo con train)_ | `group_rare_categories` | `test_group_rare_categories` |
| 9 | `order_purchase_timestamp` | 2016 y 2018-09/10 casi vacíos | Conservar compras entre 2017-01-01 y 2018-08-31 | `filter_stable_period` | `test_filter_stable_period` |
| 10 | Tablas de reseñas, ids | Datos personales y fuga | No cargar reseñas; ningún id como variable de entrada | `drop_personal_data` | `test_no_personal_columns` |

> ⚠ **Orden de los pasos** (propuesto): cargar → quitar datos personales → construir objetivo (1) → filtrar periodo (9) → quitar extremos (2) → agregar ítems (3) → productos (7) → distancia (6) → quitar fuga (4) → **partir (5)** → agrupar vendedores raros (8, ajustado solo con train).

## 4. Vacíos

| Columna | Qué significa el vacío | Tratamiento hoy | Tratamiento en la etapa 4 (dentro del `Pipeline`) |
|---|---|---|---|
| `order_delivered_customer_date` | ☒ no aplica: el pedido no llegó | Se excluye la fila (regla 1) | no aplica |
| `product_category_name` | ☒ no se anotó | ☒ categoría explícita `"desconocida"` | codificación (one-hot) |
| `product_weight_g` (0 o vacío) | ☒ imposible / no se anotó | ☒ dejar vacío | imputar con la mediana calculada **solo con train** |
| Coordenadas de cliente/vendedor | ☒ perdido (478 pedidos) | ☒ dejar vacío y crear `missing_coords` | imputar la distancia con la mediana de train |

> Imputar con medias o medianas **no se hace aquí**: depende del entrenamiento y va dentro del `Pipeline`.

## 5. Entradas y salidas

| | Ruta |
|---|---|
| Entrada | `data/raw/` (los CSV de Olist: pedidos, ítems, clientes, productos, vendedores, geolocalización y traducción de categorías; nunca se modifica) |
| Salida: entrenamiento | `data/processed/train.csv` (o `.parquet`) |
| Salida: prueba (cerrada) | `data/processed/test.csv` (o `.parquet`) |
| Código de limpieza | `src/olistbr/data/cleaning.py` |
| Código de partición | `src/olistbr/data/split.py` |
| Notebook que orquesta | `notebooks/03_preprocesamiento.ipynb` (solo llama a las funciones) |

> **Nota (2026-10-31):** el código se reorganizó siguiendo el ejemplo del curso sin cambiar ninguna regla: la carga, las reglas de limpieza y el encadenado están en `src/olistbr/data/cleaning.py`; la partición en `data/split.py`; la distancia (`zip_centroids`, `haversine_km`) y los vendedores raros (`group_rare_categories`) en `src/olistbr/features/`; las pruebas en `tests/test_cleaning.py`, `test_features.py` y `test_split.py`. Detalle en `reports/02_preprocesamiento.md`.

## 6. Pruebas que deben existir

- ☒ **Partición:** ninguna entidad aparece en dos particiones y no se pierde ninguna fila (`tests/test_split.py`).
- ☒ **La prueba de partición falla** si se cambia temporalmente a partición por fila (compruébenlo y vuelvan a dejarla bien).
- ☒ **Una prueba por regla de limpieza** de la sección 3, con un caso normal y un caso sucio hechos a mano.
- ☒ **Datos personales:** el resultado de cargar no contiene las columnas de la sección 2.
- ☒ `pytest tests` en verde desde la raíz del repo (62 pruebas, 2026-10-31).

## 7. Criterios de cierre (H1)

- ☒ La limpieza corre **de principio a fin desde cero** (`data/processed/` borrada y regenerada con el notebook 03), sin pasos manuales.
- ☒ Unidad de análisis y forma de partir **escritas** (sección 1) y documentadas en `reports/01_eda.md`.
- ☒ `pytest` en verde.
- ☒ Conjunto de prueba guardado y no usado para decidir nada (un revisor lo abrió solo para verificar que no hay clientes repetidos).
- ☒ `data/raw/` intacta y `.env` fuera de Git (`git status`).
- ☐ Todo subido a GitHub.
- ☐ Revisaron el código de Claude con `checklist-revision-ia.md` y cada integrante puede explicar una función.
- ☒ **Hallazgo que no esperaban** (para la puesta en común): el EDA hablaba de 2.965 pedidos sin fecha de entrega, pero el código quita 2.971: otros 6 son `canceled` y sí traen fecha de entrega. Además, solo 5 de los 21 códigos postales con coordenadas erróneas necesitaron la aproximación por ciudad y estado.
