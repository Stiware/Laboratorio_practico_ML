---

description: "Lista de tareas de la etapa 3 · Preprocesamiento de OnTrack"
---

# Tasks: Preprocesamiento de datos de OnTrack (etapa 3)

**Input**: documentos de diseño en `specs/001-preprocesamiento-datos/`

**Prerequisites**: `plan.md` (aprobado), `spec.md`, `research.md`, `data-model.md`, `contracts/data-functions.md`

**Tests**: SÍ se piden. Lo exigen la constitución (principio IV) y `CLAUDE.md`: cada función de `src/` que limpia o transforma tiene su prueba. Las pruebas usan tablas pequeñas hechas a mano y **no leen `data/raw/`**.

**Organization**: por historia de usuario del `spec.md`.

## Format: `[ID] [P?] [Story] Descripción con ruta`

- **[P]**: se puede hacer en paralelo (archivos distintos, sin depender de una tarea sin terminar).
- **[Story]**: US1 a US4, la historia del `spec.md`.
- **Regla del proyecto:** una tarea se marca `[x]` **solo cuando su prueba pasa** (`pytest tests`). Cada código nuevo lleva docstring en español; nombres de código en inglés.
- **Comando de prueba:** `source .venv/bin/activate && pytest tests` desde la raíz del repo.

## Path Conventions

Proyecto único: `src/olistbr/data/`, `tests/`, `notebooks/`, `reports/`, `data/processed/` (rutas del `plan.md`).

---

## Phase 1: Setup

**Propósito**: dejar el terreno listo sin escribir lógica.

- [x] T001 Verificar que `pyarrow==25.0.1` está en `requirements.txt` y que `.venv/bin/python -c "import pyarrow, pandas"` funciona; `data/processed/` existe y `data/raw/` está en `.gitignore`
- [x] T002 [P] Crear los módulos vacíos con docstring de una línea en español: `src/olistbr/data/loading.py`, `cleaning.py`, `geo.py`, `categories.py`, `split.py`, `pipeline.py`
- [x] T003 [P] Crear los archivos de prueba vacíos: `tests/test_loading.py`, `tests/test_cleaning.py`, `tests/test_geo.py`, `tests/test_categories.py`, `tests/test_split.py`, `tests/test_pipeline.py`

---

## Phase 2: Foundational (bloquea las historias)

**Propósito**: carga segura de datos y constantes comunes.

- [x] T004 Definir en `src/olistbr/data/__init__.py` las constantes de las reglas aprobadas: `PERIOD_START="2017-01-01"`, `PERIOD_END="2018-08-31"`, `MAX_DELIVERY_DAYS=100`, `TEST_SIZE=0.2`, `SEED=42`, `MIN_SELLER_ITEMS=10`, `OTHER_LABEL="otros"`, `UNKNOWN_CATEGORY="desconocida"`
- [x] T005 [P] Implementar `load_raw(raw_dir)` en `src/olistbr/data/loading.py`: lee solo las 7 tablas permitidas (nunca `order_reviews` ni `order_payments`), convierte las fechas a fecha, no escribe nada en `data/raw/`; contrato en `contracts/data-functions.md`
- [x] T006 [P] Implementar `drop_personal_data(df)` en `src/olistbr/data/loading.py`: quita `review_comment_title`, `review_comment_message` y los `review_*` si existen; trabaja sobre una copia
- [x] T007 [P] Pruebas `test_load_raw_has_no_reviews` (con una carpeta temporal de CSV mínimos), `test_no_personal_columns` y `test_load_raw_does_not_modify_raw` (tamaño y fecha de los archivos iguales antes y después) en `tests/test_loading.py`
- [x] T008 Ejecutar `pytest tests/test_loading.py` y marcar T005 a T007 solo si pasan

**Checkpoint**: se pueden cargar los datos sin datos personales y sin tocar `data/raw/`.

---

## Phase 3: User Story 1 - Tabla limpia de pedidos con su objetivo (Prioridad: P1) 🎯 MVP

**Meta**: una fila por pedido entregado, con `dias_entrega`, sin fuga ni datos personales.

**Prueba independiente**: ejecutar `build_orders_table` y comprobar `order_id` único, `0 ≤ dias_entrega ≤ 100`, compras entre 2017-01-01 y 2018-08-31 y ninguna columna `order_delivered_*`, `order_status` ni `review_*`.

### Pruebas (primero, deben fallar antes de implementar)

- [x] T009 [P] [US1] Pruebas `test_build_target` (entregado con fecha entra; entregado sin fecha sale; cancelado sale), `test_filter_stable_period` (2016-12-31 sale, 2017-01-01 y 2018-08-31 entran, 2018-09-01 sale) y `test_drop_extreme_targets` (100,0 se queda; 100,1 sale y aparece en `extreme`) en `tests/test_cleaning.py`
- [x] T010 [P] [US1] Pruebas `test_clean_products` (categoría vacía → `"desconocida"`; sin traducción conserva el nombre original; peso 0 → vacío) y `test_aggregate_items` (3 ítems y 2 vendedores: sumas, conteo, vendedor principal con desempate por menor `seller_id`, `main_seller_items`; un ítem sin peso deja `weight_g_sum` vacío) en `tests/test_cleaning.py`
- [x] T011 [P] [US1] Pruebas `test_drop_leakage_columns` (quita `order_delivered_*`, `order_status`, `review_*`, `customer_id`, ciudad y códigos postales; conserva `order_approved_at`) en `tests/test_cleaning.py`

### Implementación

- [x] T012 [US1] `build_target(orders)` en `src/olistbr/data/cleaning.py`: solo `order_status == "delivered"` con `order_delivered_customer_date` no vacía; `dias_entrega` = (entrega − compra) en días
- [x] T013 [US1] `filter_stable_period(df, start, end)` en `src/olistbr/data/cleaning.py`: compra entre `PERIOD_START` y `PERIOD_END`, ambos incluidos
- [x] T014 [US1] `drop_extreme_targets(df, max_days=100)` en `src/olistbr/data/cleaning.py`: devuelve `(kept, extreme)` con `extreme` donde `dias_entrega > 100`
- [x] T015 [US1] `clean_products(products, translation)` en `src/olistbr/data/cleaning.py`: categoría vacía → `"desconocida"`, traducción al inglés, peso 0 → vacío (sin imputar)
- [x] T016 [US1] `aggregate_items(items, products, sellers)` en `src/olistbr/data/cleaning.py`: una fila por `order_id`; `price_sum`, `freight_sum`, `n_items`, `weight_g_sum` y `volume_cm3_sum` (largo×alto×ancho por ítem, luego suma) con vacío si **algún** ítem no tiene dato; vendedor principal (más ítems; empate: menor `seller_id`), `main_seller_items`, `seller_state` y la categoría del producto del vendedor principal (empate: menor `order_item_id`)
- [x] T017 [US1] `drop_leakage_columns(df)` en `src/olistbr/data/cleaning.py`: quita las columnas de fuga e identificadores de la lista de `research.md` D6; deja `order_id`, `customer_unique_id` y `order_approved_at`
- [x] T018 [US1] `build_orders_table(raw_dir)` en `src/olistbr/data/pipeline.py`: encadena, en el orden aprobado, carga → objetivo → periodo → extremos → ítems/productos → quitar fuga (la distancia se agrega en US3); devuelve `(orders_table, extreme_orders)`
- [x] T019 [US1] Prueba `test_build_orders_table_smoke` en `tests/test_pipeline.py`, con CSV mínimos en una carpeta temporal: `order_id` único, `dias_entrega` entre 0 y 100, periodo correcto, sin columnas de fuga
- [x] T020 [US1] Ejecutar `pytest tests/test_loading.py tests/test_cleaning.py tests/test_pipeline.py` y marcar T009 a T019 solo si pasan

**Checkpoint**: la historia 1 funciona sola; ya existe la tabla limpia (sin distancia).

---

## Phase 4: User Story 2 - Partición sin fuga (Prioridad: P1)

**Meta**: 80 % entrenamiento y 20 % prueba, partidos por `customer_unique_id`.

**Prueba independiente**: ningún cliente en ambos conjuntos, filas suman el total y la misma semilla da la misma partición.

- [x] T021 [P] [US2] Pruebas en `tests/test_split.py`: `test_split_by_group_no_overlap` (un cliente con 5 pedidos va todo a un lado; cero solapamiento; `len(train)+len(test)` = filas totales), `test_split_is_reproducible` (misma semilla, mismo resultado) y `test_row_split_overlaps` (partir por fila a propósito **sí** produce solapamiento: demuestra que la prueba sirve)
- [x] T022 [US2] `split_by_group(df, group_col="customer_unique_id", test_size=0.2, seed=42)` en `src/olistbr/data/split.py` con `GroupShuffleSplit(n_splits=1)`; devuelve `(train, test)` con el índice reiniciado
- [x] T023 [US2] Ejecutar `pytest tests/test_split.py` y marcar T021 y T022 solo si pasan

**Checkpoint**: las historias 1 y 2 juntas producen entrenamiento y prueba sin fuga por cliente.

---

## Phase 5: User Story 3 - Distancia, productos y vendedores raros (Prioridad: P2)

**Meta**: agregar la distancia vendedor–cliente y agrupar vendedores raros usando solo el entrenamiento.

**Prueba independiente**: con casos hechos a mano, cada regla da el valor esperado.

### Pruebas

- [x] T024 [P] [US3] Pruebas en `tests/test_geo.py`: `test_zip_centroids` (tres filas del mismo código postal dan un promedio; un punto fuera de Brasil se reemplaza por el promedio de las filas buenas del mismo código postal; si no hay, por la mediana de la misma ciudad y estado aunque se escriban distinto en mayúsculas o tildes; si tampoco hay, queda vacío; los puntos buenos no cambian), `test_haversine_km` (distancia conocida con tolerancia; el valor de referencia se escribe en la prueba) y `test_add_distance` (código postal sin punto → `distance_km` vacío y `missing_coords = 1`)
- [x] T025 [P] [US3] Pruebas en `tests/test_categories.py`: `test_frequent_values` (9 ítems no entra, 10 sí; usa `main_seller_items` como peso) y `test_group_rare_categories` (raro → `"otros"`; un vendedor que solo está en prueba → `"otros"`; la lista de train no se recalcula con test)

### Implementación

- [x] T026 [US3] `zip_centroids(geolocation)` en `src/olistbr/data/geo.py`: recuadro de Brasil (latitud −34 a 6, longitud −74 a −34); cascada del `research.md` D4 (código postal → ciudad y estado → vacío) y salida con una vía anotada por código postal para poder contar cuántos se corrigieron por cada una
- [x] T027 [P] [US3] `haversine_km(lat1, lng1, lat2, lng2)` en `src/olistbr/data/geo.py` con numpy, radio 6371 km
- [x] T028 [US3] `add_distance(df, centroids)` en `src/olistbr/data/geo.py`: usa el código postal del cliente (de ese pedido) y del vendedor principal; crea `distance_km` y `missing_coords`; no imputa
- [x] T029 [P] [US3] `frequent_values(df, column, weight_col, min_count=10)` y `group_rare_categories(df, column, frequent, other_label="otros")` en `src/olistbr/data/categories.py`
- [x] T030 [US3] Integrar `add_distance` en `build_orders_table` (`src/olistbr/data/pipeline.py`, antes de `drop_leakage_columns`) y ampliar `test_build_orders_table_smoke` para verificar `distance_km` y `missing_coords`
- [x] T031 [US3] Ejecutar `pytest tests` completo y marcar T024 a T030 solo si pasan

**Checkpoint**: la tabla tiene la distancia y existe la función que agrupa vendedores raros.

---

## Phase 6: User Story 4 - Notebook y constancia de decisiones (Prioridad: P3)

**Meta**: regenerar `data/processed/` desde cero con el notebook y documentar cada decisión.

**Prueba independiente**: borrar `data/processed/`, ejecutar el notebook y obtener los tres archivos Parquet.

- [x] T032 [US4] Completar `notebooks/03_preprocesamiento.ipynb` (ya existe como plantilla del curso): solo importa de `olistbr.data`; llama a `build_orders_table`, `split_by_group`, `frequent_values` (con train) y `group_rare_categories` (en train y test); muestra filas antes y después de cada regla; guarda `train.parquet`, `test.parquet` y `extreme_orders.parquet` en `data/processed/`; **no lee `test` para decidir nada**; comprueba en celdas los criterios SC-002 a SC-005
- [x] T033 [US4] Ejecutar el notebook desde cero (`rm -f data/processed/*.parquet` y luego `jupyter nbconvert --to notebook --execute notebooks/03_preprocesamiento.ipynb --output 03_preprocesamiento.ipynb`); confirmar que termina sin errores y que una segunda ejecución da archivos idénticos
- [x] T034 [US4] Escribir `reports/02_preprocesamiento.md`: por cada regla, filas antes y después, la cifra real producida por el código (comparada con la del EDA, sin forzarla), motivo, cuántos códigos postales se corrigieron por cada vía de coordenadas, cuántos pedidos tienen `missing_coords` y peso vacío, y el efecto de quitar extremos también en la prueba
- [x] T035 [P] [US4] Actualizar `reports/bitacora_decisiones.md` con las cifras reales de la etapa y `especificaciones/03-preprocesamiento.md` (sección 6 de pruebas y sección 7 de cierre marcadas solo con lo que realmente se verificó)

**Checkpoint**: las cuatro historias funcionan; la etapa se puede regenerar de principio a fin.

---

## Phase 7: Polish y revisión

- [x] T036 [P] Revisión de fuga con el subagente `auditor-de-fuga` sobre `src/olistbr/data/` y `notebooks/03_preprocesamiento.ipynb`; corregir lo que encuentre o anotar por qué no aplica
- [x] T037 [P] Revisión con el subagente `revisor-de-codigo-ia` (checklist del curso) para que cada integrante pueda explicar una función; dejar el informe en `reports/02_preprocesamiento.md`
- [x] T038 Verificar `git status`: sin cambios en `data/raw/`, sin `.env`, sin datos personales; `pytest tests` en verde; probar `test_row_split_overlaps` con partición por fila y confirmar que la prueba de no solapamiento falla, y volver a dejarla bien
- [x] T039 Cerrar la etapa: marcar los criterios de la sección 7 de `especificaciones/03-preprocesamiento.md` y anotar el hallazgo que no esperaban para la puesta en común

---

## Dependencias y orden

- **Fase 1** → **Fase 2** → historias. Fase 2 bloquea todo.
- **US1 (P1)**: arranca tras la Fase 2; es el MVP.
- **US2 (P1)**: depende de que exista una tabla con `customer_unique_id` (la prueba usa datos de mano, así que se puede escribir en paralelo con US1; el uso real necesita T018).
- **US3 (P2)**: las funciones (T026 a T029) son independientes de US1; T030 integra en el pipeline y necesita T018.
- **US4 (P3)**: necesita US1, US2 y US3 terminadas.
- **Polish**: después de US4.

## Oportunidades en paralelo

- T002 y T003 juntos; T005, T006 y T007 juntos.
- Las pruebas T009, T010 y T011 (mismo archivo `tests/test_cleaning.py`) se escriben en una sola sesión, no en paralelo.
- T024 y T025 juntos; T027 y T029 son independientes.
- T036 y T037 juntos.

## Estrategia de implementación

1. **MVP**: Fases 1 y 2 + US1 (T001 a T020). Ya se tiene la tabla limpia.
2. Sumar US2 (partición) y comprobar cero solapamiento.
3. Sumar US3 (distancia y vendedores raros).
4. Cerrar con US4 (notebook, reporte y bitácora) y la revisión de la Fase 7.
5. Parar y mostrar al grupo el resultado de cada checkpoint antes de seguir; no se hace commit sin que el grupo lo pida.

---

## Cambios posteriores a la revisión (2026-10-31)

- [x] T040 [US2] Añadida `prepare_train_test` en `src/olistbr/data/split.py` (parte por cliente y agrupa vendedores raros solo con train) con dos pruebas en `tests/test_split.py`; el notebook la llama. Surgió del hallazgo 3 del `revisor-de-codigo-ia`.
- [x] T041 Arreglos menores de la revisión: contraste de partición sobre la función real, casos borde en `tests/test_geo.py` y `tests/test_cleaning.py`, `np.clip` en `haversine_km`, deduplicar la traducción en `clean_products`, textos del notebook sin cifras del EDA.

**Pendiente solo del grupo** (no lo puede cerrar Claude): subir todo a GitHub; que cada integrante pueda explicar una función (preguntas en `reports/02_preprocesamiento.md`); decidir si `data/processed/*.parquet` se ignora o se versiona.

## Reorganización del código siguiendo el ejemplo del curso (2026-10-31)

- [x] T042 Modularizar como `material-del-curso/ejemplo-creditsight/`: `src/olistbr/config.py` central (rutas y constantes; `data/__init__.py` queda vacío), `load_raw` y el encadenado dentro de `data/cleaning.py`, `assert_disjoint_groups` en `data/split.py`, distancia y categorías raras en `features/`, costo del error en `models/metrics.py` (antes vivía dentro del notebook 01), pruebas por módulo (`test_cleaning`, `test_features`, `test_split`, `test_metrics`). Sin cambiar resultados: Parquet y salidas de los notebooks 01 a 03 idénticos.
- Los nombres de archivos de las tareas T002 a T030 describen la distribución original; la estructura vigente está en `plan.md`.
