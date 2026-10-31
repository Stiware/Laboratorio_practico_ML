---

description: "Lista de tareas de la etapa 4 · Variables de OnTrack"
---

# Tasks: Variables nuevas de OnTrack y su medición (etapa 4)

**Input**: documentos de diseño en `specs/002-features-engineering/`

**Prerequisites**: `plan.md` (aprobado el 2026-10-31), `spec.md`, `research.md`, `data-model.md`, `contracts/features-functions.md`

**Tests**: SÍ se piden (constitución, principio IV, y `CLAUDE.md`): cada función de `src/` que transforma variables tiene su prueba. Las pruebas usan tablas pequeñas hechas a mano y **no leen `data/processed/`**. Primero se escribe la prueba y se ve que falla; después la función.

**Organization**: por historia de usuario del `spec.md`. **Orden de las fases:** las dos historias P1 están en el orden en que dependen una de otra: US2 (crear las variables) va antes que US1 (medirlas), porque no se puede medir lo que no existe; después US3, US4 y US5.

## Format: `[ID] [P?] [Story] Descripción con ruta`

- **[P]**: se puede hacer en paralelo (archivos distintos, sin depender de una tarea sin terminar).
- **[Story]**: US1 a US5, la historia del `spec.md`.
- **Regla del proyecto:** una tarea se marca `[x]` **solo cuando su prueba pasa**. Código nuevo con docstring en español; nombres de código en inglés.
- **Comando de prueba:** `.venv/bin/python -m pytest tests -q` desde la raíz del repo.
- **Prohibido en toda la etapa:** abrir `data/processed/test.parquet`; incluir `promised_days` en el modelo final; usar `git add .` sin revisar `git status`.

## Path Conventions

Proyecto único: `src/olistbr/`, `tests/`, `notebooks/`, `reports/` (rutas del `plan.md`).

---

## Phase 1: Setup

**Propósito**: dejar el terreno listo sin escribir lógica.

- [x] T001 Verificar el punto de partida: `data/processed/train.parquet` existe (si no, ejecutar el notebook 03), `.venv/bin/python -c "from sklearn.preprocessing import TargetEncoder; from sklearn.model_selection import GroupKFold"` funciona y `pytest tests -q` da 62 pruebas en verde
- [x] T002 [P] Crear los módulos con docstring de una línea en español: `src/olistbr/features/engineering.py`, `src/olistbr/models/train.py`, `src/olistbr/models/evaluation.py`
- [x] T003 [P] Crear los archivos de prueba nuevos con docstring: `tests/test_models.py`, `tests/test_evaluation.py` (`tests/test_features.py` y `tests/test_split.py` ya existen y se amplían)
- [x] T004 Agregar a `src/olistbr/config.py` las constantes de la etapa: `N_FOLDS = 5`, `FORWARD_TRAIN_END = "2018-02-28"`, `FORWARD_VALID_START = "2018-03-01"`, `FORWARD_VALID_END = "2018-08-31"`, `MIN_CATEGORY_FREQUENCY = 0.01`, `MIN_IMPROVEMENT_RATIO = 1.0`, cada una con su comentario del motivo

---

## Phase 2: Foundational (bloquea las historias)

**Propósito**: datos de ejemplo comunes para todas las pruebas.

- [x] T005 Agregar a `tests/conftest.py` una fixture `features_frame` con una tabla pequeña hecha a mano con las columnas de `data-model.md` (entrada): `order_id`, `customer_unique_id`, `dias_entrega`, `order_purchase_timestamp`, `order_estimated_delivery_date`, `price_sum`, `freight_sum`, `n_items`, `weight_g_sum`, `volume_cm3_sum`, `distance_km`, `missing_coords`, `product_category_name`, `seller_id`, `main_seller_items`, `seller_state`, `customer_state`; incluir filas normales y sucias (precio 0, flete vacío, estado fuera de la tabla, fecha en domingo, compra el 2017-11-20)
- [x] T006 Ejecutar `pytest tests -q` y confirmar que las 62 pruebas anteriores siguen en verde con la fixture nueva

**Checkpoint**: se puede empezar a escribir pruebas de variables.

---

## Phase 3: User Story 2 - Crear las variables aprobadas sin filtrar información (Prioridad: P1) 🎯 MVP

**Meta**: cada variable aprobada se calcula con una función pequeña que solo usa lo que se conoce al comprar.

**Prueba independiente**: con la fixture, cada variable da el valor esperado; división por cero y vacíos dan NaN (nunca `inf`); la tabla de entrada no se modifica.

### Pruebas (primero; deben fallar antes de implementar)

- [x] T007 [P] [US2] En `tests/test_features.py`, pruebas de `add_features` con los casos de `contracts/features-functions.md`: `test_log_features` (`log1p(0) = 0`; vacío o negativo → NaN), `test_same_state` (`SP/SP → 1`, `SP/RJ → 0`, vacío → NaN), `test_regions_table` (`SP → Sudeste`, `AM → Norte`, `"XX" → NaN`) y `test_region_table_covers_27_states`, `test_calendar_features` (lunes → 0 sin marca; jueves, viernes y sábado marcados; domingo sin marca; `purchase_dow` como texto `"0"` a `"6"`), `test_year_end_peak_window` (`2017-11-19 → 0`, `2017-11-20 → 1`, `2017-12-31 → 1`, `2018-01-01 → 0`) y `test_black_friday_dates` (`black_friday(2017) == 2017-11-24`), `test_distance_interaction`, `test_route` (texto `seller_state→customer_state`), `test_ratios_and_density` (precio 0 o vacío → NaN), `test_multi_seller` (`n_items=3, main_seller_items=2 → 1`; `3 y 3 → 0`)
- [x] T008 [P] [US2] En `tests/test_features.py`, `test_division_by_zero_or_missing_gives_nan_not_inf` (ninguna columna numérica derivada contiene `inf`) y `test_add_features_does_not_mutate_input`

### Implementación

- [x] T009 [US2] En `src/olistbr/features/engineering.py`, definir las constantes `BASE_NUMERIC` (`price_sum`, `freight_sum`, `n_items`, `weight_g_sum`, `volume_cm3_sum`, `distance_km`, `missing_coords`), `BASE_CATEGORICAL` (`product_category_name`, `seller_id`, `seller_state`, `customer_state`), `REGION_BY_STATE` con los 27 estados (Norte: AC, AP, AM, PA, RO, RR, TO; Nordeste: AL, BA, CE, MA, PB, PE, PI, RN, SE; Centro-Oeste: DF, GO, MT, MS; Sudeste: ES, MG, RJ, SP; Sur: PR, RS, SC) y `REFERENCE_ONLY_COLUMNS = ["promised_days"]`
- [x] T010 [US2] En `src/olistbr/features/engineering.py`, `black_friday(year)` (cuarto viernes de noviembre) y `add_features(df)`: crea las columnas de `data-model.md` §«Variables derivadas»; **«división por cero o dato ausente da vacío (NaN), nunca infinito; la tabla de entrada no se modifica»**; logaritmo con `log1p` y NaN si el dato es negativo; `purchase_dow` como texto; `is_year_end_peak` entre el lunes de la semana de Black Friday y el 31 de diciembre de ese año (calendario fijo, no aprendido); un estado fuera de `REGION_BY_STATE` da NaN; `promised_days` en días con las dos fechas
- [x] T011 [US2] Ejecutar `pytest tests/test_features.py -q` y marcar T005 a T010 solo si pasan

**Checkpoint**: las variables se calculan y están probadas; ya se puede medir.

---

## Phase 4: User Story 3 - Preparar los datos para el modelo sin fuga (Prioridad: P2)

**Meta**: todo lo que depende de los datos se ajusta solo con el entrenamiento de cada pliegue.

**Prueba independiente**: el preparador se ajusta con un conjunto, transforma otro con categorías nuevas y devuelve una matriz sin vacíos; el promedio histórico de una fila de validación no usa su propio objetivo.

### Pruebas

- [x] T012 [P] [US3] En `tests/test_features.py`: `test_feature_columns_toggles` (cada grupo agrega sus columnas y sin repetidas cuando se activan varios), `test_unknown_group_raises`, `test_promised_days_never_in_any_group_combination` (ninguna combinación de los 9 grupos contiene `promised_days`; solo aparece con `include_reference=True`)
- [x] T013 [P] [US3] En `tests/test_features.py`: `test_preprocessor_handles_unseen_category_and_missing` (vendedor y ruta nuevos y datos vacíos → matriz sin NaN), `test_preprocessor_output_has_no_missing_or_infinite`, `test_history_group_uses_target_encoder` (el grupo `history` agrega columnas; sin él no hay `TargetEncoder`), `test_history_does_not_use_own_target` (a una fila de entrenamiento le corresponde el promedio de las otras filas de su vendedor, no su propio valor)
- [x] T014 [P] [US3] En `tests/test_models.py`: `test_pipeline_fits_and_predicts` y `test_pipeline_is_fit_only_on_train` (las medianas del imputador salen del entrenamiento: se ajusta con una tabla y se comprueba que no cambian al transformar otra)

### Implementación

- [x] T015 [US3] En `src/olistbr/features/engineering.py`, `GROUPS` (9 grupos de `data-model.md` §«Grupos de ideas») y `feature_columns(groups=(), include_reference=False)`: devuelve numéricas, categóricas y con historial sin repetir; un grupo desconocido lanza `ValueError`; `promised_days` solo con `include_reference=True`
- [x] T016 [US3] En `src/olistbr/features/engineering.py`, `build_preprocessor(groups=(), include_reference=False, scale=True)`: numéricas con `SimpleImputer(strategy="median", add_indicator=True)` y `StandardScaler`; categóricas con `SimpleImputer(strategy="constant", fill_value="Unknown")` y `OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01, sparse_output=False)`; historial con `TargetEncoder(target_type="continuous", cv=KFold(5, shuffle=True, random_state=42))` (el `shuffle` y `random_state` propios de `TargetEncoder` están obsoletos en scikit-learn 1.9); `remainder="drop"`
- [x] T017 [US3] En `src/olistbr/models/train.py`, `build_pipeline(groups=(), include_reference=False)`: `Pipeline` con los pasos `prep` y `model` (`LinearRegression`)
- [x] T018 [US3] Ejecutar `pytest tests/test_features.py tests/test_models.py -q` y marcar T012 a T017 solo si pasan

**Checkpoint**: existe un pipeline completo que se ajusta solo con entrenamiento.

---

## Phase 5: User Story 1 - Saber qué ideas sirven y cuáles no (Prioridad: P1)

**Meta**: una tabla que compara cada idea contra la referencia con la misma validación y la regla de decisión escrita.

**Prueba independiente**: con tablas pequeñas, se calculan los pliegues, la mejora, el ruido y la decisión según los casos del contrato.

### Pruebas

- [x] T019 [P] [US1] En `tests/test_split.py`: `test_group_folds_do_not_share_customers`, `test_group_folds_cover_each_row_once` (los 5 pliegues de validación suman todas las filas sin repetir) y `test_group_folds_are_reproducible` (misma semilla, mismos pliegues)
- [x] T020 [P] [US1] En `tests/test_evaluation.py`: `test_evaluate_folds_returns_one_row_per_fold` (columnas `mae`, `cost`, `n_train`, `n_valid`), `test_evaluate_folds_does_not_leak_validation_into_fit` (el pipeline se clona y se ajusta solo con el entrenamiento de cada pliegue), `test_improvement_and_noise_are_paired_differences` (con números de mano: mejora = media de `referencia − idea`, ruido = desviación con `ddof=1`), `test_zero_noise_is_not_evaluable`
- [x] T021 [P] [US1] En `tests/test_evaluation.py`, los cinco casos de decisión del contrato: `test_decide_keeps_idea_that_beats_noise`, `test_decide_discards_when_ratio_below_one` (0,8 → descartar), `test_decide_discards_when_forward_flips_sign`, `test_decide_discards_when_not_known_at_prediction`, `test_decide_keeps_when_forward_not_evaluable` (motivo «no evaluable hacia adelante»)

### Implementación

- [x] T022 [US1] En `src/olistbr/data/split.py`, `make_group_folds(df, n_splits=N_FOLDS, group_col=GROUP_COL, seed=RANDOM_STATE)`: `GroupKFold(shuffle=True, random_state=seed)`; devuelve la lista de `(índices_entrenamiento, índices_validación)` y verifica con `assert_disjoint_groups` que ningún cliente cae en los dos lados de un pliegue
- [x] T023 [US1] En `src/olistbr/models/evaluation.py`, `evaluate_folds(pipeline, X, y, folds)`: clona el pipeline en cada pliegue, lo ajusta con el entrenamiento y devuelve una fila por pliegue con `mae` (`mae_days`) y `cost` (`mean_delivery_cost`) de `olistbr.models.metrics`
- [x] T024 [US1] En `src/olistbr/models/evaluation.py`, `compare_to_reference(reference, variant)`: mejora = media de `error_referencia − error_idea` por pliegue, ruido = desviación estándar muestral (`ddof=1`) de esas diferencias, cociente = mejora / ruido, para el error en días y para el costo 3 a 1; si `ruido < 1e-12` el cociente es «no evaluable» (nunca se divide por cero)
- [x] T025 [US1] En `src/olistbr/models/evaluation.py`, `decide_idea(cv, forward, known_at_prediction, min_ratio=MIN_IMPROVEMENT_RATIO)`: devuelve `("conservar" | "descartar", motivo de una línea)`; conservar solo si (1) se conoce al predecir, (2) `mejora / ruido > 1` en el error en días y la mejora del costo es positiva, (3) la mejora hacia adelante no cambia de signo; si no hay datos hacia adelante, la condición 3 es «no evaluable» y no descarta
- [x] T026 [US1] Ejecutar `pytest tests/test_split.py tests/test_evaluation.py -q` y marcar T019 a T025 solo si pasan

**Checkpoint**: la medición y la regla de decisión funcionan y están probadas.

---

## Phase 6: User Story 4 - Comprobar que la mejora sirve hacia adelante (Prioridad: P2)

**Meta**: entrenar con compras hasta febrero de 2018 y validar con marzo a agosto de 2018.

**Prueba independiente**: la partición respeta las fechas y no comparte clientes.

- [x] T027 [P] [US4] En `tests/test_split.py`: `test_forward_split_respects_dates` (entrenamiento ≤ 2018-02-28, validación de 2018-03-01 a 2018-08-31) y `test_forward_split_has_no_shared_customers` (un cliente con compras en los dos periodos queda fuera del entrenamiento)
- [x] T028 [P] [US4] En `tests/test_evaluation.py`: `test_evaluate_holdout_fits_only_on_train` (devuelve `mae`, `cost`, `n_train`, `n_valid`)
- [x] T029 [US4] En `src/olistbr/data/split.py`, `forward_split(df, train_end=FORWARD_TRAIN_END, valid_start=FORWARD_VALID_START, valid_end=FORWARD_VALID_END, group_col=GROUP_COL, date_col="order_purchase_timestamp")`: **el entrenamiento no incluye a los clientes que aparecen en la validación**; llama a `assert_disjoint_groups`
- [x] T030 [US4] En `src/olistbr/models/evaluation.py`, `evaluate_holdout(pipeline, train, valid, target=TARGET)`: ajusta con `train` y devuelve `{"mae", "cost", "n_train", "n_valid"}` evaluados en `valid`
- [x] T031 [US4] Ejecutar `pytest tests/test_split.py tests/test_evaluation.py -q` y marcar T027 a T030 solo si pasan

**Checkpoint**: las cuatro historias de código están listas.

---

## Phase 7: User Story 5 - Notebook, tabla y reporte (Prioridad: P3)

**Meta**: regenerar la tabla desde cero con el notebook y dejar constancia de cada decisión.

**Prueba independiente**: ejecutar el notebook 04 de principio a fin y obtener `reports/04_features_tabla.csv` con 12 filas.

- [x] T032 [US5] Completar `notebooks/04_features.ipynb` (ya existe como plantilla del curso; se genera con un script como el del notebook 03) para que **solo llame a funciones de `src/`**: leer `train.parquet` (nunca `test.parquet`), `add_features` una sola vez, `make_group_folds` una sola vez, medir la referencia y los 9 grupos con `evaluate_folds`, medir el plazo prometido como referencia («no usable en producción»), repetir con `forward_split` + `evaluate_holdout`, aplicar `compare_to_reference` y `decide_idea`, medir el conjunto de ideas conservadas como comprobación y guardar `reports/04_features_tabla.csv` (14 filas tras la revisión: referencia, 9 grupos, plazo prometido, línea base trivial y dos conjuntos; columnas de `data-model.md`)
- [x] T033 [US5] Ejecutar el notebook desde cero (`jupyter nbconvert --to notebook --execute notebooks/04_features.ipynb --output 04_features.ipynb`) sin errores; ejecutarlo una segunda vez y confirmar que `reports/04_features_tabla.csv` sale idéntico; confirmar con `grep -c "test.parquet"` que da `0` en el notebook y en los módulos nuevos
- [x] T034 [US5] Redactar `reports/04_features.md` **con las cifras reales de la tabla**, una fila por idea (conservada o descartada) con su cifra, `mejora / ruido`, resultado hacia adelante y motivo; explicar las limitaciones (tendencia entre años que el modelo no anticipa, una sola temporada de fin de año, separador interno de `TargetEncoder` que no agrupa por cliente, posible redundancia entre las ideas 1, 2 y 7); no escribir ninguna cifra que no salga del notebook
- [x] T035 [P] [US5] Actualizar `reports/bitacora_decisiones.md` con los resultados reales de la etapa y `especificaciones/04-features.md` (secciones 6 y 7: marcar solo lo verificado)

**Checkpoint**: la etapa se puede regenerar de principio a fin.

---

## Phase 8: Polish y revisión

- [x] T036 [P] Revisión de fuga con el subagente `auditor-de-fuga` sobre `src/olistbr/features/`, `src/olistbr/models/`, `src/olistbr/data/split.py` y el notebook 04; corregir lo que encuentre o anotar por qué no aplica
- [x] T037 [P] Revisión con el subagente `revisor-de-codigo-ia` para que cada integrante pueda explicar una variable; dejar el informe resumido en `reports/04_features.md`
- [x] T038 Aplicar los arreglos de T036 y T037, volver a ejecutar el notebook 04 y `pytest tests -q` en verde; verificar `git status` (sin cambios en `data/raw/`, sin `.env`, `data/processed/` ignorado)
- [x] T039 Cerrar la etapa: marcar los criterios de la sección 7 de `especificaciones/04-features.md` que se verificaron y anotar el hallazgo que no esperaban para la puesta en común

---

## Dependencias y orden

- **Fase 1 → Fase 2 → historias.** La Fase 2 bloquea todo.
- **US2 (variables)**: arranca tras la Fase 2; es el MVP.
- **US3 (preprocesador y pipeline)**: necesita las constantes de US2 (T009) y `add_features` para sus pruebas.
- **US1 (medición y decisión)**: necesita US2 y US3 (hay que tener variables y pipeline para medir). Las pruebas de `make_group_folds` y de `decide_idea` no dependen de ellas y se pueden escribir antes.
- **US4 (hacia adelante)**: necesita `evaluate_folds`/pipeline de US1 y US3.
- **US5 (notebook y reporte)**: necesita US1 a US4 terminadas.
- **Polish**: después de US5.

## Oportunidades en paralelo

- T002 y T003 juntos; T007 y T008 (mismo archivo: se escriben en una sola sesión, no en paralelo real).
- T012, T013 y T014 (archivos distintos salvo los dos primeros); T019, T020 y T021.
- T036 y T037 juntos.

## Estrategia de implementación

1. **MVP**: Fases 1 a 3 (T001 a T011). Ya se tienen las variables creadas y probadas.
2. Sumar US3 y comprobar que el pipeline se ajusta solo con entrenamiento.
3. Sumar US1 (pliegues, medición y regla) y US4 (hacia adelante).
4. Cerrar con US5 (notebook, tabla, reporte) y la revisión de la Fase 8.
5. Parar y mostrar al grupo el resultado en cada checkpoint antes de seguir. **No se hace commit sin que el grupo lo pida.** Las cifras del reporte se escriben después de ver la tabla, nunca antes.

## Cambios posteriores a las revisiones (2026-10-31)

Surgieron de los informes de `auditor-de-fuga` y `revisor-de-codigo-ia` (resumidos en `reports/04_features.md`, sección «Revisiones»). No cambian ninguna regla aprobada.

- [x] T040 `models/evaluation.py`: `measure_reference`, `reference_row`, `measure_idea` (con `forward_check_column`, que detecta en `src/` si la validación hacia adelante contiene el fenómeno), `kept_groups`, `results_table` y `RESULT_COLUMNS`; columna nueva `adelante_mae`. Pruebas en `tests/test_evaluation.py`.
- [x] T041 `models/train.py`: `build_baseline_pipeline` (línea base trivial: siempre la mediana). Prueba en `tests/test_models.py`.
- [x] T042 `features/engineering.py`: `_log1p` y `_ratio` también dan vacío ante infinitos de entrada; `data/split.py`: `forward_split` falla con `ValueError` si las fechas se superponen o la validación queda vacía. Pruebas nuevas.
- [x] T043 Pruebas más estrictas: el historial no usa su propio objetivo (aserción que de verdad discrimina), una fila de validación recibe el promedio de entrenamiento, `promised_days` no llega al modelo final (y sí a la referencia), línea muerta eliminada.
- [x] T044 Texto de decisión: el motivo dice «no empeora hacia adelante» en vez de «se mantiene»; `especificaciones/04-features.md` corregida (`GroupKFold` sí admite mezcla con semilla); plan corregido (66 columnas la referencia y 103 con todos los grupos, no 1.200).
- Pendientes menores, improbables con estos datos: identificadores de cliente vacíos en `make_group_folds` y columnas 100 % vacías dentro de un pliegue.
