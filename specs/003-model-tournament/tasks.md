---

description: "Lista de tareas de la etapa 5 · Fase 1 · Torneo de modelos de OnTrack"
---

# Tasks: Torneo de modelos de OnTrack (etapa 5, Fase 1)

**Input**: documentos de diseño en `specs/003-model-tournament/`

**Prerequisites**: `plan.md` (aprobado el 2026-10-31), `spec.md`, `research.md`, `data-model.md`, `contracts/tournament-functions.md`

**Tests**: SÍ se piden (constitución, principio IV, y `CLAUDE.md`). Las pruebas usan tablas pequeñas o datos sintéticos hechos a mano y **no leen `data/processed/`**. Primero se escribe la prueba y se ve que falla; después la función.

**Organization**: por historia de usuario del `spec.md`. **Orden de las fases:** US3 (candidatos y pipeline) → US2 (métricas y modelo de plazo) → US1 (evaluación y tabla) → US4 (comparación y finalistas) → US5 (notebook y reporte), porque cada una usa a la anterior.

**Alcance de hoy:** solo la Fase 1 (torneo). **La Fase 2 (optimización, `tuning.py` del ejemplo) NO se hace hoy: se hará en otra sesión**, con su propia especificación (`especificaciones/06-optimizacion.md`).

## Format: `[ID] [P?] [Story] Descripción con ruta`

- **[P]**: se puede hacer en paralelo (archivos distintos, sin depender de una tarea sin terminar).
- **[Story]**: US1 a US5, la historia del `spec.md`.
- **Regla del proyecto:** una tarea se marca `[x]` **solo cuando su prueba pasa**. Código nuevo con docstring en español; nombres de código en inglés.
- **Comando de prueba:** `.venv/bin/python -m pytest tests -q` desde la raíz del repo.
- **Prohibido en toda la etapa:** abrir `data/processed/test.parquet`; usar `promised_days` o la marca de fin de año (`peak_season`) en el torneo; usar `git add .` sin revisar `git status`; hacer commit sin que el grupo lo pida.

## Path Conventions

Proyecto único: `src/olistbr/`, `tests/`, `notebooks/`, `reports/` (rutas del `plan.md`).

---

## Phase 1: Setup

- [x] T001 Verificar el punto de partida: `data/processed/train.parquet` existe, `lightgbm`, `xgboost` y `sklearn.ensemble.HistGradientBoostingRegressor` se importan en el `.venv`, y `pytest tests -q` da 119 pruebas en verde
- [x] T002 Agregar a `src/olistbr/config.py`, cada una con su comentario del motivo: `TOURNAMENT_GROUPS = ("logs", "same_state", "regions", "calendar", "distance_interaction", "ratios", "multi_seller")`, `TARGET_QUANTILE = COST_UNDERESTIMATE / (COST_UNDERESTIMATE + COST_OVERESTIMATE)` (= 0,75), `CALIBRATION_SIZE = 0.2`, `MIN_WIN_SHARE = 0.8`

---

## Phase 2: Foundational (bloquea las historias)

- [x] T003 Agregar a `tests/conftest.py` la fixture `quantile_frame`: fábrica de una tabla sintética con las columnas de `train.parquet` (como `frame_factory`) pero con `dias_entrega = 4 + distance_km / 200 + ruido gamma` (ruido asimétrico, para que el percentil 75 difiera de la mediana), clientes repetidos y fechas entre 2017-06 y 2018-04; devuelve la tabla con las variables derivadas de `add_features`
- [x] T004 Ejecutar `pytest tests -q` y confirmar que las 119 pruebas anteriores siguen en verde con la fixture nueva

**Checkpoint**: se puede empezar a escribir pruebas del torneo.

---

## Phase 3: User Story 3 - Candidatos y pipeline general (Prioridad: P2)

**Meta**: los 10 candidatos definidos en un solo lugar y un pipeline que acepta cualquier modelo.

**Prueba independiente**: cada candidato se ajusta con una tabla pequeña; el pipeline general da lo mismo que el de la etapa 4 por defecto.

### Pruebas (primero; deben fallar antes de implementar)

- [x] T005 [P] [US3] En `tests/test_models.py`: `test_candidates_have_expected_names_and_families` (10 candidatos: 2 `base`, 5 `A`, 3 `B`), `test_candidate_parameters_match_the_spec` (bosque con 100 árboles, profundidad 12, hoja 20, 33 % de variables, 50 % de filas; los modelos de la familia B con cuantil 0,75; semilla 42), `test_every_candidate_fits_a_tiny_table` (cada uno se ajusta y predice con la fixture `quantile_frame` reducida, sin vacíos en la salida)
- [x] T006 [P] [US3] En `tests/test_models.py`: `test_build_pipeline_accepts_any_model` (con `Ridge` y con `LGBMRegressor`) y `test_build_pipeline_default_is_still_linear_regression` (con `model=None` el resultado coincide con el de la etapa 4: mismas predicciones que `LinearRegression`)

### Implementación

- [x] T007 [US3] En `src/olistbr/models/train.py`: `Candidate` (NamedTuple `model, family, scale`) y `candidates(random_state=RANDOM_STATE)` con los parámetros de `research.md` D4 (LightGBM con `deterministic=True` y `force_row_wise=True`; XGBoost cuantil con `objective="reg:quantileerror"` y `quantile_alpha=0.75`; HistGradientBoosting con `loss="quantile"`); **«las dos líneas base van con `family="base"`»**
- [x] T008 [US3] En `src/olistbr/models/train.py`: generalizar `build_pipeline(model=None, groups=(), include_reference=False, scale=True)`: con `model=None` usa `LinearRegression()`; el resto no cambia
- [x] T009 [US3] Ejecutar `pytest tests/test_models.py -q` y marcar T003 a T008 solo si pasan

**Checkpoint**: existen los candidatos y un pipeline general.

---

## Phase 4: User Story 2 - Plazo recomendado y métricas sin filtrar información (Prioridad: P1) 🎯 MVP

**Meta**: cada modelo produce el plazo que recomendaría la app y las métricas salen de ese plazo.

**Prueba independiente**: con datos sintéticos de percentil conocido, el plazo cubre cerca del 75 %; el margen sale solo de clientes de calibración; la identidad costo = 4 × pérdida cuantílica se cumple.

### Pruebas

- [x] T010 [P] [US2] En `tests/test_metrics.py`: `test_bias_sign` (positivo cuando se subestima), `test_share_late_and_coverage_are_complements` (`real=[10,10,10,10]`, `recomendado=[9,10,11,12]` → tarde 0,25 y cobertura 0,75), `test_pinball_loss_times_four_equals_cost` (con datos aleatorios, diferencia menor a 1e-9; caso de mano: `real=10`, predicho 8 → 1,5 y costo 6; predicho 12 → 0,5 y costo 2), `test_pinball_loss_of_perfect_prediction_is_zero`
- [x] T011 [P] [US2] En `tests/test_metrics.py`: `test_metrics_by_group_splits_by_region` (MAE, porcentaje tarde y número de pedidos por región) y `test_metrics_by_group_handles_missing_group` (un grupo vacío no falla y queda aparte)
- [x] T012 [P] [US2] En `tests/test_models.py`: `test_regression_metrics_by_hand` (MAE, RMSE, R², sesgo, costo, tarde, cobertura con números de mano), `test_perfect_predictions_give_zero_error`, `test_regression_metrics_uses_recommended_for_cost` (el costo sale de `recommended`, el MAE de `predicted`)
- [x] T013 [P] [US2] En `tests/test_models.py`: `test_deadline_model_recommend_adds_margin`, `test_margin_comes_from_calibration_customers_only` (un modelo que siempre subestima 2 días da margen ≈ 2; uno perfecto, ≈ 0), `test_no_customer_in_fit_and_calibration`, `test_margin_gives_about_75_percent_coverage` (ruido normal: cobertura entre 70 % y 80 %), `test_family_b_has_zero_margin_and_uses_all_rows`, `test_margin_can_be_negative` (modelo que sobreestima), `test_family_b_quantile_models_cover_about_75_percent` (LightGBM cuantil sobre `quantile_frame`: cobertura entre 70 % y 80 %)

### Implementación

- [x] T014 [US2] En `src/olistbr/models/metrics.py`: `bias`, `share_late`, `coverage`, `pinball_loss(real, predicted, quantile=TARGET_QUANTILE)` y `metrics_by_group(real, predicted, recommended, groups)` (un grupo con valor vacío se informa aparte, no se descarta en silencio)
- [x] T015 [US2] En `src/olistbr/models/train.py`: `regression_metrics(real, predicted, recommended=None)` con MAE, RMSE (`root_mean_squared_error`), R² (`r2_score`), sesgo, costo, tarde y cobertura; si `recommended` es `None`, el plazo es `predicted`
- [x] T016 [US2] En `src/olistbr/models/train.py`: `DeadlineModel` (guarda `pipeline`, `margin`, tamaños; métodos `predict` y `recommend`) y `fit_deadline_model(candidate, X, y, groups, quantile=TARGET_QUANTILE, calibration_size=CALIBRATION_SIZE, seed=RANDOM_STATE)`: **«para la familia A se aparta el 20 % de los clientes con `GroupShuffleSplit`, se ajusta con el resto, el margen es el percentil 75 de (real − predicción) de los clientes apartados y se verifica con `assert_disjoint_groups` que ningún cliente está en el ajuste y la calibración»**; para `base` y `B` ajusta con todo y `margin = 0`
- [x] T017 [US2] Ejecutar `pytest tests/test_metrics.py tests/test_models.py -q` y marcar T010 a T016 solo si pasan

**Checkpoint**: el modelo de plazo y las métricas funcionan y están probados.

---

## Phase 5: User Story 1 - Evaluar a todos los candidatos y saber cuáles descartar (Prioridad: P1)

**Meta**: una tabla con todos los candidatos, cada pliegue guardado y la partición hacia adelante.

**Prueba independiente**: con tablas pequeñas, se obtienen las filas por pliegue con todas las columnas y el resumen correcto.

### Pruebas

- [x] T018 [P] [US1] En `tests/test_evaluation.py`: `test_evaluate_candidate_folds_returns_all_columns` (columnas de `data-model.md`: `modelo, familia, pliegue, mae, rmse, r2, sesgo, costo, tarde, cobertura, margen, n_ajuste, n_calibracion, n_validacion, fit_seconds`), `test_evaluate_candidate_folds_uses_same_folds_for_all` (dos candidatos evaluados con los mismos pliegues tienen los mismos `n_validacion`), `test_evaluate_candidate_holdout_fits_only_on_train` (la línea base de la mediana predice la mediana del **entrenamiento**, no de la validación)
- [x] T019 [P] [US1] En `tests/test_evaluation.py`: `test_tournament_summary_means_and_std` (media y desviación con `ddof=1` por modelo y métrica, columna `familia`, resultado hacia adelante incluido y el tiempo fuera del resumen)

### Implementación

- [x] T020 [US1] En `src/olistbr/models/evaluation.py`: `evaluate_candidate_folds(name, candidate, X, y, folds, groups_col=GROUP_COL)`: para cada pliegue llama a `fit_deadline_model` con el entrenamiento, evalúa con `regression_metrics` sobre la validación y mide `fit_seconds`; **no modifica `evaluate_folds` ni `evaluate_holdout` de la etapa 4**
- [x] T021 [US1] En `src/olistbr/models/evaluation.py`: `evaluate_candidate_holdout(name, candidate, train, valid, groups_col=GROUP_COL)` (la partición hacia adelante, con `pliegue = "adelante"`) y `tournament_summary(by_fold, forward)`
- [x] T022 [US1] Ejecutar `pytest tests/test_evaluation.py -q` y marcar T018 a T021 solo si pasan

**Checkpoint**: se pueden evaluar los 10 candidatos con los mismos pliegues.

---

## Phase 6: User Story 4 - Comparar por pares y elegir finalistas con una regla escrita (Prioridad: P2)

**Meta**: las reglas de empate, de superación de líneas base y de posiciones, en código con pruebas.

**Prueba independiente**: con costos por pliegue hechos a mano, cada función da el valor esperado, incluido el caso sin ruido.

### Pruebas

- [x] T023 [P] [US4] En `tests/test_evaluation.py`: `test_compare_models_counts_wins` (diferencias `[0,5; 0,6; 0,4; 0,5; 0,7]` → gana 5 de 5), `test_compare_models_zero_noise_is_not_evaluable`, `test_compare_models_consistent_vs_practical_tie` (diferencias `[0,5; −0,4; 0,3; −0,2; 0,1]` → 3 de 5 y cociente menor que 1: empate práctico)
- [x] T024 [P] [US4] En `tests/test_evaluation.py`: `test_rank_table_orders_by_metric` (1 = mejor; en `r2` mayor es mejor, en las demás menor), `test_beats_baselines_requires_more_than_noise`, `test_practical_ties_lists_indistinguishable_models`, `test_per_thousand`

### Implementación

- [x] T025 [US4] En `src/olistbr/models/evaluation.py`: `compare_models(best_costs, other_costs)`: por pliegue `diferencia = costo(otro) − costo(mejor)`; devuelve diferencia media, `wins` («k de n»), ruido con `ddof=1`, `diferencia / ruido` (NaN y `evaluable=False` si el ruido es menor a `NOISE_FLOOR`) y `consistent` (gana en al menos `MIN_WIN_SHARE` de los pliegues y el cociente es mayor que 1)
- [x] T026 [US4] En `src/olistbr/models/evaluation.py`: `rank_table(summary, metrics)`, `beats_baselines(by_fold, model, baselines)`, `practical_ties(by_fold, best)` y `per_thousand(difference_per_order)`
- [x] T027 [US4] Ejecutar `pytest tests/test_evaluation.py -q` y marcar T023 a T026 solo si pasan

**Checkpoint**: las cuatro historias de código están listas.

---

## Phase 7: User Story 5 - Notebook, tablas y reporte (Prioridad: P3)

**Meta**: regenerar las tablas desde cero y dejar constancia de cada decisión.

**Prueba independiente**: ejecutar el notebook 05 de principio a fin y obtener los cuatro CSV.

- [x] T028 [US5] Completar `notebooks/05_torneo.ipynb` (ya existe como plantilla del curso; se genera con un script, como el 03 y el 04) para que **solo llame a funciones de `src/`**: leer `train.parquet` (nunca `test.parquet`), `add_features`, `make_group_folds` y `forward_split`, evaluar los 10 candidatos con `evaluate_candidate_folds` y `evaluate_candidate_holdout` sobre las columnas de `TOURNAMENT_GROUPS`, resumir, ordenar por costo, tabla de posiciones, comparación por pares del mejor contra los siguientes y contra las líneas base, empates prácticos, diferencia de la regresión lineal contra el ganador (`per_thousand`), `metrics_by_group` por región para los finalistas, y guardar `reports/05_torneo_por_pliegue.csv`, `05_torneo_resumen.csv`, `05_torneo_tiempos.csv` y `05_torneo_regiones.csv`
- [x] T029 [US5] Ejecutar el notebook desde cero (`jupyter nbconvert --to notebook --execute notebooks/05_torneo.ipynb --output 05_torneo.ipynb`, unos 5 a 6 minutos) sin errores; ejecutarlo una segunda vez y confirmar que `05_torneo_por_pliegue.csv` y `05_torneo_resumen.csv` salen idénticos **sin la columna de tiempo**; confirmar con `grep -c "test.parquet"` que da `0` en el notebook y en los módulos de la etapa
- [x] T030 [US5] Redactar `reports/05_torneo.md` **con las cifras reales de los CSV**, generando las tablas con un script que lea los archivos: tabla de negocio y de exactitud (media ± desviación), posiciones, comparación por pares, lectura de la cobertura, aviso de que la exactitud de la familia B no es comparable con la A, diferencia de la regresión lineal contra el ganador en días por pedido y por 1.000 pedidos (sin dinero), revisión por región, 1 o 2 finalistas con su motivo y por qué el torneo no elige un campeón, y las limitaciones; no escribir ninguna cifra que no salga de los CSV
- [x] T031 [P] [US5] Actualizar `reports/bitacora_decisiones.md` (incluida la sección «Decisiones clave para la exposición»), `especificaciones/05-torneo.md` (sección 8: marcar solo lo verificado) y la sección 7 de `especificacion-proyecto.md` si surge una decisión nueva

**Checkpoint**: la etapa se puede regenerar de principio a fin.

---

## Phase 8: Polish y revisión

- [x] T032 [P] Revisión de fuga con el subagente `auditor-de-fuga` sobre `src/olistbr/models/`, `src/olistbr/config.py` y el notebook 05; corregir lo que encuentre o anotar por qué no aplica
- [x] T033 [P] Revisión con el subagente `revisor-de-codigo-ia` para que cada integrante pueda explicar el modelo de plazo, el margen y la regla de decisión; dejar el resumen en `reports/05_torneo.md`
- [x] T034 Aplicar los arreglos de T032 y T033, volver a ejecutar el notebook 05 (dos veces) y `pytest tests -q` en verde; verificar `git status` (sin cambios en `data/raw/`, sin `.env`, `data/processed/` ignorado)
- [x] T035 Cerrar la etapa: marcar los criterios de la sección 8 de `especificaciones/05-torneo.md` que se verificaron, anotar el hallazgo que no esperaban y dejar escrito que **la Fase 2 (optimización) queda para otra sesión**, con los finalistas que le pasa esta fase

---

## Dependencias y orden

- **Fase 1 → Fase 2 → historias.** La Fase 2 bloquea todo.
- **US3 (candidatos y pipeline)**: arranca tras la Fase 2.
- **US2 (plazo y métricas)**: necesita `Candidate` y el pipeline general de US3.
- **US1 (evaluación)**: necesita `fit_deadline_model` y `regression_metrics` de US2.
- **US4 (comparación)**: necesita los resultados por pliegue de US1 (sus pruebas usan números de mano y se pueden escribir antes).
- **US5 (notebook y reporte)**: necesita US1 a US4 terminadas.
- **Polish**: después de US5.

## Oportunidades en paralelo

- T005 y T006; T010, T011, T012 y T013 (archivos distintos salvo los de `test_models.py`, que se escriben en una sola sesión); T018 y T019; T023 y T024; T032 y T033.

## Estrategia de implementación

1. **MVP**: Fases 1 a 4 (T001 a T017). Ya existe el modelo de plazo con las métricas, probados con datos sintéticos.
2. Sumar US1 (evaluación) y comprobar que todos los candidatos se evalúan con los mismos pliegues.
3. Sumar US4 (comparación y reglas) y cerrar con US5 (notebook, reportes) y la revisión de la Fase 8.
4. Parar y mostrar al grupo el resultado en cada checkpoint. **No se hace commit sin que el grupo lo pida.** Las cifras del reporte se escriben después de ver los CSV, nunca antes.
5. **No empezar la Fase 2 (optimización) en esta sesión.**

## Tareas añadidas durante la implementación (2026-10-31)

Surgieron al escribir el notebook: para que solo orqueste, esas dos piezas viven en `src/` con su prueba. No cambian ninguna regla aprobada.

- [x] T036 `models/evaluation.py`: `tournament_table(by_fold, forward, baselines)`: resumen ordenado por costo, posiciones, superación de las líneas base, «mejor» (sin contar las líneas base), comparación contra el mejor y empates prácticos. Pruebas: `test_tournament_table_has_everything_the_report_needs`, `test_tournament_table_best_ignores_baselines`.
- [x] T037 `models/evaluation.py`: `regional_metrics(name, candidate, X, y, folds)`: MAE y porcentaje tarde por región con predicciones fuera de muestra. Prueba: `test_regional_metrics_splits_out_of_fold_predictions_by_region`.
- [x] T038 `build_pipeline(groups=(), include_reference=False, model=None, scale=True)`: `groups` se queda en primera posición para no romper el código de la etapa 4 (el contrato decía `model` primero). `fit_deadline_model(pipeline, family, X, y, groups, …)` recibe un pipeline ya construido, para poder probar el margen con modelos de juguete.

## Cambios posteriores a las revisiones (2026-10-31)

Surgieron de los informes de `auditor-de-fuga` (sin fuga) y `revisor-de-codigo-ia` (aceptar con cambios), resumidos en `reports/05_torneo.md`, sección «Revisiones». No cambian ninguna regla aprobada ni ninguna cifra del torneo (las tablas salieron idénticas).

- [x] T039 `models/train.py`: `fit_deadline_model` valida las entradas (`groups` con la misma longitud que `X`, sin vacíos) y `DeadlineModel.predict_with_deadline` evita calcular el margen en dos lugares; `models/evaluation.py` lo reutiliza.
- [x] T040 `models/evaluation.py`: `_fold_costs` falla con `KeyError` si el modelo no existe; `regional_metrics` exige la columna de región; `tournament_table` calcula el resumen una sola vez.
- [x] T041 `models/evaluation.py`: `coverage_by_month` (cobertura del plazo mes a mes) y su prueba; el notebook guarda `reports/05_torneo_cobertura_mensual.csv`.
- [x] T042 Pruebas fortalecidas: el margen es exactamente el percentil 75 (con errores conocidos), el modelo del margen no ve a los clientes de calibración, las predicciones regionales son fuera de muestra, reproducibilidad, las entradas no se modifican, la familia B pasa por `evaluate_candidate_folds`, y el cliente/pedido/plazo prometido no llegan al modelo del torneo. La banda de cobertura de la familia A se mantuvo en 70 %–80 % porque la calibración con ≈ 400 pedidos tiene un error de muestreo de ≈ ±2 puntos.
- [x] T043 Notebook 05: sin la comparación de MAE entre familias ni el cálculo manual de pedidos por mes; revisión por región para los 8 modelos; cobertura mes a mes; el aviso de pocos clientes de calibración ya no se silencia.
- [x] T044 Reporte reescrito con comprobaciones automáticas de cada afirmación; documentos corregidos (contrato de funciones, sección 7 de `especificacion-proyecto.md`, bitácora).
- Pendientes menores: control de la desventaja de datos de la familia A (80 % contra 100 %) y variación por semilla (Fase 2).
