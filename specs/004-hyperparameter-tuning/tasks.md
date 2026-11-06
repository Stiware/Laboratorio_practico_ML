---

description: "Lista de tareas de la etapa 6 · Fase 2 · Optimización de hiperparámetros de OnTrack"
---

# Tasks: Optimización de hiperparámetros de OnTrack (etapa 6, Fase 2)

**Input**: documentos de diseño en `specs/004-hyperparameter-tuning/`

**Prerequisites**: `plan.md` (aprobado el 2026-11-05), `spec.md`, `research.md`, `data-model.md`, `contracts/tuning-functions.md`, `quickstart.md`

**Tests**: SÍ se piden (constitución, principio IV; `CLAUDE.md`, regla 9; `especificaciones/06-optimizacion.md` §6). Las pruebas usan tablas pequeñas o datos sintéticos hechos a mano y **no leen `data/processed/`**. Primero se escribe la prueba y se ve que falla; después la función.

**Organization**: por historia de usuario del `spec.md`. **Orden de las fases:** US1 (comprobación honesta y reglas) → US2 (búsqueda) → US3 (registro y exportación) → US4 (hacia adelante y regiones) → notebook y reporte. US1 va primero porque sus funciones se prueban solas con plazos inventados y son la decisión central de la etapa.

## Format: `[ID] [P?] [Story] Descripción con ruta`

- **[P]**: se puede hacer en paralelo (archivos distintos, sin depender de una tarea sin terminar).
- **[Story]**: US1 a US4, la historia del `spec.md`.
- **Regla del proyecto:** una tarea se marca `[x]` **solo cuando su prueba pasa**. Código nuevo con docstring en español; nombres de código en inglés.
- **Comando de prueba:** `.venv/bin/python -m pytest tests -q` desde la raíz del repo.
- **Prohibido en toda la etapa:** abrir `data/processed/test.parquet`; modificar `models/train.py`, `models/evaluation.py`, `models/metrics.py` o `tracking/mlflow_utils.py`; usar `promised_days`, la marca de fin de año (`peak_season`) o el historial (`history`); abrir o imprimir `.env`; usar `git add .` sin revisar `git status`; hacer commit sin que el grupo lo pida.

## Path Conventions

Proyecto único: `src/olistbr/`, `tests/`, `notebooks/`, `reports/`, `scripts/generadores/` (rutas del `plan.md`).

---

## Phase 1: Setup

- [x] T001 Verificar el punto de partida: `lightgbm`, `optuna`, `mlflow` y `sklearn.ensemble.HistGradientBoostingRegressor` se importan en el `.venv`, y `.venv/bin/python -m pytest tests -q` da 166 pruebas en verde (las pruebas no necesitan `data/processed/`)
- [x] T002 Agregar a `src/olistbr/config.py`, cada una con su comentario del motivo: `RESERVE_SIZE = 0.25` y `RESERVE_SEED = 7` (guía §3.4), `TUNING_TRIALS = 20` (Q3), `BOOTSTRAP_ROUNDS = 1000` (regla 5), `TUNING_EXPERIMENT = "ontrack-optimizacion"` (Q8) y `TIE_WINNER = "LightGBM cuantil 0,75"` (Q7); agregar en `tests/test_split.py` la prueba `test_config_tuning_constants` que verifica esos valores

---

## Phase 2: Foundational (bloquea las historias)

- [x] T003 Crear `src/olistbr/models/tuning.py` con el docstring del módulo (qué hace la optimización, por qué el percentil 75 queda fijo, por qué se sortean clientes) y `tests/test_tuning.py` vacío con sus importaciones
- [x] T004 [P] Prueba `test_reserve_split_25_percent_is_disjoint` en `tests/test_split.py`: `split_by_group(df, test_size=RESERVE_SIZE, seed=RESERVE_SEED)` sobre una tabla hecha a mano con clientes repetidos no comparte clientes y aparta ≈ 25 % de los clientes (la función ya existe: no se modifica)
- [x] T005 [P] En `src/olistbr/models/tuning.py`: `starting_candidate(name)` (devuelve `candidates()[name]`; `KeyError` con mensaje claro si `name` no es uno de los dos finalistas) y la constante `FINALIST_NAMES = ("LightGBM cuantil 0,75", "HistGradientBoosting cuantil 0,75")`. Pruebas en `tests/test_tuning.py`: `test_starting_candidate_matches_tournament` (mismos parámetros que `candidates()`, familia `"B"`, sin escalar) y `test_starting_candidate_rejects_other_models`

**Checkpoint:** 166 pruebas anteriores + las nuevas en verde.

---

## Phase 3: User Story 1 - Saber si optimizar vale la pena, con una regla escrita antes de mirar (Prioridad: P1) 🎯 MVP

**Goal:** comparar dos plazos recomendados en la reserva con un intervalo del 95 % que sortea clientes completos y aplicar las reglas aprobadas.

**Independent Test:** con plazos inventados (por ejemplo, uno siempre un día mejor que el otro) la diferencia y su intervalo coinciden con el cálculo a mano, y las reglas devuelven la decisión esperada en cada caso.

### Pruebas (se escriben primero y deben fallar)

- [x] T006 [P] [US1] En `tests/test_tuning.py`, pruebas de `bootstrap_cost_difference`: `test_identical_plans_give_zero_interval` (diferencia 0 e intervalo [0; 0]), `test_constant_difference_collapses_interval` (si A cuesta siempre lo mismo menos que B, el intervalo es ese valor), `test_resamples_customers_not_rows` (con un cliente de muchos pedidos el intervalo es más ancho que sorteando filas, y el resultado coincide con la media ponderada por pedidos), `test_bootstrap_is_reproducible` (misma semilla → mismo intervalo) y `test_bootstrap_rejects_bad_input` (largos distintos, clientes vacíos o `n_boot < 1` → `ValueError`)
- [x] T007 [P] [US1] En `tests/test_tuning.py`, `test_decide_params_rule`: `high < 0` → `"optimizados"`; intervalo que cruza el cero → `"partida"`; `high == 0` (toca el cero) → `"partida"`; `low > 0` → `"partida"` con motivo «el optimizado es peor»; `low > high` → `ValueError`
- [x] T008 [P] [US1] En `tests/test_tuning.py`, `test_choose_finalist_rule`: diferencia A − B con `high < 0` → A; `low > 0` → B; cruza el cero → `tie_winner` con motivo «empate práctico»; `tie_winner` que no es A ni B → `ValueError`
- [x] T009 [P] [US1] En `tests/test_tuning.py`, `test_fit_and_recommend_does_not_see_valid` (el modelo se ajusta solo con `train`: cambiar el objetivo de `valid` no cambia los plazos) y `test_n_trees_used_reports_early_stopping` (HistGradientBoosting con parada temprana sobre una tabla sintética de más de 10.000 filas usa menos árboles que `max_iter`; LightGBM devuelve su número de árboles; otro modelo → `TypeError`)

### Implementación

- [x] T010 [US1] En `src/olistbr/models/tuning.py`: `bootstrap_cost_difference(real, plan_a, plan_b, groups, n_boot=BOOTSTRAP_ROUNDS, seed=RANDOM_STATE)` según `research.md` D8 (costo por pedido con `delivery_cost` 3 a 1, suma y conteo por cliente, 1.000 sorteos de clientes con reemplazo, devuelve `(diferencia, percentil 2,5, percentil 97,5)`), con las validaciones de T006
- [x] T011 [US1] En `src/olistbr/models/tuning.py`: `decide_params(low, high)` y `choose_finalist(low, high, name_a, name_b, tie_winner=TIE_WINNER)` según `research.md` D9, cada una devuelve `(decisión, motivo)` en español
- [x] T012 [US1] En `src/olistbr/models/tuning.py`: `fit_and_recommend(candidate, train, valid)` (arma `build_pipeline(TOURNAMENT_GROUPS, model=clone(candidate.model), scale=candidate.scale)`, ajusta con `fit_deadline_model` usando solo `train` y devuelve `(real, plazo recomendado, modelo)` en `valid`) y `n_trees_used(deadline_model)` (`n_iter_` en HistGradientBoosting, número de árboles del booster en LightGBM)

**Checkpoint:** T006 a T009 en verde. La regla de la etapa ya está en código probado.

---

## Phase 4: User Story 2 - Buscar parámetros dentro de un presupuesto y con rangos razonados (Prioridad: P1)

**Goal:** estudios de Optuna de 20 ensayos por finalista, con los rangos aprobados, el percentil 75 fijo, reproducibles y reanudables.

**Independent Test:** con una tabla pequeña y 2 ensayos (`log=False`), `tune` devuelve 2 ensayos con costos finitos y parámetros dentro de los rangos; con la misma semilla repite los parámetros; con `storage` reanuda.

### Pruebas (se escriben primero y deben fallar)

- [x] T013 [P] [US2] En `tests/test_tuning.py`: `test_lgbm_space_within_ranges` y `test_hgb_space_within_ranges` con `optuna.trial.FixedTrial` en los dos extremos de cada rango de `especificaciones/06-optimizacion.md` §4 (LightGBM: `n_estimators` 150–800, `learning_rate` 0,01–0,2, `num_leaves` 15–127, `min_child_samples` 20–300, `subsample` 0,6–1,0, `colsample_bytree` 0,5–1,0, `reg_lambda` 0,001–10; HistGradientBoosting: `max_iter` 150–800, `learning_rate` 0,01–0,2, `max_leaf_nodes` 15–127, `min_samples_leaf` 20–300, `l2_regularization` 0,001–10, `max_features` 0,5–1,0), y que el espacio devuelve exactamente esas claves
- [x] T014 [P] [US2] En `tests/test_tuning.py`: `test_lgbm_quantile_is_fixed` (`objective="quantile"`, `alpha=0.75`, `subsample_freq=1`), `test_hgb_quantile_is_fixed` (`loss="quantile"`, `quantile=0.75`), `test_hgb_has_no_early_stopping` (`early_stopping=False`), `test_lgbm_rejects_quantile_override` y `test_hgb_rejects_quantile_override` (`params` con `objective`, `alpha`, `loss`, `quantile` o `early_stopping` → `ValueError`), `test_finalists_are_the_two_tournament_names` y `test_tuned_candidate_is_family_b`
- [x] T015 [P] [US2] En `tests/test_tuning.py`: `test_tune_runs_requested_trials` (2 ensayos, `log=False`, costos finitos, cada ensayo guarda en `user_attrs` las medias y desviaciones de las métricas), `test_tune_is_reproducible` (misma semilla → mismos parámetros), `test_tune_resumes_from_storage` (con un `.db` en `tmp_path`: 1 ensayo, luego `n_trials=2` hace solo 1 más), `test_tune_counts_failed_trials_in_budget` (un ensayo con costo no finito queda `FAIL` y cuenta en los 2) y `test_tune_rejects_bad_input` (`n_trials < 1` o `X` sin `customer_unique_id` → `ValueError`)

### Implementación

- [x] T016 [US2] En `src/olistbr/models/tuning.py`: `lgbm_quantile_space(trial)` y `hgb_quantile_space(trial)` con los rangos de T013 (`learning_rate` y regularización en escala logarítmica), cada rango con un comentario del motivo
- [x] T017 [US2] En `src/olistbr/models/tuning.py`: `make_lgbm_quantile(params, random_state=RANDOM_STATE)` (fijos: `objective="quantile"`, `alpha=TARGET_QUANTILE`, `subsample_freq=1`, `n_jobs=2`, `deterministic=True`, `force_row_wise=True`, `verbose=-1`) y `make_hgb_quantile(params, random_state=RANDOM_STATE)` (fijos: `loss="quantile"`, `quantile=TARGET_QUANTILE`, `early_stopping=False`), las dos rechazan las claves fijas (`research.md` D2); `Finalist(space, make)` como `NamedTuple`, `FINALISTS` y `tuned_candidate(name, params)`
- [x] T018 [US2] En `src/olistbr/models/tuning.py`: `tune(name, X, y, folds, n_trials=TUNING_TRIALS, seed=RANDOM_STATE, log=True, storage=None)` según `research.md` D6: objetivo con `evaluate_candidate_folds(name, tuned_candidate(name, params), X, y, folds)`, minimiza `costo` medio, guarda medias y desviaciones (`ddof=1`) en `user_attrs`, `ValueError` si el costo no es finito, `TPESampler(seed)`, `study_name=f"ontrack-{slug}-costo"`, `load_if_exists=True`, `catch=(ValueError,)`, hace solo los ensayos que faltan y avisa al reanudar. Con `log=False` no toca MLflow

**Checkpoint:** T013 a T015 en verde.

---

## Phase 5: User Story 3 - Dejar registrado cada ensayo para comparar, reproducir y auditar (Prioridad: P2)

**Goal:** una ejecución principal por finalista con una ejecución hija por ensayo en el experimento `ontrack-optimizacion`, y la tabla de ensayos exportable.

**Independent Test:** con MLflow local en `tmp_path` y 2 ensayos, el experimento tiene 1 ejecución principal y 2 hijas, cada una con sus parámetros, `costo_media`, `costo_desv` y las etiquetas; `trials_table` tiene las columnas de `data-model.md`.

- [x] T019 [P] [US3] En `tests/test_tuning.py`: `test_tune_logs_one_nested_run_per_trial` (MLflow con `configure_mlflow(force_local=True, local_db=tmp_path / "mlflow.db")`; 2 ensayos → 1 principal + 2 hijas con etiquetas `etapa=06_optimizacion`, `modelo` y `metrica_optimizada=costo`, y sin ningún identificador de cliente en parámetros ni métricas) y `test_trials_table_columns` (columnas `modelo`, `ensayo`, `estado`, `param_<nombre>`, `costo_media`, `costo_desv`, `mae_media`, `rmse_media`, `r2_media`, `sesgo_media`, `tarde_media`, `cobertura_media`, `fit_seconds_media`, `mlflow_run_id`; una fila por ensayo, incluidos los fallidos)
- [x] T020 [US3] En `src/olistbr/models/tuning.py`: dentro de `tune`, si `log=True`, abrir `mlflow.start_run(run_name=f"optuna-{slug}")`, registrar cada ensayo con `log_run(f"{slug}-ensayo-{n:02d}", params, métricas, tags, nested=True)` (sin reescribir `mlflow_utils.py`), guardar el `run_id` en `user_attrs` y, al terminar, `best_costo` y los mejores parámetros en la ejecución principal
- [x] T021 [US3] En `src/olistbr/models/tuning.py`: `trials_table(study, name)` con las columnas de T019

**Checkpoint:** T019 en verde.

---

## Phase 6: User Story 4 - Revisar el modelo elegido hacia adelante y por región, sin decidir con eso (Prioridad: P3)

**Goal:** partición hacia adelante que valida solo con clientes reservados, para las tablas informativas.

**Independent Test:** con una tabla hecha a mano con fechas, la validación solo tiene clientes reservados y compras del 2018-03-01 al 2018-08-31, el entrenamiento solo clientes de búsqueda con compras hasta el 2018-02-28, y ningún cliente está en los dos lados.

- [x] T022 [P] [US4] En `tests/test_split.py`: `test_reserved_forward_split_uses_only_reserved_customers_in_valid`, `test_reserved_forward_split_dates` (días límite incluidos completos) y `test_reserved_forward_split_rejects_shared_customers` (`AssertionError`), más `ValueError` si las fechas se cruzan o un lado queda vacío
- [x] T023 [US4] En `src/olistbr/data/split.py`: `reserved_forward_split(search, reserved, train_end=FORWARD_TRAIN_END, valid_start=FORWARD_VALID_START, valid_end=FORWARD_VALID_END, group_col=GROUP_COL, date_col="order_purchase_timestamp")` según `research.md` D10, con `assert_disjoint_groups`; `forward_split` no se modifica

**Checkpoint:** T022 en verde. Las tablas por mes y por región se calculan en el notebook con funciones que ya existen (`evaluate_candidate_holdout`, `coverage_by_month`, `metrics_by_group`).

---

## Phase 7: Notebook, archivos y reporte (todas las historias)

- [x] T024 Crear `scripts/generadores/generar_notebook_06.py` (misma forma que `generar_notebook_05.py`) que arma `notebooks/06_optimizacion.ipynb` sobre la plantilla del curso, en el orden del `plan.md`: (1) `configure_mlflow()` y `set_experiment(TUNING_EXPERIMENT)` mostrando dónde quedó; (2) leer `train.parquet` (no `train.joblib`), `add_features`, `split_by_group(RESERVE_SIZE, RESERVE_SEED)` con `assert_disjoint_groups`; (3) `make_group_folds` sobre la búsqueda con su `assert`; (4) `tune` por finalista con `storage="data/processed/optuna_optimizacion.db"` y `trials_table`, mostrando el mejor costo **como optimista**; (5) comprobación honesta por finalista (`fit_and_recommend` partida y optimizado, `n_trees_used` de la partida, `bootstrap_cost_difference`, `decide_params`); (6) `bootstrap_cost_difference` LightGBM − HistGradientBoosting y `choose_finalist`; (7) informativo: `reserved_forward_split`, `evaluate_candidate_holdout` y `coverage_by_month` de partida y optimizado de cada finalista, y `metrics_by_group` por región del elegido; (8) guardar `reports/06_ensayos.csv` y `reports/06_best_params.json` (forma de `data-model.md`). Actualizar `scripts/generadores/README.md`
- [x] T025 Generar el notebook y comprobar que **no** contiene `test.parquet` (`grep -c "test.parquet" notebooks/06_optimizacion.ipynb src/olistbr/models/tuning.py` da 0 en ambos)
- [x] T026 Ejecutar el notebook de principio a fin (requiere `data/processed/train.parquet`; ver `quickstart.md`) y verificar en MLflow 20 ejecuciones hijas por finalista. **Si faltan los datos en esta máquina, la tarea queda pendiente y se avisa; no se inventan cifras**
- [x] T027 Reproducibilidad: borrar `data/processed/optuna_optimizacion.db`, ejecutar dos veces y comparar `06_best_params.json` y `06_ensayos.csv` sin la columna de tiempo (SC-006)
- [x] T028 Completar `reports/06_optimizacion.md` (plantilla existente) **después** de ver las cifras: espacio, presupuesto y por qué, reserva, dónde quedó el seguimiento, mejor ejecución (marcada como optimista), comprobación honesta con la tabla de la plantilla, resultado de la regla, decisión, árboles que usó realmente la partida de HistGradientBoosting, tablas informativas hacia adelante y por región, lo que no se pudo concluir y limitaciones (vendedores frecuentes de la etapa 3, reserva usada dos veces, costos no comparables con el torneo)

---

## Phase 8: Polish y revisión

- [x] T029 `.venv/bin/python -m pytest tests -q` en verde (166 anteriores + nuevas) y confirmar que `models/train.py`, `models/evaluation.py`, `models/metrics.py` y `tracking/mlflow_utils.py` no cambiaron (`git diff --stat`)
- [x] T030 [P] Revisión con `auditor-de-fuga` sobre `src/olistbr/models/tuning.py`, `src/olistbr/data/split.py` y el notebook 06 (pedir que confirme el riesgo de la lista de vendedores frecuentes)
- [x] T031 [P] Revisión con `revisor-de-codigo-ia` sobre los mismos archivos y el reporte
- [x] T032 Aplicar lo que digan las revisiones y resumirlo en `reports/06_optimizacion.md` (sección «Revisiones»)
- [x] T033 Actualizar `reports/bitacora_decisiones.md`, la sección 7 de `especificacion-proyecto.md`, los criterios de cierre de `especificaciones/06-optimizacion.md` (§8) y `reports/proxima_sesion.md` con el modelo elegido que pasa a la etapa 7

---

## Dependencias y orden

- Phase 1 → Phase 2 → (Phase 3 US1, Phase 4 US2 y Phase 6 US4 pueden ir en cualquier orden) → Phase 5 US3 (necesita `tune` de US2) → Phase 7 (necesita todo) → Phase 8.
- T010 depende de T006; T011 de T007 y T008; T012 de T009; T016 y T017 de T013 y T014; T018 de T015, T016 y T017; T020 y T021 de T019 y T018; T023 de T022; T024 de T012, T018, T021 y T023; T026 de T024 y de que exista `train.parquet`.

## Oportunidades en paralelo

- T004 y T005 (archivos distintos).
- T006, T007, T008 y T009 (pruebas en el mismo archivo, escritas en una sola sesión).
- T013, T014 y T015; T022 (otro archivo) en paralelo con las fases 3 y 4.
- T030 y T031.

## Estrategia de implementación

1. **MVP**: Fases 1 a 3 (T001 a T012). La regla de decisión y el intervalo por cliente quedan probados con datos inventados.
2. Sumar US2 (búsqueda) y US3 (registro): ya se puede correr una búsqueda corta con datos sintéticos.
3. Sumar US4 (partición hacia adelante) y cerrar con el notebook, el reporte y las revisiones.
4. Parar y mostrar al grupo el resultado en cada checkpoint. **No se hace commit sin que el grupo lo pida.** Las cifras del reporte se escriben después de ver los archivos, nunca antes.
5. En esta máquina faltan `data/raw/` y `data/processed/`: las fases 1 a 6 y T024–T025 se pueden hacer sin datos; T026 a T028 necesitan regenerar `train.parquet`.

## Cambios posteriores a las revisiones (2026-11-05)

Surgieron de los informes de `auditor-de-fuga` (fuga leve, ninguna grave) y `revisor-de-codigo-ia` (aceptar con cambios), resumidos en `reports/06_optimizacion.md`, sección «Revisiones». No cambian ninguna regla aprobada.

- [x] T034 `data/split.py`: `reserved_forward_split(..., known_at_cutoff=True)` deja en el entrenamiento hacia adelante solo los pedidos ya entregados antes del corte (compra + días de entrega). Prueba: `test_reserved_forward_split_drops_orders_not_delivered_by_the_cutoff`.
- [x] T035 `models/tuning.py`: excepción `NonFiniteCost` (solo un costo no finito marca un ensayo como fallido; otro error detiene la búsqueda), el ensayo fallido también se registra en MLflow (`estado=FAIL`), el intervalo y las reglas rechazan entradas vacías o no finitas, y se rechazan más parámetros fijos. Pruebas: `test_only_non_finite_cost_is_caught`, `test_failed_trial_is_also_logged`, `test_bootstrap_rejects_missing_values_and_empty_input`, `test_rules_reject_non_finite_intervals`, `test_other_fixed_settings_cannot_be_overridden`.
- [x] T036 Pruebas que faltaban: partida con los números del torneo escritos a mano, la comprobación mide igual que `evaluate_candidate_holdout`, y el plazo prometido, el cliente y el pedido no llegan al modelo.
- [x] T037 Generador del notebook 06: guarda `06_comprobacion.csv` y `06_adelante.csv`, anota los pedidos hacia adelante y los minutos en el JSON, define explícitamente los días reales de la reserva. Hecho: notebook ejecutado de nuevo (reutilizó los 40 ensayos; misma decisión) y cifras hacia adelante actualizadas.
