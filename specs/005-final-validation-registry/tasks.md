---

description: "Lista de tareas de las etapas 7 y 8 · Validación final y registro del modelo de OnTrack"
---

# Tasks: Validación final y registro del modelo de OnTrack (etapas 7 y 8)

**Input**: documentos de diseño en `specs/005-final-validation-registry/`

**Prerequisites**: `plan.md` (aprobado el 2026-11-05), `spec.md`, `research.md`, `data-model.md`, `contracts/validation-registry-functions.md`, `quickstart.md`

**Tests**: SÍ se piden (constitución, principio IV; `CLAUDE.md`, regla 9). Las pruebas usan plazos inventados, tablas pequeñas y modelos de juguete, y **no leen `data/processed/`**. Primero la prueba (debe fallar), después la función.

**Organization**: por historia del `spec.md`. **Orden:** US1 (validación, P1) → US2 (segmentos) y US3 (SHAP) → US4 (registro, P1) → notebooks y reportes. **La prueba se abre en una sola tarea (T020)**, después de que todo lo anterior esté probado.

## Format: `[ID] [P?] [Story] Descripción con ruta`

- **[P]**: se puede hacer en paralelo (archivos distintos, sin depender de una tarea sin terminar).
- **Regla del proyecto:** una tarea se marca `[x]` **solo cuando su prueba pasa**.
- **Comando de prueba:** `.venv/bin/python -m pytest tests -q`.
- **Prohibido en toda la pieza:** leer `data/processed/test.parquet` fuera de T020 (apertura) y T026 (ida y vuelta); cambiar el modelo después de T020; modificar módulos existentes de `src/`; abrir o imprimir `.env`; volver a correr un generador sobre un notebook ya ejecutado sin guardar antes sus salidas; `git add .` sin revisar `git status`; commit sin que el grupo lo pida.

---

## Phase 1: Setup y precondiciones

- [x] T001 Verificar: etapa 6 cerrada (`reports/06_best_params.json` con `"modelo": "LightGBM cuantil 0,75"` en `eleccion`), `reports/prueba_abierta.txt` **no** existe, `shap` y `mlflow` se importan, y `pytest tests -q` en verde
- [x] T002 **Bloqueante para T020:** el `.gitignore` excluye `data/processed/` (`git status --untracked-files=all data/` no muestra nada). Si no, avisar al grupo y no seguir a T020
- [x] T003 Crear `src/olistbr/models/validation.py` y `src/olistbr/tracking/registry.py` con su docstring de módulo, y `tests/test_validation.py` y `tests/test_registry.py` con sus importaciones

---

## Phase 2: User Story 1 - Saber cuánto vale de verdad el modelo, abriendo la prueba una sola vez (Prioridad: P1) 🎯 MVP

**Independent Test:** con archivos temporales, la apertura deja constancia y falla la segunda vez; con plazos inventados, la tabla de la prueba coincide con el cálculo a mano.

### Pruebas (primero; deben fallar)

- [x] T004 [P] [US1] `tests/test_validation.py`: `test_test_set_opens_only_once` y `test_mark_records_date_and_time` (parquet y constancia en `tmp_path`; la segunda apertura lanza `RuntimeError` que dice cuándo se abrió)
- [x] T005 [P] [US1] `tests/test_validation.py`: `test_bootstrap_mean_constant`, `test_bootstrap_mean_resamples_customers` (un cliente con muchos pedidos ensancha el intervalo frente a sortear pedidos), `test_bootstrap_mean_reproducible`, `test_bootstrap_mean_rejects_bad_input` (largos distintos, vacíos, no finitos, clientes vacíos, `n_boot < 1`)
- [x] T006 [P] [US1] `tests/test_validation.py`: `test_baseline_plans_by_hand` (percentil 75 y mediana del entrenamiento, y plazo prometido = estimada − compra en días; falta de columna → `ValueError`)
- [x] T007 [P] [US1] `tests/test_validation.py`: `test_test_report_matches_manual_metrics` (costo, % tarde, cobertura, MAE y sesgo a mano) y `test_test_report_flags_confident_improvement` (modelo siempre mejor → `mejora_con_confianza = True`; igual → `False`); `KeyError` si falta el modelo o el baseline

### Implementación

- [x] T008 [US1] `src/olistbr/models/validation.py`: `open_test_once(path, mark_path)` (research D2)
- [x] T009 [US1] `src/olistbr/models/validation.py`: `bootstrap_mean(values, groups, n_boot=BOOTSTRAP_ROUNDS, seed=RANDOM_STATE)` (research D4)
- [x] T010 [US1] `src/olistbr/models/validation.py`: `baseline_plans(train, test, target=TARGET, purchase_col="order_purchase_timestamp", promised_col="order_estimated_delivery_date")` (research D3)
- [x] T011 [US1] `src/olistbr/models/validation.py`: `test_report(real, plans, groups, main_baseline, model_name)` reutilizando `bootstrap_cost_difference` para la diferencia (research D4)

**Checkpoint:** T004 a T007 en verde.

---

## Phase 3: User Story 2 - Ver dónde y cuándo falla el modelo (Prioridad: P2)

- [x] T012 [P] [US2] `tests/test_validation.py`: `test_slack_and_delay_by_hand`, `test_segment_table_by_hand` (región y mes), `test_ratio_one_is_mae`
- [x] T013 [US2] `src/olistbr/models/validation.py`: `slack_and_delay(real, plan)`, `segment_table(real, plan, segments)` (reutiliza `metrics_by_group`) y `cost_sensitivity(real, plans, ratios=COST_RATIOS)` (research D5 a D7)

**Checkpoint:** T012 en verde.

---

## Phase 4: User Story 3 - Explicar qué mueve el plazo (Prioridad: P2)

- [x] T014 [P] [US3] `tests/test_validation.py`: `test_top_shap_finds_the_only_useful_feature` (pipeline con preprocesador y LightGBM de juguete donde solo una variable importa: sale primera; modelo que no es de árboles → `TypeError`)
- [x] T015 [US3] `src/olistbr/models/validation.py`: `top_shap(pipeline, X, n=10, sample=2000, seed=RANDOM_STATE)` (research D9)

**Checkpoint:** T014 en verde.

---

## Phase 5: User Story 4 - Registrar el modelo evaluado y poder recuperarlo (Prioridad: P1)

- [x] T016 [P] [US4] `tests/test_registry.py` (MLflow local en `tmp_path`, modelo de juguete): `test_input_example_has_no_identifiers` (ni `customer_unique_id`, ni `order_id`, ni `order_estimated_delivery_date`, ni `promised_days`; solo las 25 columnas del modelo), `test_register_creates_a_version`, `test_champion_points_to_latest_version` (dos registros → apunta a la 2), `test_round_trip_predicts_the_same`
- [x] T017 [US4] `src/olistbr/tracking/registry.py`: `input_example`, `register_final_model` (firma inferida, `cloudpickle`, `registered_model_name` de `mlflow_utils`), `set_champion`, `load_champion` (research D10)

**Checkpoint:** T016 en verde; `pytest tests -q` completo en verde.

---

## Phase 6: Notebook 07 (abre la prueba) y su reporte

- [x] T018 `scripts/generadores/generar_notebook_07.py` sobre la plantilla del curso, en el orden del `plan.md`: (1) decisiones congeladas con fecha y hora leídas de `06_best_params.json`; (2) modelo final con `train.parquet` y `models/final_pipeline.joblib`; (3) `open_test_once`; (4) `baseline_plans` + `test_report`; (5) `segment_table`, `slack_and_delay`, `cost_sensitivity`, escenario con extremos; (6) `top_shap`; (7) guardar `reports/07_metricas_test.json`, `07_segmentos.csv`, `07_shap.csv`. Con `MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT`
- [x] T019 **Ensayo sin la prueba:** correr una copia temporal del notebook 07 con una tabla **falsa** en lugar de `test.parquet` (filas del entrenamiento con otro nombre, constancia y salidas en la carpeta temporal). Debe correr de principio a fin. Sus cifras no se usan para nada
- [x] T020 **Abrir la prueba (una sola vez):** generar `notebooks/07_validacion_interpretacion.ipynb` y ejecutarlo. Comprobar que `reports/prueba_abierta.txt` tiene una sola fecha. **Después de esta tarea no se cambia el modelo**
- [x] T021 Completar `reports/07_validacion.md` (plantilla) **después** de ver las cifras: decisiones congeladas, «calibración» (cobertura y recalibración como regla de uso), la prueba con intervalos y la conclusión de la regla, segmentos, holgura y atraso, sensibilidad, extremos, SHAP con sus límites y el atributo sensible, lo no evaluado y la conclusión

---

## Phase 7: Notebook 08 (registro) y su reporte

- [x] T022 `scripts/generadores/generar_notebook_08.py`: (1) `configure_mlflow`, `set_experiment("ontrack-modelo-final")` mostrando solo el nombre; (2) cargar `final_pipeline.joblib` y `07_metricas_test.json`; (3) `input_example` (de `train.parquet`) y `register_final_model` con nombre `ontrack_plazo_entrega`; (4) `set_champion`; (5) `load_champion` e ida y vuelta (500 pedidos de la prueba con `np.allclose`, y costo en toda la prueba contra `07_metricas_test.json`); (6) guardar `reports/08_registro.json`
- [x] T023 Generar y ejecutar `notebooks/08_registro_modelo.ipynb`; si Databricks falla, respaldo local con el motivo
- [x] T024 Comprobar en Databricks que existe `workspace.default.ontrack_plazo_entrega` con alias `champion` (consulta de solo lectura)
- [x] T025 Completar `reports/08_registro.md` (plantilla)

---

## Phase 8: Polish y revisión

- [x] T026 `grep -c "gmail\|databricks.com"` en `notebooks/07_*`, `notebooks/08_*`, `reports/07_*` y `reports/08_*` da 0; ningún archivo nuevo tiene identificadores de cliente o pedido
- [x] T027 `pytest tests -q` en verde y `git diff --stat` confirma que no cambió ningún módulo existente de `src/`
- [x] T028 [P] Revisión con `auditor-de-fuga` (validación: que la prueba no decida nada; registro: que sea el mismo modelo)
- [x] T029 [P] Revisión con `revisor-de-codigo-ia` (código, pruebas y cifras de los reportes contra los archivos)
- [x] T030 Aplicar lo que digan las revisiones **sin cambiar el modelo ni reabrir la prueba para decidir**; resumir en `reports/07_validacion.md` y `08_registro.md`
- [x] T031 Actualizar `reports/bitacora_decisiones.md`, `especificacion-proyecto.md` §7, criterios de cierre de `especificaciones/07-validacion.md` y `08-registro.md`, y `reports/proxima_sesion.md` (siguiente: etapa 9, app)

---

## Dependencias y orden

- Phase 1 → Phase 2 → (Phase 3, Phase 4 y Phase 5 en cualquier orden) → Phase 6 → Phase 7 → Phase 8.
- **T020 depende de T002 (gitignore), T018, T019 y de que todas las pruebas pasen.** T022 depende de T020. T024 depende de T023.

## Oportunidades en paralelo

- T004 a T007 (mismo archivo, una sesión); T012, T014 y T016 (archivos distintos); T028 y T029.

## Estrategia de implementación

1. **MVP:** Fases 1 y 2: la apertura única y la tabla de la prueba, probadas con datos inventados.
2. Sumar segmentos, SHAP y registro (fases 3 a 5), todo probado sin abrir la prueba.
3. Ensayo con tabla falsa (T019) y **solo entonces** abrir la prueba (T020).
4. Parar y mostrar al grupo el resultado de T020 antes de registrar (T022). **No se hace commit sin que el grupo lo pida.**

## Cambios posteriores a las revisiones (2026-11-05)

De `auditor-de-fuga` (fuga leve; ningún problema de datos ni de modelo) y `revisor-de-codigo-ia` (aceptar con cambios). **Ninguno cambia el modelo ni una cifra, y la prueba no se reabrió.** Resumen en `reports/07_validacion.md` y `08_registro.md`, sección «Revisiones».

- [x] T032 Informes: «abierta una sola vez para medir» y la segunda lectura de la etapa 8 declarada; la evidencia del congelamiento es la marca del notebook; hipótesis marcadas como tales; advertencias de la etapa 6; riesgo de Python 3.14; los 64 extremos no comparten clientes con el entrenamiento (comprobado).
- [x] T033 `validation.py`: `slack_and_delay` rechaza valores no finitos, `segment_table` agrupa etiquetas vacías como «sin dato» y valida largos, `baseline_plans` valida el objetivo y las fechas. Pruebas nuevas, incluidas la del intervalo de la cobertura y la de la semilla.
- [x] T034 `registry.py`: `set_champion(full_name, version=None)`; generador 08 con la versión explícita y un `except` limitado a errores de MLflow o de conexión. Pruebas nuevas: firma guardada, firma que acepta datos faltantes (con `pyfunc`) y versión explícita.
- [x] T035 `reports/08_verificacion_registro.json`: consulta de solo lectura del registro (versión 1, `READY`, única versión).
- Deuda técnica declarada (no se corrige aquí porque la regla era no modificar módulos existentes): `bootstrap_mean` repite la lógica de `bootstrap_cost_difference`; el escenario con extremos vive en el notebook.
