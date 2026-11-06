---

description: "Lista de tareas de la etapa 9 · Inferencia y app de OnTrack"
---

# Tasks: Inferencia y app de OnTrack (etapa 9)

**Input**: `specs/006-inference-app/` · **Prerequisites**: `plan.md`, `spec.md`, `contracts/inference-functions.md`

**Tests**: SÍ (principio IV; `CLAUDE.md` regla 9). Pruebas primero; ninguna lee `data/processed/` ni `models/model.joblib`.

**Prohibido:** leer `data/processed/test.parquet`; que la app consulte Databricks o lea `.env`; modificar módulos de `src/` fuera de `inference/`; regenerar un notebook ya
ejecutado sin guardar antes sus salidas; commit sin que el grupo lo pida.

## Phase 1: Setup

- [x] T001 Verificar precondiciones: existen `models/final_pipeline.joblib`, `reports/07_metricas_test.json`, `reports/07_segmentos.csv` y `reports/08_registro.json`; `streamlit.testing.v1.AppTest` se importa; `pytest tests -q` en verde (242)
- [x] T002 Confirmar que solo `app/streamlit_app.py` y `tests/test_inference.py` importan `olistbr.inference` (para reescribirlo sin romper otra etapa)

## Phase 2: User Story 1 y 2 - Predicción que valida (Prioridad: P1) 🎯 MVP

- [x] T003 [US1] `tests/test_inference.py` reescrito: pruebas de esquema (`test_form_has_the_eleven_fields`, `test_valid_ranges_match_the_spec`, `test_build_schema_round_trip`, `test_schema_has_no_identifiers`) con una tabla sintética (`frame_factory`)
- [x] T004 [US1] `tests/test_inference.py`: `test_order_to_features_fills_seller_and_missing_coords`, `test_features_match_manual_construction` (mismo plazo que el pipeline con las columnas hechas a mano), `test_predict_normal_order`, `test_date_rounds_up`, `test_several_orders`, `test_load_model_missing_file`
- [x] T005 [US2] `tests/test_inference.py`: `test_impossible_values_are_rejected` (cada rango de FR-004), `test_required_fields`, `test_unknown_state_rejected`, `test_seller_items_not_above_items`, `test_old_date_rejected`, `test_unusual_values_warn` (fuera de lo observado, fuera del periodo, categoría no vista), `test_predict_with_unknown_values` («No sé» en la distancia y la categoría)
- [x] T006 [US1] `src/olistbr/inference/schema.py` reescrito: `FORM_FIELDS`, `VALID_RANGES` (con el motivo de cada uno), `LABELS` en español, `build_schema`, `load_schema`
- [x] T007 [US1] `src/olistbr/inference/predict.py` reescrito: `InputValidationError`, `load_model`, `validate_orders`, `order_to_features`, `predict_deadline` (plan D1 a D3)

**Checkpoint:** T003 a T005 en verde.

## Phase 3: User Story 1 - La app (Prioridad: P1)

- [x] T008 [US1] `tests/test_app.py` con `AppTest`: sin archivos muestra el aviso de archivos faltantes; con modelo y esquema de juguete (en `tmp_path`, por variables de entorno) muestra el título, el formulario y el aviso de uso responsable; un pedido normal muestra una fecha; un dato imposible muestra un error
- [x] T009 [US1] `app/streamlit_app.py` reescrito (plan D6)

**Checkpoint:** T008 en verde.

## Phase 4: User Story 3 - Exportar y verificar el modelo (Prioridad: P1)

- [x] T010 [US3] `scripts/verificar_modelo.py` (plan D5)
- [x] T011 [US3] `scripts/generadores/generar_notebook_09.py` sobre la plantilla: (1) exportar desde el champion (respaldo `final_pipeline.joblib`) a `models/model.joblib`, sin correo ni enlace en las salidas; (2) `build_schema` con `train.parquet`, `07_metricas_test.json` y `07_segmentos.csv` → `models/input_schema.json`; (3) tres casos: normal, «No sé», imposible; (4) comparación con `final_pipeline.joblib` en 1.000 pedidos del entrenamiento; (5) estado de la ficha
- [x] T012 [US3] Generar y ejecutar `notebooks/09_inferencia.ipynb`; guardar copia de las salidas; comprobar que no tiene correo ni enlace
- [x] T013 [US3] Verificación en Python 3.12: referencia con el `.venv`, entorno temporal `uv` 3.12 con `requirements.txt` y comparación. Si no carga, parar y avisar al grupo

## Phase 5: User Story 4 - Ficha del modelo (Prioridad: P2)

- [x] T014 [US4] Completar `models/ficha_modelo.md` con cifras de `reports/07_*` y `08_*` (sin campos en blanco)

## Phase 6: Polish

- [x] T015 `streamlit run app/streamlit_app.py` en local: abre el formulario de OnTrack y recomienda una fecha (comprobación con la herramienta de ejecución o captura)
- [x] T016 `pytest tests -q` en verde y `grep` sin correo ni enlace en los archivos nuevos
- [x] T017 Revisión con `revisor-de-codigo-ia` y aplicar lo que diga
- [x] T018 Actualizar `reports/bitacora_decisiones.md`, criterios de `especificaciones/09-inferencia-app.md`, `README.md` (cómo correr la app) y `reports/proxima_sesion.md`

## Dependencias

Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5 → Phase 6. T012 depende de T007; T013 de T012; T015 de T009 y T012.
