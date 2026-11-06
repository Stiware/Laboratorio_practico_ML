# Contrato: funciones de la validación final y el registro (etapas 7 y 8)

Docstrings en español; nombres en inglés. Todas son puras salvo `open_test_once` (lee y escribe archivos) y las de registro (escriben en MLflow).

## `src/olistbr/models/validation.py` (nuevo)

| Firma | Devuelve | Errores | Pruebas (`tests/test_validation.py`) |
|---|---|---|---|
| `open_test_once(path, mark_path) -> pd.DataFrame` | la tabla de prueba; escribe en `mark_path` la fecha y hora de apertura | `RuntimeError` si `mark_path` existe (dice cuándo se abrió) | `test_test_set_opens_only_once`, `test_mark_records_date_and_time` (archivos temporales) |
| `bootstrap_mean(values, groups, n_boot=1000, seed=42) -> tuple` | `(media por pedido, p2,5, p97,5)` sorteando clientes completos | `ValueError` por largos distintos, vacíos, no finitos, clientes vacíos o `n_boot < 1` | `test_bootstrap_mean_constant`, `test_bootstrap_mean_resamples_customers`, `test_bootstrap_mean_reproducible`, `test_bootstrap_mean_rejects_bad_input` |
| `baseline_plans(train, test, target, purchase_col, promised_col) -> dict` | `{"Plazo fijo (percentil 75)": ..., "Siempre la mediana": ..., "Plazo prometido por Olist": ...}` (un plazo por pedido de `test`) | `ValueError` si falta una columna | `test_baseline_plans_by_hand` |
| `test_report(real, plans, groups, main_baseline, model_name) -> pd.DataFrame` | una fila por regla: costo, % tarde, cobertura (cada uno con su intervalo), MAE, sesgo; y para el modelo, la diferencia de costo contra `main_baseline` con su intervalo y `mejora_con_confianza` | `KeyError` si faltan `model_name` o `main_baseline` en `plans` | `test_test_report_matches_manual_metrics`, `test_test_report_flags_confident_improvement` |
| `slack_and_delay(real, plan) -> dict` | `holgura_media`, `atraso_medio`, `pct_a_tiempo`, `pct_tarde` | `ValueError` por entradas inválidas | `test_slack_and_delay_by_hand` |
| `segment_table(real, plan, segments: dict) -> pd.DataFrame` | por segmento (`region`, `mes`) y grupo: pedidos, % tarde, cobertura, MAE, holgura, atraso | — | `test_segment_table_by_hand` |
| `cost_sensitivity(real, plans, ratios=COST_RATIOS) -> pd.DataFrame` | costo por regla y proporción | — | `test_ratio_one_is_mae` |
| `top_shap(pipeline, X, n=10, sample=2000, seed=42) -> pd.DataFrame` | `variable`, `shap_abs_medio`, `shap_medio`, ordenado | `TypeError` si el modelo no es de árboles | `test_top_shap_finds_the_only_useful_feature` (LightGBM de juguete) |

## `src/olistbr/tracking/registry.py` (nuevo; usa `mlflow_utils.py` sin reescribirlo)

| Firma | Devuelve | Errores | Pruebas (`tests/test_registry.py`, MLflow local en carpeta temporal) |
|---|---|---|---|
| `input_example(df, n=3) -> pd.DataFrame` | n filas solo con las 25 columnas del modelo | `ValueError` si falta alguna columna | `test_input_example_has_no_identifiers` |
| `register_final_model(pipeline, params, metrics, example, name, experiment, run_name="modelo-final", tags=None) -> dict` | `{"run_id", "nombre", "version"}` | — | `test_register_creates_a_version` (modelo de juguete) |
| `set_champion(full_name) -> int` | la versión a la que apunta `champion` (la última) | `ValueError` si no hay versiones | `test_champion_points_to_latest_version` |
| `load_champion(full_name)` | el pipeline recuperado | — | `test_round_trip_predicts_the_same` |

## Lo que no cambia

`train.py`, `evaluation.py`, `metrics.py`, `tuning.py`, `data/split.py`, `tracking/mlflow_utils.py` e `inference/` **no se modifican**; solo se importan.
