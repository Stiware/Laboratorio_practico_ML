# Contrato de las funciones de la etapa 4

Todas devuelven copias, no modifican sus entradas y llevan docstring en español. Los nombres son los que se probarán.

## `src/olistbr/features/engineering.py`

| Función / constante | Entrada → salida | Pruebas (`tests/test_features.py`) |
|---|---|---|
| `add_features(df)` | tabla de `train.parquet` → copia con las columnas derivadas de `data-model.md` | `test_log_features`, `test_same_state`, `test_regions_table`, `test_calendar_features`, `test_year_end_peak_window`, `test_distance_interaction`, `test_route`, `test_ratios_and_density`, `test_multi_seller` |
| | división por cero o vacío → NaN, nunca `inf`; no muta la entrada | `test_division_by_zero_or_missing_gives_nan_not_inf`, `test_add_features_does_not_mutate_input` |
| `black_friday(year)` | año → fecha del cuarto viernes de noviembre | `test_black_friday_dates` (2017-11-24) |
| `feature_columns(groups=(), include_reference=False)` | grupos activos → columnas numéricas, categóricas y con historial (sin repetir) | `test_feature_columns_toggles`, `test_unknown_group_raises` |
| | `promised_days` solo con `include_reference=True` | `test_promised_days_never_in_any_group_combination` |
| `build_preprocessor(groups=(), include_reference=False, scale=True)` | → `ColumnTransformer` listo para ajustar | `test_preprocessor_handles_unseen_category_and_missing`, `test_preprocessor_output_has_no_missing_or_infinite`, `test_history_group_uses_target_encoder` |
| `BASE_NUMERIC`, `BASE_CATEGORICAL`, `GROUPS`, `REGION_BY_STATE`, `REFERENCE_ONLY_COLUMNS` | constantes | `test_region_table_covers_27_states` |

## `src/olistbr/models/train.py`

| Función | Entrada → salida | Pruebas (`tests/test_models.py`) |
|---|---|---|
| `build_pipeline(groups=(), include_reference=False)` | → `Pipeline` con los pasos `prep` (preprocesador) y `model` (`LinearRegression`) | `test_pipeline_fits_and_predicts`, `test_pipeline_is_fit_only_on_train` |

## `src/olistbr/data/split.py` (se agregan)

| Función | Entrada → salida | Pruebas (`tests/test_split.py`) |
|---|---|---|
| `make_group_folds(df, n_splits=5, group_col, seed)` | tabla → lista de `(índices_entrenamiento, índices_validación)` con `GroupKFold` mezclado | `test_group_folds_do_not_share_customers`, `test_group_folds_cover_each_row_once`, `test_group_folds_are_reproducible` |
| `forward_split(df, train_end, valid_start, valid_end, group_col, date_col)` | tabla → `(train, valid)`; entrenamiento sin los clientes de la validación | `test_forward_split_respects_dates`, `test_forward_split_has_no_shared_customers` |

## `src/olistbr/models/evaluation.py`

| Función | Entrada → salida | Pruebas (`tests/test_evaluation.py`) |
|---|---|---|
| `evaluate_folds(pipeline, X, y, folds)` | pipeline sin ajustar y pliegues → tabla con una fila por pliegue (`mae`, `cost`, `n_train`, `n_valid`); clona el pipeline en cada pliegue | `test_evaluate_folds_returns_one_row_per_fold`, `test_evaluate_folds_does_not_leak_validation_into_fit` |
| `evaluate_holdout(pipeline, train, valid, target)` | pipeline sin ajustar y una partición (p. ej. la prueba hacia adelante) → `{mae, cost, n_train, n_valid}` | `test_evaluate_holdout_fits_only_on_train` |
| `compare_to_reference(reference, variant)` | dos tablas por pliegue → mejora, ruido y cociente para el error y el costo | `test_improvement_and_noise_are_paired_differences`, `test_zero_noise_is_not_evaluable` |
| `decide_idea(cv, forward, known_at_prediction, min_ratio=1.0)` | comparaciones → `("conservar" \| "descartar", motivo)` | `test_decide_keeps_idea_that_beats_noise`, `test_decide_discards_when_ratio_below_one`, `test_decide_discards_when_forward_flips_sign`, `test_decide_discards_when_not_known_at_prediction`, `test_decide_keeps_when_forward_not_evaluable` |

## `src/olistbr/config.py` (se agregan)

`N_FOLDS = 5`, `FORWARD_TRAIN_END = "2018-02-28"`, `FORWARD_VALID_START = "2018-03-01"`, `FORWARD_VALID_END = "2018-08-31"`,
`MIN_CATEGORY_FREQUENCY = 0.01`, `MIN_IMPROVEMENT_RATIO = 1.0`.

## Casos que se prueban a mano (normal y sucio)

- Logaritmos: `log1p(0) = 0`; valor vacío o negativo → NaN.
- Mismo estado: `SP/SP → 1`, `SP/RJ → 0`, vacío → NaN.
- Regiones: `SP → Sudeste`, `AM → Norte`, `"XX" → NaN`.
- Calendario: lunes `→ 0` y no marcado; jueves, viernes y sábado marcados; domingo no marcado.
- Temporada alta 2017: `2017-11-19 → 0`, `2017-11-20 → 1`, `2017-12-31 → 1`, `2018-01-01 → 0`.
- Razón flete/precio con precio 0 o vacío → NaN.
- Varios vendedores: `n_items=3, main_seller_items=2 → 1`; `3 y 3 → 0`.
- Preprocesador: categoría nueva de vendedor o ruta y datos vacíos → matriz sin NaN.
- `TargetEncoder`: un cliente de validación con el vendedor `s1` recibe el promedio de entrenamiento de `s1`, no su propio objetivo.
- Decisión: `mejora/ruido = 0,8` → descartar; `1,5` con adelante positivo → conservar; `1,5` con adelante negativo → descartar; `1,5` sin
  datos hacia adelante → conservar con motivo «no evaluable hacia adelante»; no conocida al predecir → descartar.

## Funciones añadidas durante la implementación y la revisión (2026-10-31)

| Módulo | Función | Qué hace | Prueba |
|---|---|---|---|
| `models/train.py` | `build_baseline_pipeline()` | línea base trivial: predice siempre la mediana del entrenamiento | `test_baseline_pipeline_predicts_the_training_median` |
| `models/evaluation.py` | `measure_reference(pipeline, X, y, folds, forward)` | mide la referencia con los pliegues y con la prueba hacia adelante | `test_reference_row_has_all_columns` |
| `models/evaluation.py` | `reference_row(name, reference)` | fila de la referencia en la tabla | `test_reference_row_has_all_columns` |
| `models/evaluation.py` | `measure_idea(name, group, pipeline, X, y, folds, reference, forward, *, known_at_prediction, forward_evaluable, forward_check_column, kind)` | mide una idea y devuelve su fila; `kind` = `idea`, `referencia_no_usable`, `comprobacion` o `linea_base`; `forward_check_column` informa «no evaluable hacia adelante» si esa columna no varía en la validación | `test_measure_idea_row_for_a_useful_idea`, `test_measure_idea_detects_when_the_forward_validation_lacks_the_phenomenon`, `test_measure_idea_reference_only_is_never_kept`, `test_measure_idea_baseline_is_not_decided` |
| `models/evaluation.py` | `kept_groups(rows)`, `results_table(rows)` | grupos conservados; tabla final sin ideas repetidas | `test_kept_groups_and_results_table` |
| `data/split.py` | `forward_split` valida argumentos | falla con `ValueError` si el entrenamiento no termina antes de la validación o si la validación queda vacía | `test_forward_split_rejects_overlapping_dates`, `test_forward_split_rejects_empty_validation` |
| `features/engineering.py` | `_log1p` y `_ratio` | un infinito de entrada también da vacío | `test_infinite_inputs_give_missing_not_infinite` |
