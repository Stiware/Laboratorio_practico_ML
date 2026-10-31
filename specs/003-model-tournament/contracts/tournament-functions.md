# Contrato de las funciones del torneo

> **Actualizado el 2026-10-31 tras la implementación y las revisiones:** las firmas y los nombres de pruebas vigentes están en la sección «Versión final» del final de este documento; donde la tabla de arriba difiere, manda la versión final.

Todas llevan docstring en español, no modifican sus entradas y devuelven copias. Los nombres son los que se probarán. La numeración de las pruebas
coincide con `tests/`.

## `src/olistbr/config.py` (se agregan)

| Constante | Valor | Para qué |
|---|---|---|
| `TOURNAMENT_GROUPS` | `("logs", "same_state", "regions", "calendar", "distance_interaction", "ratios", "multi_seller")` | los 7 grupos de variables del torneo (sin `peak_season` ni `history`) |
| `TARGET_QUANTILE` | `COST_UNDERESTIMATE / (COST_UNDERESTIMATE + COST_OVERESTIMATE)` = 0,75 | percentil óptimo para el costo 3 a 1 |
| `CALIBRATION_SIZE` | 0,2 | proporción de clientes apartados para calcular el margen |
| `MIN_WIN_SHARE` | 0,8 | fracción mínima de pliegues que debe ganar el mejor para ser «consistente» |

## `src/olistbr/models/metrics.py` (se agregan)

| Función | Qué hace | Pruebas (`tests/test_metrics.py`) |
|---|---|---|
| `bias(real, predicted)` | media de `real − predicho` (positivo: se subestima en promedio) | `test_bias_sign` |
| `share_late(real, recommended)` | proporción de pedidos con `real > recomendado` | `test_share_late_and_coverage_are_complements` |
| `coverage(real, recommended)` | `1 − share_late` | idem |
| `pinball_loss(real, predicted, quantile=0.75)` | pérdida cuantílica media | `test_pinball_loss_times_four_equals_cost`, `test_pinball_loss_of_perfect_prediction_is_zero` |
| `metrics_by_group(real, predicted, recommended, groups)` | MAE, porcentaje tarde y número de pedidos por grupo (región) | `test_metrics_by_group_splits_by_region`, `test_metrics_by_group_handles_missing_group` |

## `src/olistbr/models/train.py` (se agregan o generalizan)

| Función / clase | Qué hace | Pruebas (`tests/test_models.py`) |
|---|---|---|
| `Candidate` (NamedTuple `model, family, scale`) | un candidato del torneo | `test_candidates_have_expected_names_and_families` |
| `candidates(random_state=42)` | devuelve el diccionario de los 10 candidatos con los parámetros de `research.md` D4 | `test_candidates_have_expected_names_and_families`, `test_candidate_parameters_match_the_spec`, `test_every_candidate_fits_a_tiny_table` |
| `build_pipeline(model=None, groups=(), include_reference=False, scale=True)` | se generaliza: `model=None` sigue siendo `LinearRegression` (resultados idénticos a la etapa 4) | `test_build_pipeline_accepts_any_model`, y las pruebas existentes de la etapa 4 siguen pasando |
| `regression_metrics(real, predicted, recommended=None)` | MAE, RMSE, R², sesgo (de `predicted`) y costo, porcentaje tarde, cobertura (de `recommended`; si es `None`, de `predicted`) | `test_regression_metrics_by_hand`, `test_perfect_predictions_give_zero_error`, `test_regression_metrics_uses_recommended_for_cost` |
| `DeadlineModel` | guarda `pipeline`, `margin`; métodos `predict(X)` y `recommend(X)` | `test_deadline_model_recommend_adds_margin` |
| `fit_deadline_model(candidate, X, y, groups, quantile, calibration_size, seed)` | familia A: aparta clientes, ajusta con el resto y calcula el margen; base y B: ajusta con todo y margen 0 | `test_margin_comes_from_calibration_customers_only`, `test_no_customer_in_fit_and_calibration`, `test_margin_gives_about_75_percent_coverage`, `test_family_b_has_zero_margin_and_uses_all_rows`, `test_margin_can_be_negative`, `test_family_b_quantile_models_cover_about_75_percent` (los modelos de la familia B cubren entre 70 % y 80 % en datos sintéticos) |

## `src/olistbr/models/evaluation.py` (se agregan; `evaluate_folds` y `evaluate_holdout` de la etapa 4 no cambian)

| Función | Qué hace | Pruebas (`tests/test_evaluation.py`) |
|---|---|---|
| `evaluate_candidate_folds(candidate_name, candidate, X, y, folds, groups_col)` | una fila por pliegue con todas las métricas, el margen, los tamaños y `fit_seconds` | `test_evaluate_candidate_folds_returns_all_columns`, `test_evaluate_candidate_folds_uses_same_folds_for_all` |
| `evaluate_candidate_holdout(candidate_name, candidate, train, valid, groups_col)` | lo mismo con una sola partición (hacia adelante) | `test_evaluate_candidate_holdout_fits_only_on_train` |
| `tournament_summary(by_fold, forward)` | media y desviación por modelo y métrica, más el resultado hacia adelante | `test_tournament_summary_means_and_std` |
| `compare_models(best_costs, other_costs)` | diferencia media, pliegues ganados («k de n») y diferencia / ruido; «no evaluable» sin ruido | `test_compare_models_counts_wins`, `test_compare_models_zero_noise_is_not_evaluable`, `test_compare_models_consistent_vs_practical_tie` |
| `rank_table(summary, metrics)` | posiciones (1 = mejor) por métrica | `test_rank_table_orders_by_metric` |
| `beats_baselines(by_fold, model, baselines)` | si el modelo supera a cada línea base por más que su ruido | `test_beats_baselines_requires_more_than_noise` |
| `practical_ties(by_fold, best)` | modelos que no se distinguen del mejor | `test_practical_ties_lists_indistinguishable_models` |
| `per_thousand(difference_per_order)` | multiplica por 1.000 | `test_per_thousand` |

## Casos que se prueban a mano (normal y sucio)

- `share_late`/`coverage`: `real=[10,10,10,10]`, `recomendado=[9,10,11,12]` → tarde 0,25 (solo el primero), cobertura 0,75.
- `pinball_loss`: con `real=10`, predicho 8 (subestima 2) → `0,75 × 2 = 1,5`; predicho 12 (sobrestima 2) → `0,25 × 2 = 0,5`; y el costo respectivo es 6 y 2 (4 veces).
- `fit_deadline_model` (familia A) con un modelo perfecto: margen ≈ 0; con un modelo que siempre subestima 2 días, margen ≈ 2.
- Cobertura: con ruido normal (media 0), margen ≈ 0,674 × desviación → cobertura entre 70 % y 80 %.
- Familia B: `LGBMRegressor(quantile 0,75)` sobre datos sintéticos con ruido asimétrico → cobertura entre 70 % y 80 %.
- `compare_models`: diferencias `[0,5, 0,6, 0,4, 0,5, 0,7]` → gana 5 de 5, diferencia / ruido grande → consistente; diferencias `[0,5, −0,4, 0,3, −0,2, 0,1]` → 3 de 5 y cociente menor que 1 → empate práctico.
- `metrics_by_group` con una región vacía (`NaN`): no falla; la deja aparte.


## Versión final (lo que quedó implementado)

| Cambio respecto a lo planeado | Motivo |
|---|---|
| `build_pipeline(groups=(), include_reference=False, model=None, scale=True)`: `groups` queda **primero** | Para no romper el código de la etapa 4, que lo llama con los grupos en primera posición |
| `fit_deadline_model(pipeline, family, X, y, groups, quantile, calibration_size, seed)`: recibe un **pipeline ya construido** y la familia (`"A"`, `"B"` o `"base"`), no un candidato | Se puede probar el margen con modelos de juguete que controlamos |
| `fit_deadline_model` valida las entradas: `groups` con la misma longitud que `X` y sin vacíos (`ValueError`); advierte si hay menos de 50 clientes de calibración | Casos borde señalados en la revisión |
| `DeadlineModel` suma `predict_with_deadline(X)` → `(predicción, plazo recomendado)` | Evita calcular el margen en dos lugares |
| `evaluate_candidate_folds(name, candidate, X, y, folds, groups_col)` y `evaluate_candidate_holdout(name, candidate, train, valid, groups_col, target)` | Como se planeó; `evaluate_folds` y `evaluate_holdout` de la etapa 4 no cambian |
| `tournament_table(by_fold, forward, baselines)` y `regional_metrics(name, candidate, X, y, folds)` | Se agregaron para que el notebook solo orqueste (tareas T036 y T037) |
| `coverage_by_month(name, candidate, train, valid)` | Se agregó tras la revisión: muestra mes a mes si el plazo sigue cubriendo ≈ 75 % |
| `compare_models` devuelve además `texto` («k de n»), `n` y `evaluable`; `beats_baselines` y `practical_ties` usan `_fold_costs`, que ahora falla con `KeyError` si el modelo no existe | Evita resultados NaN silenciosos |
| Las líneas base van con `family = "base"` (la spec las ubicaba en A y B) | Tienen margen 0 y se comparan aparte |

Nombres reales de las pruebas nuevas (en lugar de los planeados): `test_candidates_have_expected_names_and_families`, `test_candidates_are_fresh_objects_each_time`, `test_candidate_parameters_match_the_spec`,
`test_every_candidate_fits_a_tiny_table`, `test_build_pipeline_accepts_any_model`, `test_regression_metrics_by_hand`, `test_deadline_model_recommend_adds_margin`,
`test_margin_comes_from_calibration_customers_only`, `test_margin_is_exactly_the_75th_percentile_of_calibration_errors`, `test_no_customer_in_fit_and_calibration`,
`test_fit_deadline_model_validates_the_customers`, `test_fit_deadline_model_does_not_modify_its_inputs`, `test_tournament_pipeline_never_receives_identifiers_or_discarded_ideas`,
`test_evaluate_candidate_folds_returns_all_columns`, `test_evaluate_candidate_folds_is_reproducible_and_does_not_touch_the_candidate`, `test_regional_metrics_are_really_out_of_sample`,
`test_coverage_by_month_reports_one_row_per_month`, `test_tournament_table_has_everything_the_report_needs`, `test_compare_models_*`, `test_rank_table_orders_by_metric`.
