# Contrato: funciones de la optimización (etapa 6)

Funciones públicas nuevas. Docstrings en español; nombres en inglés. Todas son puras salvo `tune` (registra en MLflow si `log=True`, y escribe en `storage` si se da).

## `src/olistbr/models/tuning.py` (módulo nuevo)

| Firma | Devuelve | Errores | Pruebas (`tests/test_tuning.py`) |
|---|---|---|---|
| `lgbm_quantile_space(trial) -> dict` | `n_estimators` [150, 800], `learning_rate` [0,01, 0,2] log, `num_leaves` [15, 127], `min_child_samples` [20, 300], `subsample` [0,6, 1], `colsample_bytree` [0,5, 1], `reg_lambda` [0,001, 10] log | — | `test_lgbm_space_within_ranges` (con `FixedTrial` en los dos extremos) |
| `hgb_quantile_space(trial) -> dict` | `max_iter` [150, 800], `learning_rate` [0,01, 0,2] log, `max_leaf_nodes` [15, 127], `min_samples_leaf` [20, 300], `l2_regularization` [0,001, 10] log, `max_features` [0,5, 1] | — | `test_hgb_space_within_ranges` |
| `make_lgbm_quantile(params, random_state=42) -> LGBMRegressor` | `objective="quantile"`, `alpha=0.75`, `subsample_freq=1`, `n_jobs=2`, `deterministic=True`, `force_row_wise=True`, `verbose=-1` + `params` | `ValueError` si `params` trae `objective` o `alpha` | `test_lgbm_quantile_is_fixed`, `test_lgbm_rejects_quantile_override` |
| `make_hgb_quantile(params, random_state=42) -> HistGradientBoostingRegressor` | `loss="quantile"`, `quantile=0.75`, `early_stopping=False` + `params` | `ValueError` si `params` trae `loss`, `quantile` o `early_stopping` | `test_hgb_quantile_is_fixed`, `test_hgb_has_no_early_stopping`, `test_hgb_rejects_quantile_override` |
| `FINALISTS: dict[str, Finalist]` | `Finalist(space, make)` para los dos nombres | — | `test_finalists_are_the_two_tournament_names` |
| `starting_candidate(name) -> Candidate` | `candidates()[name]` (familia `"B"`, sin escalar) | `KeyError` claro si el nombre no es finalista | `test_starting_candidate_matches_tournament` (mismos parámetros que `candidates()`) |
| `tuned_candidate(name, params) -> Candidate` | `Candidate(FINALISTS[name].make(params), "B", False)` | igual que arriba | `test_tuned_candidate_is_family_b` |
| `tune(name, X, y, folds, n_trials, seed=42, log=True, storage=None) -> optuna.Study` | estudio que minimiza el costo 3 a 1 medio de `evaluate_candidate_folds`; guarda en cada ensayo las medias y desviaciones (`user_attrs`) y, si `log`, una ejecución hija en MLflow; reanuda si `storage` tiene ensayos | `ValueError` si `n_trials < 1` o `X` no trae `customer_unique_id`; ensayos con costo no finito quedan `FAIL` | `test_tune_runs_requested_trials` (2 ensayos, `log=False`), `test_tune_is_reproducible`, `test_tune_resumes_from_storage`, `test_tune_counts_failed_trials_in_budget` |
| `trials_table(study, name) -> pd.DataFrame` | una fila por ensayo con las columnas de `data-model.md` | — | `test_trials_table_columns` |
| `fit_and_recommend(candidate, train, valid) -> tuple[np.ndarray, np.ndarray, DeadlineModel]` | ajusta con `train` (vía `fit_deadline_model`) y devuelve `(real, plazo recomendado, modelo)` en `valid` | — | `test_fit_and_recommend_does_not_see_valid` |
| `n_trees_used(deadline_model) -> int` | árboles realmente usados (`n_iter_` en HistGradientBoosting, `n_estimators_`/`booster_` en LightGBM) | `TypeError` si el modelo no es un finalista | `test_n_trees_used_reports_early_stopping` |
| `bootstrap_cost_difference(real, plan_a, plan_b, groups, n_boot=1000, seed=42) -> tuple[float, float, float]` | `(diferencia A − B, IC 2,5 %, IC 97,5 %)` con costo 3 a 1, sorteando clientes completos | `ValueError` por largos distintos, clientes vacíos o `n_boot < 1` | `test_identical_plans_give_zero_interval`, `test_constant_difference_collapses_interval`, `test_resamples_customers_not_rows`, `test_bootstrap_is_reproducible`, `test_bootstrap_rejects_bad_input` |
| `decide_params(low, high) -> tuple[str, str]` | `("optimizados" \| "partida", motivo)` | `ValueError` si `low > high` | `test_decide_params_rule` (bajo cero, cruza, toca el cero, sobre cero) |
| `choose_finalist(low, high, name_a, name_b, tie_winner) -> tuple[str, str]` | modelo elegido y motivo (diferencia A − B) | `ValueError` si `tie_winner` no es A ni B | `test_choose_finalist_rule` |

## `src/olistbr/data/split.py` (se agrega una función)

| Firma | Devuelve | Errores | Pruebas (`tests/test_split.py`) |
|---|---|---|---|
| `reserved_forward_split(search, reserved, train_end, valid_start, valid_end, group_col, date_col) -> tuple[DataFrame, DataFrame]` | `(search` con compras ≤ `train_end`, `reserved` con compras entre `valid_start` y `valid_end`), días completos | `ValueError` si las fechas se cruzan o algún lado queda vacío; `AssertionError` si comparten clientes | `test_reserved_forward_split_uses_only_reserved_customers_in_valid`, `test_reserved_forward_split_dates`, `test_reserved_forward_split_rejects_shared_customers` |

Además, en `tests/test_split.py`: `test_reserve_split_25_percent_is_disjoint` (`split_by_group(test_size=0.25, seed=7)` no comparte clientes y aparta ≈ 25 % de ellos).

## Lo que no cambia

`train.py`, `evaluation.py`, `metrics.py` y `tracking/mlflow_utils.py` **no se modifican**; solo se importan. Las 166 pruebas existentes deben seguir pasando.
