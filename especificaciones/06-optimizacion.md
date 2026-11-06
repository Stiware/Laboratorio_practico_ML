# Especificación · Etapa 6 · Optimización de hiperparámetros (Fase 2: lo que corresponde a `tuning.py`)

> Se llena **antes** de pedirle código a Claude. Borrador propuesto por Claude el 5 nov 2026 a partir de
> `material-del-curso/ejemplo-creditsight/src/creditsight/models/tuning.py`, la guía `material-del-curso/kit/guia-modelado-y-optimizacion.md`
> (secciones 3 y 4), los prompts de la etapa 6 (`kit/prompts-por-etapa.md`) y los resultados de `reports/05_torneo.md`.
> **CreditSight era clasificación; OnTrack es regresión** (predice el plazo del percentil 75), así que cada pieza del ejemplo se adapta (sección 1).
> Los tiempos de la sección 4 son **estimaciones** hechas con los tiempos medidos en el torneo; no se midió nada nuevo para este borrador.

**Grupo:** _______________   **Fecha:** 5 nov 2026   **Estado:** ☐ borrador  ☒ cerrada _(el grupo aceptó todas las propuestas de Q1 a Q8 el 5 nov 2026)_

## 1. Del ejemplo (clasificación) a OnTrack (regresión)

| Pieza de `tuning.py` del ejemplo | Qué hace en CreditSight | Qué haremos en OnTrack |
|---|---|---|
| `lgbm_space`, `rf_space` | rangos de búsqueda de LightGBM y del bosque | `lgbm_quantile_space` y `hgb_quantile_space`: rangos de los **dos finalistas** (sección 4). El objetivo cuantílico (`alpha` = 0,75) **queda fijo** y no se busca |
| `make_lgbm`, `make_rf` | crean el modelo con los parámetros del ensayo | `make_lgbm_quantile`, `make_hgb_quantile`: crean el modelo con los parámetros del ensayo **y** el percentil 75 fijo |
| `MODELS` | nombre → (espacio, constructor, si escala) | `FINALISTS`: nombre → (espacio, constructor, **parámetros de partida** = los del torneo) |
| `OBJETIVOS` (9 métricas para elegir) | se puede optimizar cualquier métrica | **Una sola: el costo 3 a 1** (minimizar), la misma que decidió el torneo. Las demás se registran, pero no se optimizan (Q2) |
| `tune(...)` con Optuna + MLflow anidado | un estudio por modelo; cada ensayo es una ejecución hija en MLflow | Igual, reutilizando `evaluate_candidate_folds` de la etapa 5 (devuelve costo, MAE, RMSE, R², sesgo, tarde, cobertura y tiempo por pliegue) |
| `bootstrap_cost_difference` | diferencia de costo con intervalo del 95 % **remuestreando clientes** | Igual, pero sobre el **plazo recomendado** (`mean_delivery_cost` de `models/metrics.py`) en vez de una matriz de costos |
| `bootstrap_metric_ci`, `bootstrap_metric_difference` | intervalo de cualquier métrica | Solo se agrega si hace falta para el informe (por ejemplo, la cobertura con su intervalo). No es necesario para decidir |
| — (no existe en el ejemplo) | — | `decide_params(low, high)`: aplica la regla escrita (sección 3) y devuelve `"optimizados"` o `"partida"` con su motivo, para que la regla esté en código probado y no solo en el notebook |

## 2. Preguntas y decisiones del grupo (5 nov 2026)

**Decisión:** el grupo aceptó **todas** las propuestas de Claude (columna «Propuesta») el 5 nov 2026, incluidos los temas que pasan a la etapa 7.

| # | Pregunta | Por qué importa | Propuesta de Claude |
|---|---|---|---|
| Q1 | **¿Qué finalistas se optimizan?** | El torneo descarta, no corona: la optimización puede cambiar el orden de los dos primeros (están separados por 0,36 % del costo) | **LightGBM cuantil 0,75 y HistGradientBoosting cuantil 0,75**, los dos que da la regla aprobada en el torneo. LightGBM de valor típico (familia A) **no** entra: no lo respalda la regla y obligaría a resolver antes el control de datos entre familias. Como los dos finalistas son de la familia B, ese control **no aplica en esta etapa**, y así se anota |
| Q2 | **¿Qué métrica se optimiza?** | La guía pide optimizar con la métrica con la que se eligió a los finalistas | **El costo 3 a 1 del plazo recomendado** (media de los pliegues). MAE, RMSE, R², sesgo, tarde y cobertura se registran en cada ensayo, pero no deciden. Advertencia: en la familia B el MAE sale más alto **por diseño** (predice el percentil 75) |
| Q3 | **¿Cuántos ensayos por finalista (presupuesto)?** | Más ensayos sobreajustan a la propia validación y consumen tiempo de clase. La guía sugiere de 12 a 20 | **20 ensayos por finalista**, fijados antes de empezar. Tiempo estimado: LightGBM ≈ 5 a 10 min y HistGradientBoosting ≈ 15 a 25 min (sección 4). Si el grupo prefiere menos tiempo, 15 ensayos (la cifra que quedó en `reports/proxima_sesion.md`) |
| Q4 | **¿Cómo se reservan clientes que la búsqueda no ve?** | La mejor puntuación de una búsqueda es optimista (el máximo de muchas estimaciones con ruido). Se comprueba en clientes que la búsqueda nunca vio | **25 % de los clientes de `train.parquet`** con `split_by_group(test_size=0.25, seed=7)` (`GroupShuffleSplit` por `customer_unique_id`, como dice la guía), con `assert` de que no comparten clientes. La búsqueda usa el 75 % restante, con **5 pliegues nuevos** por cliente (`make_group_folds`, semilla 42) sobre ese 75 %. Los pliegues de la etapa 4 no se pueden usar tal cual, porque incluyen a los clientes reservados |
| Q5 | **¿Qué hacemos con la parada temprana de HistGradientBoosting?** En el torneo quedó en su valor por defecto (`early_stopping="auto"`), que **con más de 10.000 filas se activa sola**: aparta un 10 % de las **filas** (no de los clientes) del entrenamiento y deja de agregar árboles antes de llegar a 300 | No es fuga hacia la validación (ocurre dentro del entrenamiento de cada pliegue), pero (a) el número de árboles no era el que creíamos, (b) ese 10 % se aparta por fila, no por cliente, y (c) LightGBM no la tiene, así que los dos finalistas no se trataron igual | **Parámetros de partida = exactamente los del torneo** (con la parada temprana como estaba), para que la comparación «partida contra optimizado» sea contra lo que realmente se midió. **En la búsqueda, `early_stopping=False`** y se busca el número de árboles (`max_iter`) de forma explícita, igual que en LightGBM. Se reporta cuántos árboles usó realmente el modelo de partida |
| Q6 | **¿Se usa una prueba hacia adelante en esta etapa?** | En las etapas 4 y 5 sirvió de guardia contra la tendencia entre años. Pero la del torneo ya no es limpia: sus meses de validación (marzo a agosto de 2018) incluyen clientes que la búsqueda vio | **Sí, solo informativa y sin clientes de la búsqueda:** entrenar con los clientes de búsqueda que compraron hasta el 2018-02-28 y validar con los **clientes reservados** que compraron del 2018-03-01 al 2018-08-31. Es una sola partición sin ruido medido: **no cambia la decisión de la regla**, pero si el optimizado empeora hacia adelante se avisa en el informe |
| Q7 | **¿Cómo se elige el modelo final entre los dos finalistas?** | Al terminar quedan dos modelos, cada uno con sus parámetros elegidos (de partida u optimizados) | Se comparan **en los clientes reservados** con el mismo intervalo del 95 % por cliente. Si todo el intervalo queda de un lado del cero, gana ese modelo. **Si cruza el cero es un empate práctico y gana LightGBM cuantil 0,75** (el más rápido: 3,1 s contra 7,8 s por ajuste, y el ganador del torneo). Esta regla se escribe aquí, antes de mirar |
| Q8 | **¿Dónde se registra el seguimiento?** `CLAUDE.md` (regla 7) dice MLflow en **Databricks**; la guía dice que en la clase 3 era local | Si Databricks no conecta, la etapa no debe detenerse | **`configure_mlflow()` sin `force_local`**: usa Databricks si `.env` tiene la conexión y es alcanzable; si no, cae solo a `mlflow_local.db` (ya está en `.gitignore`) y lo dice en pantalla. El experimento se llama `ontrack-optimizacion`. Se usa el módulo que ya existe (`src/olistbr/tracking/mlflow_utils.py`), **sin reescribirlo**. El `.env` no se abre ni se imprime |

**Temas pendientes del torneo que esta etapa NO resuelve** (propuesta: dejarlos para la especificación de la etapa 7, con su motivo):

- **Recalibrar con datos recientes** (pendiente 2 de `reports/proxima_sesion.md`): es una decisión de uso del modelo y de la app, no de hiperparámetros. Mezclarla aquí haría que la comprobación honesta mida dos cosas a la vez.
- **Marca de fin de año (F6)** y **historial por vendedor (F8):** son variables, no hiperparámetros. Se mantienen fuera, igual que en el torneo (los mismos 7 grupos de variables, `TOURNAMENT_GROUPS`).
- **Indicador por región:** se reporta la tabla por región del modelo elegido en los clientes reservados (principio VIII), sin decidir nada con ella.

## 3. Reglas escritas antes de medir

1. **Datos:** solo `data/processed/train.parquet`. La prueba final (`test.parquet`) **no se abre** (principio III).
2. **Reserva:** 25 % de los clientes (semilla 7). La búsqueda **nunca** ve a esos clientes, ni en los pliegues ni en el preprocesador. **Única excepción conocida:** la lista de vendedores frecuentes, que viene de la etapa 3 y se contó con todo el entrenamiento (solo conteos de ítems, sin el objetivo; ver «Riesgos»).
3. **Búsqueda:** Optuna con muestreador TPE (semilla 42), **20 ensayos por finalista** (Q3), minimizando el costo 3 a 1 medio de 5 pliegues por cliente sobre el 75 % de búsqueda. Cada ensayo queda registrado en MLflow como ejecución anidada.
4. **No se reporta la mejor puntuación de la búsqueda como resultado.** Se muestra marcada como «optimista».
5. **Comprobación honesta, por finalista:** se ajustan **dos** modelos con todo el 75 % de búsqueda (parámetros de partida y parámetros optimizados) y se evalúan en los clientes reservados. Se calcula `diferencia = costo(optimizado) − costo(partida)` pedido a pedido y un **intervalo del 95 % remuestreando clientes completos** (1.000 remuestreos, semilla 42).
6. **Regla de decisión:** «Uso los parámetros optimizados **solo si todo el intervalo está por debajo de cero**. Si el intervalo cruza el cero (o queda por encima), la mejora no se distingue del ruido y uso los de partida.»
7. **Elección entre finalistas:** la regla de Q7.
8. **Todo lo que aprende de los datos va dentro del `Pipeline`** y se ajusta solo con el entrenamiento de cada pliegue (principio III).
9. **Las decisiones se guardan** en `reports/06_best_params.json` (finalista, parámetros de partida, parámetros optimizados, diferencia e intervalo, decisión y motivo), y se explican en `reports/06_optimizacion.md`.

## 4. Espacio de búsqueda (conocimiento del algoritmo, no azar)

Partida = parámetros del torneo (`candidates()` en `src/olistbr/models/train.py`). El percentil 75 (`alpha` / `quantile` = 0,75) **queda fijo** en todos los ensayos.

### 4.1 LightGBM cuantil 0,75 (partida: 300 árboles, tasa 0,05, 31 hojas; el resto con los valores por defecto)

| Parámetro | Rango | Escala | Motivo |
|---|---|---|---|
| `n_estimators` | 150 a 800 | entera | Con tasas bajas hacen falta más árboles; 800 acota el tiempo |
| `learning_rate` | 0,01 a 0,2 | logarítmica | Rango típico de la guía; tasa y número de árboles se compensan |
| `num_leaves` | 15 a 127 | entera | Controla la complejidad de cada árbol; 31 (partida) queda en el medio |
| `min_child_samples` | 20 a 300 | entera | Un percentil necesita más pedidos por hoja que una media para ser estable; el valor por defecto (20) puede quedarse corto |
| `subsample` (con `subsample_freq=1`) | 0,6 a 1,0 | continua | Muestrear filas reduce el sobreajuste |
| `colsample_bytree` | 0,5 a 1,0 | continua | Hay ≈ 100 columnas codificadas, muchas casi constantes |
| `reg_lambda` | 0,001 a 10 | logarítmica | Regularización L2 de las hojas |

Fijos: `objective="quantile"`, `alpha=0.75`, `random_state=42`, `n_jobs=2`, `deterministic=True`, `force_row_wise=True`, `verbose=-1`.

### 4.2 HistGradientBoosting cuantil 0,75 (partida: `max_iter` 300, tasa 0,05, parada temprana por defecto; ver Q5)

| Parámetro | Rango | Escala | Motivo |
|---|---|---|---|
| `max_iter` | 150 a 800 | entera | Número de árboles, explícito porque la parada temprana se apaga (Q5) |
| `learning_rate` | 0,01 a 0,2 | logarítmica | Igual que LightGBM |
| `max_leaf_nodes` | 15 a 127 | entera | Equivalente a `num_leaves` |
| `min_samples_leaf` | 20 a 300 | entera | Equivalente a `min_child_samples` |
| `l2_regularization` | 0,001 a 10 | logarítmica | Equivalente a `reg_lambda` |
| `max_features` | 0,5 a 1,0 | continua | Proporción de columnas por división (disponible en scikit-learn 1.9.1) |

Fijos: `loss="quantile"`, `quantile=0.75`, `early_stopping=False`, `random_state=42`.

### 4.3 Tiempo estimado (con los tiempos del torneo; no medido para este borrador)

| Finalista | Ajuste en el torneo (61.500 filas) | Por ensayo (5 pliegues × ≈ 46.000 filas) | 20 ensayos |
|---|---:|---:|---:|
| LightGBM cuantil 0,75 | 3,1 s | ≈ 12 a 30 s | ≈ 5 a 10 min |
| HistGradientBoosting cuantil 0,75 | 7,8 s | ≈ 30 a 75 s | ≈ 15 a 25 min |

El rango es amplio porque los ensayos con 800 árboles y 127 hojas tardan varias veces más que la partida. Con el estudio guardado en disco (`storage`, como en el ejemplo) la búsqueda se puede **reanudar** si se corta.

## 5. Métricas

Las mismas del torneo (`regression_metrics`): **costo 3 a 1 (decide)**, MAE, RMSE, R², sesgo, porcentaje tarde, cobertura y tiempo de ajuste. Además:

- **Margen y árboles realmente usados** por el modelo de partida de HistGradientBoosting (Q5).
- **Tabla por región del cliente** del modelo elegido, en los clientes reservados (principio VIII; solo se reporta).
- **Cobertura mes a mes** en la prueba hacia adelante con clientes reservados (Q6), para seguir el hallazgo del torneo (61 % a 93 %).

## 6. Código previsto (misma forma que `tuning.py` del ejemplo)

En un módulo nuevo, `src/olistbr/models/tuning.py`. Reutiliza lo que ya existe (`split_by_group`, `make_group_folds`, `build_pipeline`, `Candidate`,
`evaluate_candidate_folds`, `fit_deadline_model`, `mean_delivery_cost`, `log_run`) y **no reescribe** `mlflow_utils.py`.

| Función | Qué hace | Pruebas (`tests/test_tuning.py`) |
|---|---|---|
| `lgbm_quantile_space(trial)`, `hgb_quantile_space(trial)` | devuelven los parámetros de un ensayo dentro de los rangos de la sección 4 | `test_spaces_stay_within_ranges` (con `optuna.trial.FixedTrial` en los extremos) |
| `make_lgbm_quantile(params)`, `make_hgb_quantile(params)` | crean el modelo con el percentil 75 fijo | `test_quantile_objective_cannot_be_overridden` (aunque `params` traiga otro `alpha`, sigue en 0,75 o falla con un mensaje claro), `test_hgb_has_no_early_stopping_in_search` |
| `FINALISTS` | nombre → (espacio, constructor, parámetros de partida) | `test_starting_params_match_tournament` (la partida es igual a `candidates()`) |
| `tune(name, X, y, folds, n_trials, seed, log, storage)` | estudio de Optuna que minimiza el costo; cada ensayo como ejecución anidada en MLflow | `test_tune_runs_requested_trials_on_tiny_table` (2 ensayos, `log=False`), `test_tune_is_reproducible_with_seed`, `test_tune_resumes_from_storage` |
| `bootstrap_cost_difference(real, plan_a, plan_b, groups, n_boot, seed)` | diferencia de costo (A − B) con intervalo del 95 %, **remuestreando clientes completos** | `test_identical_plans_give_zero_interval`, `test_constant_difference_collapses_interval`, `test_resamples_customers_not_rows` (un cliente con muchos pedidos pesa como un bloque), `test_bootstrap_is_reproducible`, entradas de distinto largo fallan |
| `decide_params(low, high)` | aplica la regla 6 y devuelve `("optimizados" \| "partida", motivo)` | `test_decide_uses_optimized_only_if_whole_interval_below_zero` (por debajo, cruza el cero, por encima, en cero exacto) |

La separación de clientes reservados reutiliza `split_by_group(test_size=0.25, seed=7)`; se agrega una prueba de que con esos valores los dos lados no comparten clientes.

## 7. Entregables

- `notebooks/06_optimizacion.ipynb` (plantilla del curso; **solo orquesta**). La plantilla lee `train.joblib`: se adapta para leer `train.parquet` (pendiente 6 de `reports/proxima_sesion.md`).
- `reports/06_optimizacion.md` (plantilla ya existente): espacio, presupuesto, reserva, dónde quedó el seguimiento, resultados de la búsqueda (marcados como optimistas), comprobación honesta, decisión y lo que no se pudo concluir.
- `reports/06_best_params.json` y `reports/06_ensayos.csv` (una fila por ensayo con sus parámetros y métricas).
- Línea nueva en `reports/bitacora_decisiones.md` y en la sección 7 de `especificacion-proyecto.md`.

## 8. Criterios de cierre

- ☒ Q1 a Q8 respondidas por escrito y confirmadas por el grupo (5 nov 2026).
- ☒ `notebooks/06_optimizacion.ipynb` corre de principio a fin y solo orquesta (la lógica está en `src/olistbr/models/tuning.py`).
- ☒ Clientes reservados y de búsqueda sin clientes en común (`assert` en el notebook y prueba en `tests/`).
- ☒ En MLflow están **todos** los ensayos (20 por finalista), cada uno con sus parámetros, métricas y etiquetas `etapa=06_optimizacion` y `modelo`; se respetó el presupuesto.
- ☒ Comprobación honesta en clientes reservados con intervalo por cliente; decisión según la regla 6, escrita antes de mirar.
- ☒ Modelo elegido entre los dos finalistas según Q7, con su motivo.
- ☒ `pytest tests` en verde y la prueba final **no se abrió** (`grep` de `test.parquet` da 0 en el notebook y en `tuning.py`).
- ☒ Revisión con `auditor-de-fuga` y `revisor-de-codigo-ia` (resumen en `reports/06_optimizacion.md`).
- ☐ Quien **no** escribió el código sabe responder las preguntas 7 y 8 de la guía (§5): cuántos ensayos y por qué, qué clientes se reservaron, qué regla se usó y qué dio.

## 9. Riesgos que conviene conocer

- **La optimización puede no servir**, y eso es un resultado válido: los parámetros del torneo ya eran razonables y la ventaja entre finalistas es pequeña. Si el intervalo cruza el cero, se usan los de partida y se escribe así.
- **Menos datos para la búsqueda:** la búsqueda ajusta con ≈ 46.000 pedidos por pliegue (antes ≈ 61.500); los costos no son comparables uno a uno con los del torneo.
- **La reserva se usa dos veces** (partida contra optimizado, y después entre finalistas). Sigue siendo validación (no es la prueba), pero el resultado del modelo elegido en la reserva es algo optimista; la estimación honesta sale de la prueba de la etapa 7.
- **Lista de vendedores frecuentes:** se calculó en la etapa 3 con todo el entrenamiento, incluidos los clientes que ahora se reservan. Son solo conteos de ítems por vendedor (no usan el objetivo), así que el riesgo es bajo; se anota como limitación y se pide al `auditor-de-fuga` que lo confirme.
- **Calibración en el tiempo:** la optimización no corrige que la cobertura cambie de mes a mes (hallazgo del torneo); eso queda para la etapa 7.
- **Tiempos:** son estimaciones; el equipo del grupo tuvo carga variable en el torneo.
- **Entorno:** el `.venv` de esta máquina usa Python 3.14 y el proyecto pide 3.12; además, aquí `data/raw/` y `data/processed/` están vacíos. Para correr el notebook hay que copiar los CSV de Olist y volver a generar `train.parquet` con el notebook 03.
