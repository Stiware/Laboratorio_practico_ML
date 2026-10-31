# Implementation Plan: Torneo de modelos de OnTrack (etapa 5, Fase 1)

**Branch**: `main` | **Date**: 2026-10-31 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/003-model-tournament/spec.md`

## Summary

Se agregan los 10 candidatos del torneo (2 líneas base, 5 modelos de la familia A y 3 de la familia B) con la misma forma que `train.py` del ejemplo
CreditSight, adaptada de clasificación a **regresión**: `candidates`, un `build_pipeline` generalizado, `regression_metrics` (en lugar de
`metrics_from_proba`) y la evaluación por pliegues. La novedad frente al ejemplo es el **modelo de plazo** (`DeadlineModel`): la familia A predice
un valor típico y le suma un **margen** (percentil 75 de los errores en clientes de calibración apartados dentro de cada entrenamiento); la familia B
predice el percentil 75 directamente. El **costo 3 a 1** decide al ganador (es 4 veces la pérdida cuantílica de 0,75) y se reportan MAE, RMSE, R², sesgo,
porcentaje tarde, cobertura y tiempo. Se usan los pliegues por cliente de la etapa 4 y la prueba hacia adelante. Las funciones de comparación por
pares, posiciones y superación de líneas base viven en `models/evaluation.py`, con pruebas. El notebook 05 solo orquesta y escribe las tablas.

## Technical Context

**Language/Version**: Python 3.12.

**Primary Dependencies**: pandas 3.0.6, numpy 2.5.4, scikit-learn 1.9.1, LightGBM 4.7.0, XGBoost 3.4.1. Todas ya están en `requirements.txt` y en el
`.venv`; **no se agrega ninguna dependencia nueva**.

**Storage**: archivos. Entrada `data/processed/train.parquet` (solo lectura). Salidas: `reports/05_torneo_por_pliegue.csv`,
`reports/05_torneo_resumen.csv`, `reports/05_torneo_tiempos.csv`, `reports/05_torneo_regiones.csv` y `reports/05_torneo.md`.

**Testing**: pytest 9.1.1 con `pytest tests`; casos hechos a mano con tablas pequeñas y datos sintéticos de percentil conocido; sin leer `data/processed/`.

**Target Platform**: Linux/Windows local del grupo (4 núcleos).

**Project Type**: biblioteca (`src/olistbr/`) más un notebook que orquesta.

**Performance Goals**: el torneo completo (10 candidatos × 5 pliegues + prueba hacia adelante) en unos 5 a 6 minutos. Tiempos medidos de un ajuste en un pliegue
(61.571 filas × 100 columnas): lineal 3,0 s, Ridge 1,5 s, LightGBM 2,8 s, XGBoost 6,1 s, LightGBM cuantil 3,4 s, XGBoost cuantil 11,6 s, HistGradientBoosting
cuantil 13,0 s, bosque liviano 18,4 s.

**Constraints**: nunca abrir `test.parquet`; todo lo que aprende de los datos (preprocesador, modelo, margen) dentro de cada pliegue; el plazo prometido por Olist no entra al
modelo; la marca de fin de año (F6) no entra al torneo; textos en español y nombres de código en inglés.

**Scale/Scope**: 76.924 pedidos de entrenamiento, 74.425 clientes; 10 candidatos; 5 pliegues + 1 partición hacia adelante.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principio | Cómo lo cumple este plan | ¿Pasa? |
|---|---|---|
| I. Datos originales no se tocan | Solo lee `train.parquet`; no toca `data/raw/` | Sí |
| II. Se parte por la unidad de análisis | Pliegues y partición hacia adelante por `customer_unique_id`; la calibración del margen también (`GroupShuffleSplit` por cliente) con `assert_disjoint_groups` | Sí |
| III. La prueba se abre una vez | No se lee `test.parquet`; el margen sale de clientes apartados del entrenamiento, nunca de la validación; todo dentro del pliegue | Sí |
| IV. Decisión con motivo, función y prueba | Cada función nueva con su prueba; cada decisión del torneo con cifra y motivo en `reports/05_torneo.md` | Sí |
| V. Nada sensible sale | No se usan identificadores como entrada; `customer_unique_id` solo agrupa | Sí |
| VI. El código es nuestro | Funciones cortas con docstring en español, forma del ejemplo; revisión con los subagentes | Sí |
| VII. No se inventan datos | Las cifras salen del notebook ejecutado; la diferencia se expresa en días y pedidos, **sin inventar un valor en dinero** | Sí |
| VIII. Uso responsable | MAE y porcentaje tarde por región del cliente para los finalistas; limitaciones temporales heredadas de la etapa 4 | Sí |

**Resultado:** sin violaciones; nada en «Complexity Tracking».

**Re-evaluación tras el diseño (Fase 1):** sigue sin violaciones. Dos puntos para vigilar: (1) el margen de la familia A depende de los clientes de calibración;
se reporta su valor por pliegue; (2) las métricas de exactitud de la familia B no son comparables con las de la A (ver `research.md`, D3).

## Project Structure

### Documentation (this feature)

```text
specs/003-model-tournament/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── tournament-functions.md
├── checklists/requirements.md
└── tasks.md             # lo genera /speckit-tasks (aún no existe)
```

### Source Code (repository root)

```text
src/olistbr/
├── config.py            # EDITAR: TOURNAMENT_GROUPS, TARGET_QUANTILE, CALIBRATION_SIZE, MIN_WIN_SHARE
└── models/
    ├── metrics.py       # EDITAR: bias, share_late, coverage, pinball_loss, metrics_by_group
    ├── train.py         # EDITAR: Candidate, candidates, build_pipeline general, regression_metrics,
    │                    #         DeadlineModel y fit_deadline_model
    └── evaluation.py    # EDITAR: evaluate_candidate_folds, evaluate_candidate_holdout, tournament_summary,
                         #         compare_models, rank_table, beats_baselines, practical_ties, per_thousand

tests/
├── test_metrics.py      # EDITAR: métricas nuevas y la identidad costo = 4 × pérdida cuantílica
├── test_models.py       # EDITAR: candidatos, pipeline general, modelo de plazo y margen
└── test_evaluation.py   # EDITAR: evaluación de candidatos, comparación por pares, posiciones, líneas base

notebooks/05_torneo.ipynb          # COMPLETAR la plantilla del curso; solo llama a funciones
reports/05_torneo.md               # NUEVO: tablas, lectura de la cobertura, finalistas
reports/05_torneo_por_pliegue.csv  # NUEVO: cada modelo en cada pliegue
reports/05_torneo_resumen.csv      # NUEVO: media ± desviación y partición hacia adelante
reports/05_torneo_tiempos.csv      # NUEVO: tiempos de ajuste (cambian entre ejecuciones)
reports/05_torneo_regiones.csv     # NUEVO: MAE y porcentaje tarde por región para los finalistas
```

**Structure Decision**: se sigue el ejemplo CreditSight (`models/train.py` con los candidatos, las métricas y la evaluación). Se mantiene
`models/evaluation.py` de la etapa 4 y se **agregan** funciones nuevas en vez de modificar `evaluate_folds`/`evaluate_holdout`: así la etapa 4 sigue
funcionando sin cambios y sus pruebas siguen pasando (la especificación decía «ampliar `evaluate_folds`»; la intención —una fila por pliegue con todas las
métricas y el tiempo— se cumple con `evaluate_candidate_folds`). Las métricas simples van en `metrics.py` y la compuesta (`regression_metrics`) en
`train.py`, como `metrics_from_proba` en el ejemplo.

## Orden de ejecución del notebook

1. Leer `train.parquet`, calcular `add_features` una vez y armar `X`, `y` y los grupos de variables del torneo.
2. Crear los 5 pliegues por cliente (`make_group_folds`) y la partición hacia adelante (`forward_split`), una sola vez.
3. Para cada candidato: `evaluate_candidate_folds` (5 pliegues) y `evaluate_candidate_holdout` (hacia adelante).
4. Resumir (`tournament_summary`), ordenar por costo y construir la tabla de posiciones (`rank_table`).
5. Comparar por pares el mejor contra los siguientes (`compare_models`) y contra las líneas base (`beats_baselines`); señalar los empates prácticos.
6. Calcular la diferencia de la regresión lineal contra el ganador (`per_thousand`).
7. Para los finalistas, MAE y porcentaje tarde por región (`metrics_by_group`).
8. Guardar los CSV; el reporte con los motivos se redacta **después** de ver las cifras.

## Riesgos y puntos a confirmar

- **Margen y tamaño de datos:** la familia A se ajusta con el 80 % de cada entrenamiento (apartar clientes de calibración), la B con el 100 %; la familia A queda en leve desventaja. Se reporta.
- **Margen por pliegue:** se guarda para ver cuánto varía; un margen muy distinto entre pliegues sería una señal de inestabilidad.
- **Dos formas de leer el ranking:** el costo (entre familias) y la exactitud (dentro de cada familia). El reporte no mezcla las dos.
- **Reproducibilidad:** LightGBM, XGBoost y el bosque usan semilla 42 y `deterministic`/`force_row_wise` donde corresponde; se comprueba con dos ejecuciones (sin la columna de tiempo).
- **La partición hacia adelante es una sola:** sin ruido medido; sirve de guardia, no de prueba estadística.
