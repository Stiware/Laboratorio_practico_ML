# Implementation Plan: Optimización de hiperparámetros de OnTrack (etapa 6, Fase 2)

**Branch**: `main` | **Date**: 2026-11-05 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/004-hyperparameter-tuning/spec.md`

## Summary

Se agrega `src/olistbr/models/tuning.py` con la misma forma que `tuning.py` del ejemplo CreditSight, adaptada a **regresión**: espacios de búsqueda y constructores
de los dos finalistas (LightGBM cuantil 0,75 e HistGradientBoosting cuantil 0,75, con el percentil 75 fijo), `tune` con Optuna (20 ensayos, semilla 42, reanudable)
que registra cada ensayo en MLflow, y la comprobación honesta: `bootstrap_cost_difference` (intervalo del 95 % sorteando clientes) más las reglas `decide_params`
y `choose_finalist`. Cada ensayo se mide con `evaluate_candidate_folds` del torneo, así que se mide exactamente igual que en la etapa 5. Se reserva el 25 % de los
clientes con `split_by_group` (semilla 7) y se agrega `reserved_forward_split` a `data/split.py` para la prueba hacia adelante solo con clientes reservados.
El notebook 06 solo orquesta; `train.py`, `evaluation.py`, `metrics.py` y `mlflow_utils.py` **no se modifican**.

## Technical Context

**Language/Version**: Python 3.12 (requisito del proyecto). _En esta máquina el `.venv` es 3.14; ver Riesgos._

**Primary Dependencies**: pandas, numpy, scikit-learn 1.9.1, LightGBM 4.7.0, Optuna 5.0.0, MLflow 3.17.1 (SQLAlchemy 2.1.8 viene con MLflow). Todas ya están en
`requirements.txt`; **no se agrega ninguna dependencia**.

**Storage**: archivos. Entrada `data/processed/train.parquet` (solo lectura). Estudio de Optuna en `data/processed/optuna_optimizacion.db` (ignorado por Git).
Salidas: `reports/06_ensayos.csv`, `reports/06_best_params.json`, `reports/06_optimizacion.md`. Seguimiento: MLflow en Databricks o `mlflow_local.db`.

**Testing**: pytest con tablas pequeñas hechas a mano y datos sintéticos; `tune` con 2 ensayos y `log=False`; sin leer `data/processed/`.

**Target Platform**: equipo local del grupo (macOS/Windows), 2 hilos por modelo.

**Project Type**: biblioteca (`src/olistbr/`) más un notebook que orquesta.

**Performance Goals**: búsqueda completa en ≈ 20 a 35 minutos (estimado con los tiempos del torneo: LightGBM 3,1 s y HistGradientBoosting 7,8 s por ajuste); pruebas nuevas en menos de 30 s.

**Constraints**: nunca abrir `test.parquet`; la búsqueda no ve a la reserva; todo lo que aprende dentro del `Pipeline`; percentil 75 fijo; mismas variables del torneo
(`TOURNAMENT_GROUPS`); no imprimir `.env`; textos en español y nombres de código en inglés.

**Scale/Scope**: 76.924 pedidos → ≈ 57.700 de búsqueda y ≈ 19.200 reservados; 2 finalistas × 20 ensayos × 5 pliegues = 200 ajustes, más 4 ajustes de comprobación y los informativos.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principio | Cómo lo cumple este plan | ¿Pasa? |
|---|---|---|
| I. Datos originales no se tocan | Solo lee `train.parquet`; no toca `data/raw/` | Sí |
| II. Se parte por la unidad de análisis | Reserva con `split_by_group` (por cliente), pliegues con `make_group_folds`, hacia adelante con clientes reservados; el intervalo sortea **clientes completos** | Sí |
| III. La prueba se abre una vez | `test.parquet` no se lee (comprobación con `grep`); se decide con la reserva, que es parte del entrenamiento | Sí |
| IV. Decisión con motivo, función y prueba | Las reglas de decisión son funciones con pruebas; decisiones en `06_best_params.json` y `reports/06_optimizacion.md` | Sí |
| V. Nada sensible sale | `customer_unique_id` solo agrupa; MLflow no recibe identificadores ni datos; `.env` no se imprime | Sí |
| VI. El código es nuestro | Funciones cortas con docstring en español, forma del ejemplo; revisión con los subagentes | Sí |
| VII. No se inventan datos | Las cifras salen del notebook ejecutado; la mejor puntuación de la búsqueda se marca como optimista; los tiempos son estimaciones dichas como tales | Sí |
| VIII. Uso responsable | Tabla por región del modelo elegido y cobertura mes a mes hacia adelante (informativas) | Sí |

**Resultado:** sin violaciones; nada en «Complexity Tracking».

**Re-evaluación tras el diseño (Fase 1):** sigue sin violaciones. Puntos para vigilar: (1) la lista de vendedores frecuentes de la etapa 3 incluye a la reserva (solo
conteos; limitación anotada, la revisa el `auditor-de-fuga`); (2) la reserva se usa dos veces (partida contra optimizado, y entre finalistas), así que el resultado
del elegido en la reserva es algo optimista; la estimación honesta es la de la etapa 7.

## Project Structure

### Documentation (this feature)

```text
specs/004-hyperparameter-tuning/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── tuning-functions.md
├── checklists/requirements.md
└── tasks.md             # lo genera /speckit-tasks (aún no existe)
```

### Source Code (repository root)

```text
src/olistbr/
├── data/
│   └── split.py         # EDITAR: agregar reserved_forward_split
└── models/
    └── tuning.py        # NUEVO: espacios, constructores, FINALISTS, starting_candidate, tuned_candidate, tune, trials_table,
                         #        fit_and_recommend, n_trees_used, bootstrap_cost_difference, decide_params, choose_finalist

tests/
├── test_tuning.py       # NUEVO: pruebas del contrato
└── test_split.py        # EDITAR: reserva del 25 % y reserved_forward_split

scripts/generadores/generar_notebook_06.py   # NUEVO: arma el notebook sobre la plantilla del curso
notebooks/06_optimizacion.ipynb              # COMPLETAR la plantilla; lee Parquet; solo llama a funciones
reports/06_ensayos.csv                       # NUEVO
reports/06_best_params.json                  # NUEVO
reports/06_optimizacion.md                   # COMPLETAR la plantilla existente
reports/bitacora_decisiones.md               # EDITAR: una línea por hito
```

**Structure Decision**: se sigue el ejemplo CreditSight (`models/tuning.py` aparte de `train.py`). La partición hacia adelante con clientes reservados va en
`data/split.py`, junto a `forward_split`, porque es una partición. No se modifica ningún módulo de las etapas anteriores: así sus 166 pruebas siguen valiendo sin cambios.

## Orden de ejecución del notebook

1. Configurar MLflow (`configure_mlflow`, `set_experiment("ontrack-optimizacion")`) y mostrar dónde quedó.
2. Leer `train.parquet`, aplicar `add_features` y separar `busqueda` / `reserva` (`split_by_group(0.25, seed=7)`) con su `assert`.
3. Crear los 5 pliegues de búsqueda (`make_group_folds`, semilla 42) con su `assert`.
4. Para cada finalista: `tune` (20 ensayos, reanudable) → `trials_table`; mostrar el mejor costo **como optimista**.
5. Para cada finalista: `fit_and_recommend` con partida y con optimizados (ajuste con `busqueda`, evaluación en `reserva`), `n_trees_used` de la partida,
   `bootstrap_cost_difference` y `decide_params`.
6. Con los parámetros elegidos de cada uno: `bootstrap_cost_difference` (LightGBM − HistGradientBoosting) y `choose_finalist`.
7. Informativo: `reserved_forward_split` + `evaluate_candidate_holdout` y `coverage_by_month` (partida y optimizado de cada finalista); `metrics_by_group` por región del elegido.
8. Guardar `06_ensayos.csv` y `06_best_params.json`; el reporte se redacta **después** de ver las cifras.

## Riesgos y puntos a confirmar

- **Entorno:** el `.venv` de esta máquina es Python 3.14 (el proyecto pide 3.12) y aquí faltan `data/raw/` y `data/processed/`. Las pruebas corren sin datos; el notebook no.
- **Tiempo:** estimado, no medido; los ensayos con 800 árboles y 127 hojas pueden tardar varias veces más que la partida. El estudio es reanudable.
- **La optimización puede no ayudar:** es un resultado válido; se usan los de partida y se escribe así.
- **Reproducibilidad al reanudar:** una búsqueda cortada y reanudada puede no repetir la misma secuencia de ensayos; SC-006 se exige con una corrida completa.
- **Databricks:** si no conecta o se agota la cuota, se sigue en local (`MLFLOW_FORCE_LOCAL=1` lo fuerza) y se avisa.
- **Costos no comparables con el torneo:** la búsqueda ajusta con ≈ 46.000 pedidos por pliegue (en el torneo ≈ 61.500).

## Complexity Tracking

Sin violaciones que justificar.
