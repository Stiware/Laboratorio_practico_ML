# Implementation Plan: Variables nuevas de OnTrack y su medición (etapa 4)

**Branch**: `main` | **Date**: 2026-10-31 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/002-features-engineering/spec.md`

## Summary

Se agregan las 10 ideas de variables aprobadas en `especificaciones/04-features.md` como **una función de pandas sin estado**
(`add_features`), un registro de **grupos de ideas** que se activan o desactivan (`feature_columns`) y un **preprocesador
dentro de un `Pipeline`** (`build_preprocessor`), con la misma forma que `engineering.py` del ejemplo CreditSight. Luego se
mide cada idea contra una referencia: 5 pliegues por cliente, el mismo modelo (`LinearRegression`), error en días y costo 3 a 1,
`mejora / ruido` calculado con las diferencias entre pliegues, y una prueba hacia adelante (entrenar hasta 2018-02-28 y validar
de marzo a agosto de 2018). La decisión de cada idea sale de una función que aplica la regla escrita. El notebook 04 solo orquesta
y escribe `reports/04_features.md`. El plazo prometido por Olist se mide solo como referencia y el diseño impide que entre al
modelo final.

## Technical Context

**Language/Version**: Python 3.12.

**Primary Dependencies**: pandas 3.0.6, numpy 2.5.4, scikit-learn 1.9.1 (`ColumnTransformer`, `SimpleImputer`, `StandardScaler`,
`OneHotEncoder`, `TargetEncoder`, `LinearRegression`, `GroupKFold`, `KFold`). Todas ya están en `requirements.txt`; **no se agrega
ninguna dependencia nueva**.

**Storage**: archivos. Entrada `data/processed/train.parquet` (solo lectura). Salidas: `reports/04_features.md` y
`reports/04_features_tabla.csv` (la tabla de ideas medidas, como en el ejemplo del curso).

**Testing**: pytest 9.1.1 con `pytest tests`; casos hechos a mano, sin leer `data/processed/`.

**Target Platform**: Linux/Windows local del grupo (notebook ejecutado con el `.venv`).

**Project Type**: biblioteca (`src/olistbr/`) más un notebook que orquesta.

**Performance Goals**: las 14 mediciones (referencia, 9 grupos, plazo prometido, línea base trivial y dos conjuntos) con 5 pliegues más la
prueba hacia adelante deben correr en unos minutos en un portátil (≈ 77 mil filas; el preprocesador produce 66 columnas en la
referencia y 103 con todos los grupos). Medido: unos 8 minutos con todo.

**Constraints**: nunca abrir `test.parquet`; todo lo que aprende de los datos dentro del `Pipeline`; sin información posterior a la
compra; el plazo prometido nunca en el modelo final; textos en español y nombres de código en inglés.

**Scale/Scope**: 76.924 pedidos de entrenamiento, 74.425 clientes, 27 estados, 74 categorías, 1.165 vendedores (más «otros»).

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principio | Cómo lo cumple este plan | ¿Pasa? |
|---|---|---|
| I. Datos originales no se tocan | Solo lee `train.parquet`; no toca `data/raw/` | Sí |
| II. Se parte por la unidad de análisis | Pliegues con `GroupKFold` por `customer_unique_id`; la prueba hacia adelante saca del entrenamiento a los clientes que están en la validación; `assert_disjoint_groups` en ambos | Sí |
| III. La prueba se abre una vez | No se lee `test.parquet`; medianas, escalas, categorías y promedios históricos se ajustan dentro del `Pipeline` con el entrenamiento de cada pliegue | Sí |
| IV. Decisión con motivo, función y prueba | Cada idea tiene su función, su prueba y su fila en `reports/04_features.md` con la cifra | Sí |
| V. Nada sensible sale | No se usan identificadores ni texto libre como entrada; `customer_unique_id` solo agrupa | Sí |
| VI. El código es nuestro | Funciones cortas con docstring en español; revisión con los subagentes antes de cerrar | Sí |
| VII. No se inventan datos | Todas las cifras del reporte salen del notebook ejecutado | Sí |
| VIII. Uso responsable | Se conserva `customer_state` para revisar el desempeño por región en las etapas 6 y 7; la limitación temporal se reporta | Sí |

**Resultado:** sin violaciones; nada en «Complexity Tracking».

**Re-evaluación tras el diseño (Fase 1):** sigue sin violaciones. Dos puntos para vigilar: (1) el promedio histórico por vendedor,
categoría y ruta es la única parte que aprende del objetivo, por eso va solo en el `Pipeline` con `TargetEncoder`; (2) el separador
interno de `TargetEncoder` no conoce a los clientes, lo que no afecta la validación (ver `research.md`, D2).

## Project Structure

### Documentation (this feature)

```text
specs/002-features-engineering/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── features-functions.md
├── checklists/requirements.md
└── tasks.md             # lo genera /speckit-tasks (aún no existe)
```

### Source Code (repository root)

```text
src/olistbr/
├── config.py            # EDITAR: N_FOLDS, FORWARD_*, MIN_CATEGORY_FREQUENCY, MIN_IMPROVEMENT_RATIO
├── data/
│   └── split.py         # EDITAR: make_group_folds, forward_split
├── features/
│   └── engineering.py   # NUEVO: add_features, feature_columns, build_preprocessor (+ tablas fijas)
└── models/
    ├── train.py         # NUEVO: build_pipeline (preprocesador + LinearRegression)
    └── evaluation.py    # NUEVO: evaluate_folds, compare_to_reference, decide_idea

tests/
├── test_features.py     # EDITAR: se agregan las pruebas de engineering.py (ya tiene geo y categorías)
├── test_split.py        # EDITAR: pliegues y partición hacia adelante
├── test_models.py       # NUEVO: build_pipeline y evaluate_folds
└── test_evaluation.py   # NUEVO: mejora/ruido y regla de decisión

notebooks/04_features.ipynb    # COMPLETAR la plantilla del curso; solo llama a funciones
reports/04_features.md         # NUEVO: una fila por idea con su cifra
reports/04_features_tabla.csv  # NUEVO: la tabla de resultados
```

**Structure Decision**: se sigue el ejemplo CreditSight: `features/engineering.py` con `add_features`, `feature_columns` y
`build_preprocessor`; `models/train.py` con `build_pipeline`; y la partición en `data/split.py`. Se agrega `models/evaluation.py`
(el ejemplo lo tiene dentro de `train.py`/`tuning.py`) para que la regla de decisión, que es la parte que defendemos ante el
profesor, quede en un solo archivo pequeño. El ejemplo usa banderas booleanas; aquí se usa un **registro de grupos** (un nombre
por idea) porque son 9 ideas y las banderas serían difíciles de leer.

## Orden de ejecución del notebook

1. Leer `train.parquet` y calcular `add_features` **una sola vez** (no aprende de los datos, por eso no filtra información).
2. Crear los 5 pliegues por cliente **una sola vez** y verificar que no comparten clientes.
3. Medir la referencia y cada grupo de ideas con los mismos pliegues.
4. Medir el plazo prometido como referencia («no usable en producción»).
5. Repetir con la prueba hacia adelante.
6. Aplicar la regla de decisión por idea; medir el conjunto conservado como comprobación.
7. Guardar `reports/04_features_tabla.csv` (el reporte con motivos se redacta con las cifras ya obtenidas).

## Riesgos y puntos a confirmar

- **Efecto del orden de las ideas:** cada idea se mide sola contra la referencia, y después se mide la combinación de las
  conservadas. Dos ideas correlacionadas (por ejemplo, logaritmo de distancia y distancia × mismo estado) pueden verse útiles cada
  una y redundantes juntas; el reporte lo dirá si ocurre.
- **La idea 7 (distancia × mismo estado)** incluye sus componentes (`log_distance` y `same_state`) para no medir una interacción sin
  sus efectos principales; se compara también con las ideas 1 y 2.
- **Una sola temporada de fin de año:** la marca de temporada alta no se puede validar hacia adelante. Se informa «no evaluable
  hacia adelante» y la decisión descansa en la validación por cliente, con la limitación escrita.
- **La mezcla de pliegues cambia con la semilla:** se fija la semilla 42 y se reporta la desviación entre pliegues; no se cambia la
  semilla para «buscar» un resultado.
