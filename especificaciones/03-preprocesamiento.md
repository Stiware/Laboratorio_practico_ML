# Especificación · Etapa 3 · Preprocesamiento (limpieza y partición)

> Se llena **antes** de pedirle código a Claude, a partir de la tabla de decisiones de
> `especificaciones/02-eda.md`. Claude la lee y la implementa; ustedes revisan que el código haga
> **exactamente** esto (`material-del-curso/kit/checklist-revision-ia.md`).
> Guía: `material-del-curso/kit/guia-eda-y-preprocesamiento.md`, sección 4 · Ejemplo: `material-del-curso/ejemplo-creditsight/src/creditsight/data/`.

**Grupo:** _______________   **Fecha:** _______________   **Estado:** ☐ borrador  ☐ cerrada

## 1. Unidad de análisis y partición

- **Una fila es:** _______________
- **Entidad que se repite (columna):** _______________  ☐ no se repite → partir por fila es válido porque: _______________
- **Se parte por:** ☐ grupo (`GroupShuffleSplit` / `StratifiedGroupKFold`) ☐ fila
- **Proporción de prueba:** ___ %   **Semilla:** 42
- **¿Hay orden temporal que respetar?** ☐ no ☐ sí → _______________
- **Dónde se guarda la prueba:** `data/processed/` (cerrada: **no se mira** hasta la etapa 7)

## 2. Datos personales

| Columna | Qué es | Se elimina en | Prueba que lo verifica |
|---|---|---|---|
| | | `load_raw()` (primer paso al cargar) | `test_no_personal_columns` |

## 3. Reglas de limpieza

Una fila por decisión del EDA. **Una función, una prueba.** Se marca como vacío lo imposible; no se borran filas sin criterio.

| # (de la tabla del EDA) | Columna(s) | Problema | Regla exacta (con el umbral o valor) | Función en `src/olistbr/data/cleaning.py` | Prueba en `tests/test_cleaning.py` |
|---|---|---|---|---|---|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |
| 4 | | | | | |
| 5 | | | | | |

## 4. Vacíos

| Columna | Qué significa el vacío | Tratamiento hoy | Tratamiento en la etapa 4 (dentro del `Pipeline`) |
|---|---|---|---|
| | ☐ no aplica ☐ no se anotó ☐ perdido ☐ imposible | ☐ dejar vacío ☐ categoría explícita ☐ reparar con valor de la entidad | |

> Imputar con medias o medianas **no se hace aquí**: depende del entrenamiento y va dentro del `Pipeline`.

## 5. Entradas y salidas

| | Ruta |
|---|---|
| Entrada | `data/raw/_______________` (nunca se modifica) |
| Salida: entrenamiento | `data/processed/train.csv` (o `.parquet`) |
| Salida: prueba (cerrada) | `data/processed/test.csv` (o `.parquet`) |
| Código de limpieza | `src/olistbr/data/cleaning.py` |
| Código de partición | `src/olistbr/data/split.py` |
| Notebook que orquesta | `notebooks/03_preprocesamiento.ipynb` (solo llama a las funciones) |

## 6. Pruebas que deben existir

- ☐ **Partición:** ninguna entidad aparece en dos particiones y no se pierde ninguna fila (`tests/test_split.py`).
- ☐ **La prueba de partición falla** si se cambia temporalmente a partición por fila (compruébenlo y vuelvan a dejarla bien).
- ☐ **Una prueba por regla de limpieza** de la sección 3, con un caso normal y un caso sucio hechos a mano.
- ☐ **Datos personales:** el resultado de cargar no contiene las columnas de la sección 2.
- ☐ `pytest` en verde desde la raíz del repo.

## 7. Criterios de cierre (H1)

- ☐ La limpieza corre **de principio a fin desde cero** (`data/processed/` borrada y regenerada con el notebook 03), sin pasos manuales.
- ☐ Unidad de análisis y forma de partir **escritas** (sección 1) y documentadas en `reports/01_eda.md`.
- ☐ `pytest` en verde.
- ☐ Conjunto de prueba guardado y sin mirar.
- ☐ `data/raw/` intacta y `.env` fuera de Git (`git status`).
- ☐ Todo subido a GitHub.
- ☐ Revisaron el código de Claude con `checklist-revision-ia.md` y cada integrante puede explicar una función.
- ☐ **Hallazgo que no esperaban** (para la puesta en común): _______________
