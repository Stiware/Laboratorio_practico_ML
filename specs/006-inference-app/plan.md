# Implementation Plan: Inferencia y app de OnTrack (etapa 9)

**Branch**: `main` | **Date**: 2026-11-06 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/006-inference-app/spec.md`

## Summary

Se reescribe `src/olistbr/inference/` para **regresión**: `schema.py` arma el esquema del formulario (11 campos, rangos posibles y observados, estados y categorías,
textos, % tarde total y por región de la prueba) y `predict.py` valida un pedido, lo convierte en las 25 columnas del modelo con `add_features` (vendedor «otros»,
`missing_coords`) y devuelve el plazo, la fecha recomendada y advertencias, o un error claro. La app de Streamlit se adapta al formulario y al resultado, y se prueba con
`streamlit.testing.AppTest`. El notebook 09 exporta `models/model.joblib` desde el champion, guarda el esquema, prueba los tres casos y compara el exportado con el modelo
evaluado en pedidos del entrenamiento. Un script verifica la carga en Python 3.12 en un entorno temporal creado con `uv`. Se completa la ficha del modelo.

## Technical Context

**Language/Version**: Python 3.12 (requisito); el `.venv` de esta máquina es 3.14; la carga en 3.12 se verifica aparte (D5).

**Primary Dependencies**: pandas, numpy, scikit-learn, LightGBM, MLflow (solo en el notebook, para exportar), Streamlit 1.66.0, joblib. Todas en `requirements.txt`; **no se agrega ninguna**.

**Storage**: entradas `train.parquet`, `reports/07_metricas_test.json`, `07_segmentos.csv`, el champion (o `final_pipeline.joblib`). Salidas: `models/model.joblib`,
`models/input_schema.json`, `models/ficha_modelo.md`.

**Testing**: pytest; `tests/test_inference.py` (reescrito) con un pipeline de juguete y el esquema de una tabla sintética; `tests/test_app.py` con `AppTest` y rutas de prueba
por variable de entorno. Ninguna prueba lee `data/processed/` ni `models/model.joblib`.

**Target Platform**: local (`streamlit run`); despliegue en la clase 5.

**Project Type**: biblioteca + app de Streamlit + notebook que orquesta.

**Performance Goals**: una predicción en menos de 1 s; el notebook 09 en menos de 3 min (más la descarga del champion).

**Constraints**: la app no consulta Databricks ni lee `.env`; la prueba no se lee; sin correo ni enlace en salidas; textos en español, código en inglés.

**Scale/Scope**: 11 campos de entrada, 25 columnas del modelo, 27 estados, 74 categorías.

## Constitution Check

| Principio | Cómo lo cumple | ¿Pasa? |
|---|---|---|
| I. Datos originales no se tocan | Solo lee `train.parquet` y reportes | Sí |
| II. Partición por la unidad | No hay partición nueva; las cifras de la app vienen de la prueba por cliente | Sí |
| III. La prueba se abre una vez | No se lee la prueba; la equivalencia del exportado se comprueba con pedidos del entrenamiento | Sí |
| IV. Decisión con motivo, función y prueba | Rangos posibles con su motivo en el esquema y en el plan; funciones con pruebas | Sí |
| V. Nada sensible sale | La app no pide identificadores; el esquema no guarda clientes ni pedidos; sin `.env` | Sí |
| VI. El código es nuestro | Funciones cortas con docstring; revisión con `revisor-de-codigo-ia` | Sí |
| VII. No se inventan datos | Los % de la app y las cifras de la ficha salen de `reports/07_*` y `08_*` | Sí |
| VIII. Uso responsable | Aviso en la app, % tarde por región, límites en la ficha | Sí |

**Resultado:** sin violaciones. **Nota:** se reescriben `inference/` y `tests/test_inference.py`, que hoy son la plantilla de clasificación; no los usa ninguna otra etapa
(comprobado: solo los importa la app). Las 13 pruebas de clasificación se reemplazan por pruebas de regresión.

## Decisiones de diseño (research)

- **D1. Pedido → variables.** `order_to_features(orders)` arma una tabla con las columnas de `train.parquet` que usa `add_features`: `order_purchase_timestamp` = fecha
  de compra, `order_estimated_delivery_date` vacía (solo alimenta `promised_days`, que el modelo no usa), `seller_id = "otros"`, `missing_coords = 1` si falta la distancia,
  categoría «desconocida» si falta; luego `add_features`. Una prueba verifica que el plazo es igual al del pipeline con las mismas columnas armadas a mano.
- **D2. Fecha recomendada.** `fecha = normalizar(compra + plazo días)`, redondeando hacia arriba al día siguiente si sobran horas: así la promesa no queda antes del
  plazo del modelo. El plazo se muestra con un decimal.
- **D3. Validación en dos niveles.** Imposible → `InputValidationError` con todos los motivos juntos; poco común → advertencias en `result.attrs["warnings"]` y en una
  columna `advertencias` por fila. Rangos posibles (FR-004) en el esquema; rangos observados de `train.parquet`.
- **D4. Exportar.** En el notebook: `configure_mlflow()` y `load_champion` de `tracking/registry.py`; `joblib.dump` a `models/model.joblib`. Si falla la conexión
  (errores de MLflow o de red), `joblib.load("models/final_pipeline.joblib")` y se imprime «respaldo». Comparación con `final_pipeline.joblib` en 1.000 pedidos del
  entrenamiento con `np.allclose`.
- **D5. Python 3.12.** `scripts/verificar_modelo.py` (nuevo, sin dependencias de `src/`): carga `models/model.joblib`, predice una tabla de 1.000 filas guardada
  temporalmente y compara con los plazos esperados. Se corre con el `.venv` (guarda la referencia) y con un entorno `uv` de Python 3.12 en la carpeta temporal
  (`uv venv --python 3.12` + `uv pip install -r requirements.txt`). Si no carga, se para y se avisa al grupo.
- **D6. App.** Se reescribe `app/streamlit_app.py` (la de la plantilla es de clasificación): título, formulario de 11 campos en tres columnas, casilla «No sé» en los
  numéricos y en la categoría, botón «Recomendar fecha»; resultado con `st.metric` (fecha y plazo), el % tarde de la región y advertencias; errores con `st.error`;
  expansor «Uso responsable». Rutas del modelo y del esquema por variable de entorno (`ONTRACK_MODEL_PATH`, `ONTRACK_SCHEMA_PATH`) para poder probarla con archivos de juguete.
- **D7. Ficha.** Se completa a mano con las cifras de `reports/07_validacion.md` y `08_registro.md`.

## Project Structure

```text
src/olistbr/inference/schema.py     # REESCRITO: FORM_FIELDS, VALID_RANGES, build_schema, load_schema
src/olistbr/inference/predict.py    # REESCRITO: InputValidationError, load_model, validate_orders, order_to_features, predict_deadline
app/streamlit_app.py                # REESCRITO para regresión
tests/test_inference.py             # REESCRITO
tests/test_app.py                   # NUEVO (AppTest)
scripts/verificar_modelo.py         # NUEVO
scripts/generadores/generar_notebook_09.py   # NUEVO
notebooks/09_inferencia.ipynb       # COMPLETAR la plantilla
models/model.joblib, models/input_schema.json  # NUEVOS (sí se versionan)
models/ficha_modelo.md              # COMPLETAR
```

## Riesgos

- **Python 3.12:** si el pickle de 3.14 no carga, habrá que reentrenar el mismo modelo en 3.12 (mismos datos y parámetros) y volver a registrarlo; se decidiría con el grupo.
- **Diferencias de tipos** entre el formulario y el entrenamiento (enteros, fechas, día de la semana como texto): cubiertas por la prueba de D1.
- **La app no se probó en la nube:** eso es la clase 5.
