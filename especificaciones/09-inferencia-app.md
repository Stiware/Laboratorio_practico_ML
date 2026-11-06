# Especificación · Etapa 9 · Inferencia y app

> Se llena **antes** de pedirle código a Claude. Borrador propuesto por Claude el 5 nov 2026 a partir de la guía
> `material-del-curso/kit/guia-validacion-registro-y-app.md` (secciones 6 a 9), la plantilla `notebooks/09_inferencia.ipynb`, `app/streamlit_app.py`,
> `src/olistbr/inference/` y los resultados de `reports/07_validacion.md` y `08_registro.md`.
> **Toda la plantilla está hecha para clasificación** (clase, probabilidades, matriz de costos). OnTrack es **regresión**: recomienda un **plazo en días**.
> La guía lo prevé: «en regresión la salida es un número; pide a Claude la adaptación de `predict.py` y de la app».

**Grupo:** _______________   **Fecha:** 5 nov 2026   **Estado:** ☒ borrador  ☐ cerrada _(falta que el grupo responda Q1 a Q8)_

## 1. Qué cambia frente a la plantilla

| Pieza | Plantilla (clasificación) | OnTrack (regresión) |
|---|---|---|
| Salida | clase decidida, probabilidad por clase, «revisar si duda» | **plazo recomendado en días** (percentil 75) y la **fecha** de entrega recomendada (compra + plazo) |
| Entrada | las columnas del modelo, tal cual | El modelo usa **25 columnas calculadas** (logaritmos, región, mismo estado…). La app pide los **datos del pedido** (Q1) y `add_features` calcula el resto |
| `inference/predict.py`, `inference/schema.py`, `tests/test_inference.py` | clasificación | **se reescriben para regresión** (son de esta etapa; no los usa ninguna otra) |
| Comprobación «el exportado es el evaluado» | con la prueba | **sin la prueba** (ya se leyó dos veces): mismas predicciones que `final_pipeline.joblib` en pedidos del entrenamiento; el champion ya se comparó con la prueba en la etapa 8 |

## 2. Preguntas para el grupo

| # | Pregunta | Por qué importa | Propuesta de Claude |
|---|---|---|---|
| Q1 | **¿Qué le pregunta la app a la persona?** | El formulario debe pedir lo que alguien de logística sabe **al momento de la compra**, no columnas técnicas | 11 campos: **fecha de compra**, **estado del vendedor**, **estado del cliente**, **distancia en km**, **categoría del producto**, **número de ítems**, **ítems del vendedor principal**, **precio total**, **flete total**, **peso total (g)**, **volumen total (cm³)**. **No se pregunta el vendedor** (`seller_id` es un código anónimo que nadie conoce): se usa la categoría «otros» que el modelo ya conoce para vendedores poco frecuentes. Tampoco el plazo prometido por Olist (no entra al modelo) |
| Q2 | **¿Qué pasa si la persona no sabe la distancia?** | Es la variable más importante después de «mismo estado» | Casilla **«No sé»**: la distancia queda vacía y se marca `missing_coords = 1`, igual que en los 386 pedidos sin coordenadas del entrenamiento; el `Pipeline` la imputa con la mediana. La app avisa que el plazo es menos preciso. «No sé» se permite en todos los campos numéricos y en la categoría; **no** en la fecha ni en los estados (sin ellos el plazo no tiene sentido) |
| Q3 | **¿Qué valores son imposibles?** (la predicción se niega y explica) | Lo dice la guía: el rango **posible** es conocimiento del grupo, no de los datos | Distancia 0 a **4.500 km** (≈ la mayor distancia dentro de Brasil; máximo observado 3.399) · precio total 0 a **20.000** · flete 0 a **2.000** · ítems 1 a **30** e ítems del vendedor principal ≤ ítems · peso 0 a **200.000 g** · volumen 0 a **1.500.000 cm³** · fecha de compra no anterior a 2016. Fuera del rango **observado** (pero dentro del posible) solo se muestra una advertencia («el modelo vio pocos pedidos así») |
| Q4 | **¿Qué muestra la app como resultado?** | La persona necesita una fecha, no un percentil | **Fecha recomendada** (compra + plazo, redondeado hacia arriba al día) y el plazo en días. Debajo, en palabras simples: «en la prueba, ≈ 1 de cada 4 pedidos llegó después de la fecha recomendada (26 %)», y el % tarde de la **región del cliente** en la prueba (`reports/07_segmentos.csv`). No se muestra una probabilidad por pedido (no se evaluó, Q8 de la etapa 7) |
| Q5 | **¿Cómo se exporta el modelo?** | La guía: el registro es la fuente de verdad | Cargar `models:/workspace.default.ontrack_plazo_entrega@champion` desde Databricks y guardarlo con `joblib` en **`models/model.joblib`** (≈ 3 MB; sí va al repositorio). Si Databricks no conecta, se usa `models/final_pipeline.joblib` y el notebook lo dice. Se comprueba que da **las mismas predicciones** que `final_pipeline.joblib` en 1.000 pedidos del entrenamiento |
| Q6 | **¿Se comprueba que el modelo carga en Python 3.12?** | Se guardó con 3.14 (el `.venv` de esta Mac) y el proyecto pide 3.12; un pickle puede no cargar en otra versión, y el despliegue de la clase 5 usará `requirements.txt` | **Sí:** crear con `uv` un entorno temporal con Python 3.12 y `requirements.txt` (fuera del repositorio), cargar `models/model.joblib` y predecir 1.000 pedidos; deben coincidir con los de 3.14. Si no carga, se reentrena el mismo modelo con 3.12 (mismos datos y parámetros) y se declara |
| Q7 | **¿Qué textos lleva la app?** | Uso responsable (principio VIII y regla 8) | Título **«OnTrack · Fecha de entrega recomendada»**, etiquetas en español y un aviso: es una **estimación para planificar, no una promesa**; ≈ 1 de cada 4 pedidos llega después; en meses de alta demanda (fin de año) el plazo quedó corto en la validación; la ubicación del cliente cambia el plazo; modelo entrenado con pedidos de 2017-01 a 2018-08 (no validado para otras fechas); ver `models/ficha_modelo.md` |
| Q8 | **¿Quién completa la ficha del modelo?** | La guía pide completarla «a mano» con cifras reales | Claude la completa con las cifras de `reports/07_validacion.md` y `08_registro.md` (sin cifras nuevas) y el grupo la revisa |

## 3. Reglas

1. **La app no consulta Databricks** (regla 8): carga `models/model.joblib` desde el repositorio.
2. **La prueba no se vuelve a leer** en esta etapa.
3. Toda la lógica (datos del formulario → variables → plazo, validación) vive en `src/olistbr/inference/` con pruebas; el notebook 09 solo orquesta y la app solo dibuja.
4. Un dato imposible **se niega a responder** y explica qué corregir; un dato fuera de lo observado responde con advertencia; un «No sé» responde.
5. Las salidas del notebook no muestran el correo ni el enlace del workspace.

## 4. Código previsto

| Archivo | Qué hace | Pruebas |
|---|---|---|
| `src/olistbr/inference/schema.py` (reescrito) | `build_schema(train, valid_range, labels, path)`: campos del formulario, categorías conocidas (estados, categorías de producto), rangos observados y posibles, medianas; `load_schema(path)` | `tests/test_inference.py` |
| `src/olistbr/inference/predict.py` (reescrito) | `validate_order(row, schema)` (imposibles → `InputValidationError` con el motivo; fuera de lo observado → advertencias); `order_to_features(row)` (rellena `seller_id="otros"`, `missing_coords`, y aplica `add_features`); `predict_deadline(rows, model, schema)` → plazo en días, fecha recomendada y advertencias | fila normal, fila con «No sé», fila imposible, ítems del vendedor > ítems, estado desconocido, la fecha se redondea hacia arriba, mismo plazo que el pipeline con las columnas calculadas a mano |
| `app/streamlit_app.py` (adaptado) | formulario de los 11 campos con «No sé», resultado (fecha, días, % tarde de la región), errores claros y aviso de uso responsable | arranque sin errores con `streamlit run` (comprobación manual) |
| `notebooks/09_inferencia.ipynb` | exportar desde el champion, esquema, tres casos de prueba, comprobación del exportado, ficha | — |

## 5. Entregables

- `models/model.joblib`, `models/input_schema.json`, `models/ficha_modelo.md` (completa).
- `src/olistbr/inference/` y `tests/test_inference.py` para regresión; `app/streamlit_app.py` adaptada.
- `notebooks/09_inferencia.ipynb` (generado por `scripts/generadores/generar_notebook_09.py`) y `reports/09_despliegue.md` (plantilla existente).
- Línea en `reports/bitacora_decisiones.md` y en la sección 7 de `especificacion-proyecto.md`.

## 6. Criterios de cierre

- ☐ Q1 a Q8 respondidas y confirmadas por el grupo.
- ☐ `models/model.joblib` exportado desde el champion y con las mismas predicciones que el modelo evaluado.
- ☐ Carga y predice igual en Python 3.12.
- ☐ `predict` responde a una fila normal y a una con «No sé», y se niega con un mensaje claro ante un dato imposible (pruebas).
- ☐ `streamlit run app/streamlit_app.py` muestra el formulario, la fecha recomendada y el aviso de uso responsable.
- ☐ Ficha del modelo completa con cifras reales.
- ☐ `pytest tests` en verde; revisión con `revisor-de-codigo-ia`.
