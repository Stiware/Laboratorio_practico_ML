# Feature Specification: Inferencia y app de OnTrack (etapa 9)

**Feature Branch**: `main` (el grupo trabaja solo en `main`)

**Created**: 2026-11-06

**Status**: Draft

**Input**: User description: "Etapa 9: exportar el modelo champion, esquema de entrada, predicción para regresión que valida los datos y app de Streamlit que recomienda una fecha de entrega, según especificaciones/09-inferencia-app.md (cerrada el 6 nov 2026; el grupo aceptó todas las propuestas)."

> Fuente de las reglas: `especificaciones/09-inferencia-app.md` (cerrada) y la sección 7 de `especificacion-proyecto.md`. Si discrepan, mandan aquellas.
> Prerrequisito: etapas 7 y 8 cerradas (`workspace.default.ontrack_plazo_entrega@champion`, `reports/07_metricas_test.json`, `reports/07_segmentos.csv`).

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Obtener una fecha de entrega recomendada a partir de los datos del pedido (Priority: P1)

Como persona de logística o de atención al cliente, quiero escribir los datos que conozco de un pedido (fecha de compra, estados, distancia, categoría, ítems, precio, flete, peso y volumen) y recibir una **fecha de entrega recomendada**, para prometerle al cliente una fecha realista.

**Why this priority**: es lo que entrega el proyecto; sin esto el modelo no le sirve a nadie.

**Independent Test**: con un modelo de juguete y un esquema de prueba, un pedido normal devuelve un plazo en días, una fecha (compra + plazo, redondeada hacia arriba) y el % tarde de la región del cliente; con el modelo real, el plazo coincide con el que da el pipeline a las mismas columnas calculadas a mano.

**Acceptance Scenarios**:

1. **Given** un pedido con los 11 datos válidos, **When** se pide la predicción, **Then** se devuelve el plazo en días, la fecha recomendada y el % de pedidos que llegaron tarde en la región del cliente según la prueba.
2. **Given** un pedido, **When** se convierte en variables, **Then** el vendedor se toma como «otros», `missing_coords` vale 1 si falta la distancia y 0 si no, y se aplican las mismas transformaciones del entrenamiento (`add_features`).
3. **Given** la app, **When** se abre, **Then** muestra el formulario de 11 campos con etiquetas en español, el título «OnTrack · Fecha de entrega recomendada» y el aviso de uso responsable.

---

### User Story 2 - Que la app no responda cualquier cosa (Priority: P1)

Como grupo, queremos que un dato imposible haga que la predicción **se niegue y explique qué corregir**, que un dato poco común responda con una advertencia y que «No sé» funcione, para que nadie reciba una fecha absurda.

**Why this priority**: «un modelo que contesta con cualquier cosa es más peligroso que uno que dice "no"» (guía §7).

**Independent Test**: una fila con distancia de 9.000 km o con más ítems del vendedor principal que ítems totales lanza un error claro; una fila con «No sé» en la distancia responde con advertencia; una con un estado inexistente se rechaza.

**Acceptance Scenarios**:

1. **Given** un dato fuera del rango **posible**, **When** se pide la predicción, **Then** se rechaza con un mensaje que dice qué campo y qué rango.
2. **Given** un dato dentro de lo posible pero fuera de lo **observado** en el entrenamiento, **When** se pide la predicción, **Then** responde y agrega una advertencia.
3. **Given** «No sé» en un campo numérico o en la categoría, **When** se pide la predicción, **Then** responde (el modelo imputa) y avisa que el plazo es menos preciso si falta la distancia.
4. **Given** que falta la fecha de compra o un estado, o el estado no existe, **When** se pide la predicción, **Then** se rechaza con un mensaje claro.

---

### User Story 3 - Entregar el modelo evaluado, como archivo, listo para desplegar (Priority: P1)

Como grupo, queremos exportar el modelo `champion` a `models/model.joblib`, con su esquema `models/input_schema.json`, comprobar que es el mismo que se evaluó (sin volver a leer la prueba) y que carga en Python 3.12, para que la app funcione sin Databricks y se pueda desplegar en la clase 5.

**Why this priority**: sin el archivo la app no arranca (el error de hoy); con un modelo distinto se entregaría otra cosa que la evaluada.

**Independent Test**: el modelo exportado da las mismas predicciones que `models/final_pipeline.joblib` en 1.000 pedidos del entrenamiento; un entorno temporal con Python 3.12 lo carga y da las mismas predicciones.

**Acceptance Scenarios**:

1. **Given** el registro, **When** se exporta, **Then** `models/model.joblib` sale de `models:/workspace.default.ontrack_plazo_entrega@champion` (o de `final_pipeline.joblib` si Databricks no conecta, y el notebook lo dice).
2. **Given** el modelo exportado y 1.000 pedidos del entrenamiento (semilla 42), **When** se compara con `final_pipeline.joblib`, **Then** los plazos coinciden.
3. **Given** un entorno con Python 3.12 y `requirements.txt`, **When** se carga `models/model.joblib`, **Then** carga y da los mismos plazos que en el entorno del grupo.

---

### User Story 4 - Una ficha que cualquiera pueda leer antes de usar el modelo (Priority: P2)

Como grupo, queremos `models/ficha_modelo.md` completa con las cifras reales de la prueba, las variables que más pesan, los límites y lo que no se evaluó.

**Why this priority**: la exige la guía; la app la cita en el aviso de uso responsable.

**Independent Test**: la ficha no tiene campos en blanco y cada cifra está en `reports/07_metricas_test.json`, `07_shap.csv` o `08_registro.json`.

### Edge Cases

- **Distancia desconocida**: `missing_coords = 1`, el modelo imputa; advertencia de menor precisión.
- **Categoría desconocida o «No sé»**: se usa «desconocida» (como en el entrenamiento); si llega una categoría que el modelo no vio, advertencia.
- **Ítems del vendedor principal > ítems totales**: imposible, se rechaza.
- **Fecha de compra anterior a 2016 o fuera del periodo 2017-01 a 2018-08**: anterior a 2016 se rechaza; fuera del periodo observado, advertencia (el modelo no se validó para otras fechas).
- **Valores negativos** en precio, flete, peso, volumen o distancia: imposibles.
- **Varias filas a la vez**: se valida y se responde cada una.
- **Faltan `models/model.joblib` o el esquema**: la app muestra el mensaje actual («Ejecuta el notebook 09»).

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: La app y la predicción DEBEN pedir 11 datos: fecha de compra, estado del vendedor, estado del cliente, distancia (km), categoría del producto, número de ítems, ítems del vendedor principal, precio total, flete total, peso total (g) y volumen total (cm³).
- **FR-002**: La conversión DEBE fijar `seller_id = "otros"`, `missing_coords` según la distancia, la categoría «desconocida» si falta, y aplicar `add_features`, para entregar al modelo sus 25 columnas.
- **FR-003**: La predicción DEBE devolver el plazo en días, la fecha recomendada (fecha de compra + plazo, redondeada hacia arriba al día), la región del cliente y el % tarde de esa región en la prueba, más la lista de advertencias.
- **FR-004**: DEBE rechazarse con `InputValidationError` y un mensaje por campo: fecha o estados vacíos, estado inexistente, fecha anterior a 2016, y valores fuera de los rangos posibles: distancia 0–4.500 km, precio 0–20.000, flete 0–2.000, ítems 1–30, ítems del vendedor principal 1–30 y ≤ ítems, peso 0–200.000 g, volumen 0–1.500.000 cm³.
- **FR-005**: Fuera del rango observado en el entrenamiento (pero dentro del posible), fuera del periodo 2017-01 a 2018-08, con distancia desconocida o con una categoría no vista, DEBE responder con una advertencia.
- **FR-006**: El esquema `models/input_schema.json` DEBE guardar los campos, etiquetas en español, estados y categorías conocidos, rangos observados y posibles, medianas, el % tarde total y por región de la prueba, el título y el texto de uso responsable.
- **FR-007**: `models/model.joblib` DEBE exportarse desde el champion del registro (respaldo: `final_pipeline.joblib`, declarado) y dar los mismos plazos que `final_pipeline.joblib` en 1.000 pedidos del entrenamiento (semilla 42). La prueba NO se lee.
- **FR-008**: Se DEBE comprobar en un entorno con Python 3.12 y `requirements.txt` que `models/model.joblib` carga y da los mismos plazos; si no carga, se declara y se propone cómo resolverlo antes de seguir.
- **FR-009**: La app DEBE construirse desde el esquema, permitir «No sé» en los campos numéricos y en la categoría (no en la fecha ni en los estados), mostrar la fecha recomendada, el plazo y el % tarde de la región, los errores y advertencias en lenguaje claro, y el aviso de uso responsable (estimación para planificar, no promesa; ≈ 1 de cada 4 llega después; en fin de año el plazo quedó corto; la ubicación cambia el plazo; periodo de entrenamiento; ficha del modelo).
- **FR-010**: La app NO DEBE consultar Databricks ni leer `.env`.
- **FR-011**: La lógica DEBE vivir en `src/olistbr/inference/` (reescrito para regresión) con pruebas en `tests/test_inference.py`; la app se prueba con `streamlit.testing` (`tests/test_app.py`); el notebook 09 solo orquesta y se genera con `scripts/generadores/generar_notebook_09.py`.
- **FR-012**: `models/ficha_modelo.md` DEBE completarse solo con cifras de `reports/07_*` y `08_*`.
- **FR-013**: Las salidas del notebook no DEBEN mostrar el correo ni el enlace del workspace.

### Key Entities

- **Pedido de entrada**: los 11 datos del formulario.
- **Esquema**: campos, rangos, categorías, etiquetas, textos y referencias de la prueba (% tarde total y por región).
- **Resultado**: plazo, fecha recomendada, región, % tarde de la región y advertencias; o un error de validación.
- **Modelo exportado**: `models/model.joblib`, la misma versión del champion.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: `streamlit run app/streamlit_app.py` muestra el formulario de OnTrack (ya no el error de archivos faltantes) y devuelve una fecha para un pedido normal.
- **SC-002**: Un pedido normal, uno con «No sé» y uno imposible dan, respectivamente, una fecha, una fecha con advertencia y un error claro (pruebas automáticas y en el notebook).
- **SC-003**: El modelo exportado da los mismos plazos que el evaluado en 1.000 pedidos, y también en Python 3.12.
- **SC-004**: La ficha del modelo no tiene campos en blanco y no contiene cifras que no estén en los reportes.
- **SC-005**: Todas las pruebas pasan; ningún archivo nuevo ni salida tiene correo, enlace del workspace ni identificadores de clientes.

## Assumptions

- La prueba no se vuelve a leer; la equivalencia con el modelo evaluado se apoya en la ida y vuelta de la etapa 8.
- Python 3.12 se obtiene con `uv` en una carpeta temporal fuera del repositorio.
- El despliegue público (Streamlit Community Cloud) y `reports/09_despliegue.md` son de la clase 5, fuera de esta pieza.
- Los porcentajes de la app (≈ 26 % tarde en total y por región) salen de la prueba de la etapa 7; son referencias históricas, no probabilidades por pedido.
