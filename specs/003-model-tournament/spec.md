# Feature Specification: Torneo de modelos de OnTrack (etapa 5, Fase 1)

**Feature Branch**: `main` (el grupo trabaja solo en `main`)

**Created**: 2026-10-31

**Status**: Draft

**Input**: User description: "Etapa 5 · Fase 1 · Torneo de modelos de OnTrack según `especificaciones/05-torneo.md` (cerrada). Regresión: predecir `dias_entrega` y recomendar un plazo. Candidatos con parámetros fijos: dos líneas base (mediana y plazo fijo del percentil 75), regresión lineal, Ridge, bosque aleatorio liviano, LightGBM y XGBoost con un margen (familia A: margen = percentil 75 de los errores en clientes de calibración apartados dentro del entrenamiento) y LightGBM, XGBoost e HistGradientBoosting que predicen directamente el percentil 75 (familia B). Se evalúan con los 5 pliegues por cliente de la etapa 4 más la prueba hacia adelante, con 7 grupos de variables (sin la marca de fin de año). El costo 3 a 1 decide al ganador; se reportan MAE, RMSE, R², sesgo, porcentaje que llega tarde, cobertura y tiempo de ajuste. Comparación por pares, 1 o 2 finalistas, la regresión lineal siempre compite. Código en `src/olistbr/models/` con pruebas, notebook 05 que solo orquesta y `reports/05_torneo.md`."

> Fuente de las reglas: `especificaciones/05-torneo.md` (cerrada el 31 oct 2026) y la sección 7 de `especificacion-proyecto.md`. Si esta especificación y aquellas discrepan, mandan aquellas.
> Esta es la **Fase 1** de la etapa 5 (lo que corresponde a `train.py` del ejemplo). La optimización de hiperparámetros (`tuning.py`) es la Fase 2 y tendrá su propia especificación.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Saber qué modelos descartar y cuáles pasan como finalistas (Priority: P1)

Como analista del grupo, quiero una tabla que compare todos los candidatos con la misma validación y las mismas métricas, para descartar con números los que no sirven y quedarme con 1 o 2 finalistas que optimizar después.

**Why this priority**: es el entregable de la etapa; sin finalistas no se puede pasar a la optimización.

**Independent Test**: ejecutar el torneo y comprobar que cada candidato tiene una fila con el costo 3 a 1, MAE, RMSE, R², sesgo, porcentaje tarde, cobertura y tiempo de ajuste (media ± desviación entre los 5 pliegues), más su resultado en la prueba hacia adelante, y que ambas líneas base están en la tabla.

**Acceptance Scenarios**:

1. **Given** el conjunto de entrenamiento y los pliegues por cliente de la etapa 4, **When** se evalúa cada candidato, **Then** todos usan exactamente los mismos pliegues y la misma partición hacia adelante, y se guarda el resultado de **cada pliegue**, no solo el promedio.
2. **Given** las dos líneas base, **When** se comparan con un modelo, **Then** se informa si el modelo las supera por más que su ruido; un modelo que no lo hace se marca como descartado.
3. **Given** modelos de familias distintas, **When** se ordenan, **Then** se ordenan por costo 3 a 1, y el MAE, RMSE y R² se comparan solo dentro de cada familia.
4. **Given** el mejor modelo, **When** se compara con los siguientes, **Then** se informa la diferencia media, en cuántos pliegues gana y diferencia / ruido, y se distingue un empate práctico de una ventaja consistente.

---

### User Story 2 - Recomendar un plazo y medirlo bien, sin filtrar información (Priority: P1)

Como analista, quiero que cada modelo produzca el plazo que recomendaría la app y que las métricas se calculen sobre ese plazo, para que el costo 3 a 1 refleje la decisión real.

**Why this priority**: el costo 3 a 1 solo se minimiza con el percentil 75; si el plazo no se arma bien, el torneo compara cosas distintas.

**Independent Test**: con datos sintéticos de percentil conocido, el plazo recomendado cubre cerca del 75 % de los pedidos; el margen de los modelos de la familia A sale solo de clientes de calibración que no participaron en el ajuste; la identidad «costo 3 a 1 = 4 × pérdida cuantílica de 0,75» se cumple.

**Acceptance Scenarios**:

1. **Given** un modelo de la familia A, **When** se ajusta con un entrenamiento, **Then** se aparta el 20 % de los clientes para calibrar, el modelo se ajusta con el resto, el margen es el percentil 75 de los errores (real − predicción) de los clientes apartados, y ningún cliente está a la vez en el ajuste y la calibración.
2. **Given** un modelo de la familia B o una línea base, **When** se ajusta, **Then** usa todo el entrenamiento y su margen es 0 (ya predice el plazo recomendado).
3. **Given** un plazo recomendado y los días reales, **When** se calculan las métricas, **Then** el costo 3 a 1, el porcentaje que llega tarde y la cobertura salen del plazo recomendado, y MAE, RMSE, R² y sesgo de la predicción del modelo.
4. **Given** un plazo cercano al percentil 75, **When** se calcula la cobertura, **Then** vale cerca de 0,75 (y el porcentaje tarde cerca de 0,25).

---

### User Story 3 - Contar con todos los candidatos construidos igual (Priority: P2)

Como analista, quiero que cada candidato esté definido en un solo lugar con su familia, sus parámetros fijos y si necesita variables escaladas, y que todos pasen por el mismo preprocesamiento dentro de un `Pipeline`.

**Why this priority**: sin esto cada modelo se armaría a mano y el torneo no sería comparable.

**Independent Test**: cada candidato se ajusta con una tabla pequeña y predice sin errores; el pipeline general acepta cualquier modelo y el pipeline de la etapa 4 sigue funcionando igual.

**Acceptance Scenarios**:

1. **Given** la lista de candidatos, **When** se piden, **Then** se obtienen exactamente los 10 definidos en la especificación (2 líneas base, 5 de la familia A, 3 de la familia B), cada uno con su familia.
2. **Given** un modelo cualquiera, **When** se construye su pipeline con los 7 grupos de variables del torneo, **Then** el preprocesador se ajusta solo con entrenamiento y tolera categorías nuevas.
3. **Given** el pipeline de la etapa 4 (regresión lineal por defecto), **When** se usa tras la generalización, **Then** da los mismos resultados que antes.

---

### User Story 4 - Elegir finalistas con una regla escrita (Priority: P2)

Como grupo, queremos que las reglas de empate y de superación de las líneas base estén en código con pruebas, y que la regresión lineal siempre aparezca con su diferencia contra el ganador, para decidir sin interpretar cifras a ojo.

**Why this priority**: es la parte que defenderemos ante el profesor.

**Independent Test**: con resultados de ejemplo hechos a mano, las funciones de comparación y de posiciones dan los valores esperados, incluido el caso sin ruido.

**Acceptance Scenarios**:

1. **Given** dos modelos con los costos por pliegue, **When** se comparan, **Then** se obtiene la diferencia media, los pliegues ganados y diferencia / ruido; una diferencia que gana en 4 de 5 o más y supera 1 vez su ruido es consistente.
2. **Given** los resultados de todos los modelos, **When** se arma la tabla de posiciones, **Then** cada métrica tiene su ranking (1 = mejor) y se ve si el ranking por costo coincide con el de exactitud.
3. **Given** la regresión lineal y el ganador, **When** se reporta su diferencia, **Then** se expresa en días de error por pedido y por cada 1.000 pedidos, sin inventar un valor en dinero.

---

### User Story 5 - Dejar constancia y poder repetirlo (Priority: P3)

Como grupo, queremos un notebook que solo orquesta y un reporte con las tablas, la lectura de la cobertura, la revisión por región y los finalistas con su motivo, para explicarlo en la exposición.

**Why this priority**: obligatorio para el curso (principios IV y VI) pero no bloquea el trabajo técnico.

**Independent Test**: ejecutar el notebook 05 desde cero y obtener las tablas por pliegue y resumen; dos ejecuciones dan la misma tabla.

**Acceptance Scenarios**:

1. **Given** `data/processed/train.parquet`, **When** se ejecuta el notebook 05, **Then** se generan `reports/05_torneo_por_pliegue.csv` y `reports/05_torneo_resumen.csv`.
2. **Given** los finalistas, **When** se revisa el reporte, **Then** muestra para cada uno el MAE y el porcentaje que llega tarde por región del cliente (5 regiones) y advierte dónde rinde peor.

---

### Edge Cases

- Un candidato que no logra ajustarse en un pliegue: el torneo falla con un mensaje claro y no omite el pliegue sin avisar.
- Margen negativo: si un modelo de la familia A sobreestima en promedio en los clientes de calibración, el margen puede ser negativo; se reporta, no se corrige a cero.
- Pocos clientes de calibración: con el 20 % de cada entrenamiento hay miles de clientes; si fueran muy pocos, la función lo advierte.
- Ruido casi cero entre pliegues: diferencia / ruido es «no evaluable», nunca una división por cero.
- Modelos de la familia B: su MAE, RMSE y R² son los del percentil 75, **no son comparables** con los de la familia A; el reporte lo dice y solo compara el costo entre familias.
- Cobertura lejos de 0,75 con costo bajo: se señala como sospecha.
- La partición hacia adelante es una sola: sin ruido medido; se informa como una partición, no como un promedio.
- Variable de cliente (`customer_unique_id`) presente en la tabla de entrada: sirve para agrupar la calibración y los pliegues, y nunca entra al modelo.
- Un candidato demasiado lento: se anota su tiempo y queda fuera si supera lo razonable, en vez de ocultarlo.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: El sistema MUST usar solo `data/processed/train.parquet`; MUST NOT abrir `test.parquet`.
- **FR-002**: El sistema MUST evaluar todos los candidatos con los mismos 5 pliegues por cliente de la etapa 4 (semilla 42) y la misma partición hacia adelante (entrenar hasta 2018-02-28, validar 2018-03-01 a 2018-08-31), y guardar el resultado de cada pliegue.
- **FR-003**: El sistema MUST usar los 7 grupos de variables del torneo (logaritmos, mismo estado, regiones, día de la semana, distancia × mismo estado, razones de flete y varios vendedores), sin la marca de fin de año, y nunca el plazo prometido por Olist.
- **FR-004**: El sistema MUST definir 10 candidatos con parámetros fijos y sin optimizar: línea base de la mediana, línea base del plazo fijo (percentil 75), regresión lineal, Ridge, bosque aleatorio liviano, LightGBM y XGBoost con margen, y LightGBM, XGBoost e HistGradientBoosting con objetivo de percentil 75.
- **FR-005**: El bosque aleatorio liviano MUST usar 100 árboles, profundidad máxima 12, hoja mínima 20, 33 % de variables y 50 % de filas por árbol.
- **FR-006**: Para la familia A, el sistema MUST apartar el 20 % de los clientes del entrenamiento de cada pliegue, ajustar el modelo con el resto y calcular el margen como el percentil 75 de los errores (real − predicción) de los clientes apartados; MUST NOT usar la validación para calcular el margen y MUST verificar que ningún cliente está en el ajuste y en la calibración.
- **FR-007**: Para la familia B y las líneas base, el sistema MUST ajustar con todo el entrenamiento y usar margen 0.
- **FR-008**: El sistema MUST calcular para cada modelo y pliegue: costo medio 3 a 1 y porcentaje tarde y cobertura del plazo recomendado; MAE, RMSE, R² y sesgo de la predicción; y el tiempo de ajuste en segundos.
- **FR-009**: El sistema MUST dejar que el costo 3 a 1 decida al ganador y MUST reportar MAE, RMSE y R² siempre; MUST indicar que las métricas de exactitud de la familia B no son comparables con las de la A.
- **FR-010**: El sistema MUST comparar modelos por pares (diferencia media, pliegues ganados y diferencia / ruido con la desviación muestral) y tratar el ruido casi nulo como «no evaluable».
- **FR-011**: El sistema MUST mostrar una tabla de posiciones (1 = mejor) por métrica.
- **FR-012**: El sistema MUST informar si cada modelo supera a las dos líneas base por más que su ruido.
- **FR-013**: El sistema MUST reportar siempre la diferencia de la regresión lineal contra el ganador en días de error por pedido y por cada 1.000 pedidos, sin asignar un valor en dinero.
- **FR-014**: El sistema MUST calcular, para los finalistas, MAE y porcentaje tarde por región del cliente.
- **FR-015**: El sistema MUST generalizar `build_pipeline` para aceptar cualquier modelo, de modo que la regresión lineal por defecto de la etapa 4 siga dando los mismos resultados.
- **FR-016**: Todo lo que aprende de los datos (imputación, codificación, el modelo y el margen) MUST ajustarse solo con el entrenamiento de cada pliegue.
- **FR-017**: Cada función nueva de `src/olistbr/models/` MUST tener al menos una prueba en `tests/` con un caso normal y uno sucio hechos a mano.
- **FR-018**: El notebook `notebooks/05_torneo.ipynb` MUST solo llamar a funciones de `src/`, regenerar las tablas desde cero y no contener lógica reutilizable.
- **FR-019**: Cada decisión MUST quedar en `reports/05_torneo.md` con su cifra y motivo, y reflejarse en `reports/bitacora_decisiones.md`.

### Key Entities *(include if feature involves data)*

- **Candidato**: un modelo con parámetros fijos, su familia (línea base, A o B) y si necesita variables escaladas.
- **Modelo de plazo (`DeadlineModel`)**: un candidato ajustado junto con su margen; predice un valor y recomienda un plazo.
- **Pliegue**: una partición de validación cruzada armada por cliente (los de la etapa 4).
- **Resultado por pliegue**: las métricas de un candidato en un pliegue.
- **Resumen del torneo**: media y desviación entre pliegues por candidato y métrica, más la partición hacia adelante.
- **Comparación por pares**: diferencia media, pliegues ganados y diferencia / ruido entre dos candidatos.
- **Finalista**: candidato elegido para optimizar en la Fase 2, con su motivo.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Los 10 candidatos tienen una fila con todas las métricas (media ± desviación entre 5 pliegues) y su resultado hacia adelante; cada pliegue de cada candidato está guardado.
- **SC-002**: Ningún cliente aparece a la vez en el entrenamiento y la validación de un pliegue, ni en el ajuste y la calibración de un modelo de la familia A.
- **SC-003**: La prueba final (`test.parquet`) no se abre en ningún punto de la etapa.
- **SC-004**: En datos sintéticos de percentil conocido, el plazo recomendado de la familia A y de la B cubre entre el 70 % y el 80 % de los pedidos, y la identidad «costo = 4 × pérdida cuantílica» se cumple con diferencia menor a 1e-9.
- **SC-005**: Dos ejecuciones seguidas del notebook dan las mismas métricas (misma semilla 42). El tiempo de ajuste cambia de una ejecución a otra, por eso se guarda en un archivo aparte y no entra en la comparación.
- **SC-006**: Quedan 1 o 2 finalistas justificados, con la diferencia de la regresión lineal contra el ganador expresada en días por pedido y por 1.000 pedidos.
- **SC-007**: `pytest tests` termina en verde y cada función nueva tiene su prueba; las pruebas de las etapas anteriores siguen pasando.
- **SC-008**: Cada integrante puede explicar en palabras simples por qué el costo 3 a 1 decide el torneo y por qué el percentil 75 es el plazo óptimo.

## Assumptions

- `data/processed/train.parquet` existe y tiene las columnas de las etapas 3 y 4; las variables derivadas salen de `add_features`.
- Las decisiones Q1 a Q6 vienen de `especificaciones/05-torneo.md` y de la sección 7 de `especificacion-proyecto.md`; no se reabren aquí.
- El costo 3 a 1 se define con `olistbr.models.metrics` (subestimar cuesta 3, sobrestimar 1); el percentil objetivo es 0,75.
- Los parámetros de los candidatos son razonables y fijos; no se optimizan en esta fase.
- El equipo tiene 4 núcleos; los tiempos medidos (un ajuste en un pliegue) están en `especificaciones/05-torneo.md` y el torneo completo debe tardar en el orden de minutos.
- Quedan fuera de alcance: optimizar hiperparámetros, registrar en MLflow/Databricks, la prueba final, la app y la comparación estadística con intervalos por remuestreo (Fase 2 y etapa 7).
- La marca de fin de año se mide aparte, como sensibilidad, solo en los finalistas y fuera de esta fase.
- Los textos y comentarios van en español; los nombres de funciones y archivos de código, en inglés.
