# Feature Specification: Validación final y registro del modelo de OnTrack (etapas 7 y 8)

**Feature Branch**: `main` (el grupo trabaja solo en `main`)

**Created**: 2026-11-05

**Status**: Draft

**Input**: User description: "Etapas 7 y 8: validación final (abrir la prueba una sola vez con el modelo congelado) e interpretación, y registro del modelo en Unity Catalog con alias champion, según especificaciones/07-validacion.md y 08-registro.md (cerradas el 5 nov 2026; el grupo aceptó todas las propuestas)."

> Fuente de las reglas: `especificaciones/07-validacion.md` y `especificaciones/08-registro.md` (cerradas el 5 nov 2026) y la sección 7 de `especificacion-proyecto.md`.
> Si esta especificación y aquellas discrepan, mandan aquellas. Las dos etapas van en una sola pieza porque la 8 registra exactamente el modelo que evalúa la 7.
> **Prerrequisito:** la etapa 6 cerrada (`reports/06_best_params.json` con la decisión final y el notebook 06 ejecutado).

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Saber cuánto vale de verdad el modelo, abriendo la prueba una sola vez (Priority: P1)

Como grupo, queremos congelar el modelo final por escrito, ajustarlo con todo el entrenamiento y recién entonces abrir la prueba **una sola vez**, para medir su costo 3 a 1 frente al plazo fijo del percentil 75 con un intervalo por cliente, y saber si el modelo mejora la regla simple con confianza.

**Why this priority**: es la cifra honesta del proyecto; todo lo anterior fue validación. Sin ella no hay ficha, registro ni app.

**Independent Test**: con archivos temporales, la función que abre la prueba deja una constancia y se niega a abrirla una segunda vez; con plazos inventados, la tabla de la prueba da las métricas y la diferencia contra el baseline calculadas a mano.

**Acceptance Scenarios**:

1. **Given** las decisiones de la sección 2 de `especificaciones/07-validacion.md`, **When** se prepara la validación, **Then** quedan escritas con fecha y hora en `reports/07_validacion.md` antes de leer la prueba.
2. **Given** el entrenamiento completo, **When** se ajusta el modelo final, **Then** es LightGBM cuantil 0,75 con los parámetros optimizados de `reports/06_best_params.json`, se guarda en `models/final_pipeline.joblib` y no se usa la prueba.
3. **Given** la prueba cerrada, **When** se abre, **Then** queda constancia de fecha y hora en `reports/prueba_abierta.txt`; si la constancia ya existe, la apertura falla con un mensaje claro.
4. **Given** los plazos del modelo y de los baselines en la prueba, **When** se comparan, **Then** se informa el costo 3 a 1, el % tarde y la cobertura de cada uno con su intervalo del 95 % (1.000 sorteos de clientes, semilla 42), y la diferencia `costo(modelo) − costo(plazo fijo p75)` con su intervalo; «mejora con confianza» solo si todo el intervalo queda bajo cero.

---

### User Story 2 - Ver dónde y cuándo falla el modelo (Priority: P2)

Como grupo, queremos el % tarde frente al 25 % esperado por región y por mes de compra, los días de holgura y de atraso, la sensibilidad del costo a otras proporciones y el escenario con los 64 pedidos extremos, para declarar los límites del modelo (principio VIII).

**Why this priority**: un buen promedio puede esconder una región o un mes donde el plazo no sirve.

**Independent Test**: con plazos inventados, la holgura y el atraso coinciden con el cálculo a mano y la sensibilidad con proporción 1 a 1 es igual al MAE.

**Acceptance Scenarios**:

1. **Given** el modelo y la prueba, **When** se segmenta, **Then** por región del cliente y por mes de compra se informan pedidos, % tarde, cobertura, MAE, holgura y atraso.
2. **Given** las proporciones 1, 2, 3, 5 y 9 a 1, **When** se calcula la sensibilidad, **Then** se informa el costo del modelo y del baseline principal con cada una.
3. **Given** los 64 pedidos extremos, **When** se agregan a la prueba, **Then** se informan las mismas métricas como escenario pesimista, separado del resultado principal.

---

### User Story 3 - Explicar qué mueve el plazo, con sus límites (Priority: P2)

Como grupo, queremos saber qué variables mueven más el plazo recomendado y en qué dirección, declarando que la explicación es del modelo y no del mundo, y que la ubicación del cliente es un atributo sensible.

**Why this priority**: lo pide la guía y es parte de la ficha del modelo.

**Independent Test**: con un modelo de juguete donde solo una variable importa, esa variable sale primera.

**Acceptance Scenarios**:

1. **Given** el modelo final y una muestra de 2.000 pedidos de la prueba (semilla 42), **When** se calculan los valores SHAP, **Then** se informan las 10 variables con mayor peso medio absoluto y su signo medio.

---

### User Story 4 - Registrar el modelo evaluado y poder recuperarlo (Priority: P1)

Como grupo, queremos registrar en Unity Catalog exactamente el modelo evaluado, con sus parámetros, las métricas de la prueba, la firma, un ejemplo de entrada sin identificadores y etiquetas, ponerle el alias `champion` y comprobar que el modelo recuperado del registro predice igual que el local.

**Why this priority**: el registro responde «¿qué modelo usamos y en qué versión?»; la app (etapa 9) se exporta desde el `champion`.

**Independent Test**: con MLflow local en una carpeta temporal y un modelo de juguete, queda una versión registrada, el alias apunta a la última versión, el ejemplo no tiene identificadores y el modelo recuperado predice igual.

**Acceptance Scenarios**:

1. **Given** `models/final_pipeline.joblib` y `reports/07_metricas_test.json`, **When** se registra, **Then** queda una ejecución `modelo-final` en el experimento `ontrack-modelo-final` y una versión de `workspace.default.ontrack_plazo_entrega` con parámetros, métricas, firma, ejemplo y etiquetas.
2. **Given** el ejemplo de entrada, **When** se arma, **Then** tiene 3 pedidos solo con las columnas que usa el modelo, sin `customer_unique_id`, `order_id` ni el plazo prometido.
3. **Given** la versión registrada, **When** se pone el alias, **Then** `champion` apunta a la última versión.
4. **Given** el modelo recuperado con `@champion`, **When** se comparan sus plazos con los del modelo local en 500 pedidos de la prueba, **Then** coinciden, y el costo 3 a 1 en toda la prueba coincide con el de `07_metricas_test.json`.
5. **Given** que Databricks falla, **When** se registra, **Then** se usa el respaldo local y se escribe el motivo.

### Edge Cases

- **La prueba ya se abrió** (existe la constancia): la apertura falla; no se vuelve a abrir para «arreglar» nada.
- **El modelo no mejora al baseline con confianza** (el intervalo cruza el cero): se reporta tal cual; no se cambia el modelo.
- **Un segmento con pocos pedidos** (por ejemplo, el Norte): se informa con su número de pedidos; no se afirma que esté peor si no lo respalda el error de muestreo.
- **Entradas vacías, de distinto largo o no finitas** en las funciones de métricas: fallan con un mensaje claro.
- **El catálogo de Unity Catalog no acepta el registro** (permiso o cuota): respaldo local con el motivo escrito.
- **El ejemplo de entrada trae una columna de identificador**: la función la quita; una prueba lo verifica.
- **Las salidas del notebook muestran el correo o el enlace del workspace**: no deben (`MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT`).

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Las decisiones congeladas (modelo, parámetros, sin recalibración, métrica, baseline, segmentos, interpretación, lo no evaluado) DEBEN quedar escritas con fecha y hora antes de leer la prueba.
- **FR-002**: El modelo final DEBE ser LightGBM cuantil 0,75 con los parámetros optimizados de `reports/06_best_params.json`, ajustado solo con `train.parquet` dentro de un `Pipeline`, con las 7 variables del torneo, y guardado en `models/final_pipeline.joblib` antes de abrir la prueba.
- **FR-003**: La prueba DEBE abrirse una sola vez por una función que deja constancia (fecha y hora) en `reports/prueba_abierta.txt` y falla si la constancia existe.
- **FR-004**: Se DEBEN informar en la prueba, para el modelo y para los baselines (plazo fijo del percentil 75 del entrenamiento, siempre la mediana del entrenamiento y el plazo prometido por Olist), el costo 3 a 1, el % tarde, la cobertura, el MAE y el sesgo del plazo, con intervalos del 95 % sorteando clientes completos (1.000 sorteos, semilla 42).
- **FR-005**: Se DEBE informar la diferencia `costo(modelo) − costo(plazo fijo p75)` con su intervalo; el modelo «mejora con confianza» solo si todo el intervalo queda bajo cero.
- **FR-006**: Se DEBEN informar, por región del cliente y por mes de compra: pedidos, % tarde, cobertura, MAE, días de holgura media y días de atraso medio.
- **FR-007**: Se DEBE informar la sensibilidad del costo del modelo y del baseline principal con las proporciones 1, 2, 3, 5 y 9 a 1.
- **FR-008**: Se DEBE informar un escenario con la prueba más los 64 pedidos extremos, separado del resultado principal.
- **FR-009**: Se DEBEN informar las 10 variables con mayor valor SHAP medio absoluto (muestra de 2.000 pedidos de la prueba, semilla 42) y su signo medio, con los límites de la explicación y la ubicación del cliente declarada como atributo sensible.
- **FR-010**: No se DEBE cambiar nada del modelo después de abrir la prueba; la marca de fin de año, el historial por vendedor y la probabilidad de retraso por pedido se declaran como no evaluados.
- **FR-011**: Se DEBE registrar exactamente `models/final_pipeline.joblib` en Unity Catalog (`workspace.default.ontrack_plazo_entrega`), dentro de la ejecución `modelo-final` del experimento `ontrack-modelo-final`, con parámetros, métricas de la prueba, firma, ejemplo de entrada y etiquetas, guardado con `cloudpickle`.
- **FR-012**: El ejemplo de entrada DEBE tener 3 pedidos del entrenamiento solo con las columnas que usa el modelo, sin identificadores ni el plazo prometido.
- **FR-013**: El alias `champion` DEBE apuntar a la última versión registrada.
- **FR-014**: El modelo recuperado con `@champion` DEBE predecir igual que el local en 500 pedidos de la prueba y dar el mismo costo 3 a 1 en toda la prueba que `07_metricas_test.json`; la prueba se lee para esto sin decidir nada.
- **FR-015**: Si Databricks falla, el registro DEBE usar el respaldo local y escribir el motivo.
- **FR-016**: Las salidas de los notebooks y los archivos del repositorio NO DEBEN contener el correo de la cuenta, el enlace del workspace ni el contenido de `.env`.
- **FR-017**: La lógica reutilizable DEBE vivir en `src/olistbr/models/validation.py` y `src/olistbr/tracking/registry.py` con pruebas; los notebooks 07 y 08 solo orquestan y se generan con scripts en `scripts/generadores/`.
- **FR-018**: Se DEBEN guardar `reports/07_metricas_test.json`, `reports/07_segmentos.csv`, `reports/07_shap.csv`, `reports/08_registro.json` y completar `reports/07_validacion.md` y `reports/08_registro.md` con cifras del código.

### Key Entities

- **Decisiones congeladas**: lo que se fija antes de abrir la prueba, con fecha y hora.
- **Modelo final**: el pipeline ajustado con todo el entrenamiento; se guarda, se evalúa y se registra sin cambios.
- **Constancia de apertura**: archivo con la fecha y hora en que se abrió la prueba.
- **Plan de cada regla**: el plazo recomendado por el modelo y por cada baseline, pedido a pedido.
- **Resultado de la prueba**: métricas con intervalo, diferencia contra el baseline, segmentos, sensibilidad, escenario con extremos y SHAP.
- **Versión registrada**: nombre, número de versión, alias, plataforma y resultado de la ida y vuelta.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: La prueba se abrió exactamente una vez (una constancia; un segundo intento falla).
- **SC-002**: Hay una cifra de costo 3 a 1 del modelo en la prueba con su intervalo, y la diferencia contra el plazo fijo con su intervalo, más la conclusión que dicta la regla.
- **SC-003**: Hay tablas por región (5) y por mes, la sensibilidad con 5 proporciones, el escenario con extremos y las 10 variables de SHAP.
- **SC-004**: Hay una versión registrada con alias `champion`, y la ida y vuelta da plazos iguales en 500 pedidos y el mismo costo en toda la prueba.
- **SC-005**: Ningún archivo nuevo ni salida de notebook contiene el correo, el enlace del workspace ni identificadores de cliente en el ejemplo de entrada.
- **SC-006**: Todas las pruebas automáticas pasan, incluidas las nuevas.
- **SC-007**: Quien no escribió el código puede explicar con las cifras del reporte si el modelo mejora al plazo fijo, dónde falla y qué diferencia hay entre tracking y registro.

## Assumptions

- La etapa 6 se cierra antes de empezar (reproducibilidad comprobada y notebook 06 con salidas).
- El `.gitignore` vuelve a excluir `data/processed/` antes de abrir la prueba (hoy tiene esas líneas comentadas; lo restaura el grupo).
- `workspace.default` existe en Databricks (comprobado el 5 nov 2026) y la conexión del `.env` funciona.
- La prueba se partió por cliente, no por fecha: mide generalización a clientes nuevos del mismo periodo, no a meses futuros.
- El modelo registrado espera las columnas que produce `add_features`; convertir los datos del formulario a esas columnas es parte de la etapa 9.
- Exportar `models/model.joblib`, escribir `models/input_schema.json`, adaptar `predict.py` a regresión y la ficha del modelo quedan para la etapa 9.
