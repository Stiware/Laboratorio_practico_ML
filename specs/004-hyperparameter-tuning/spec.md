# Feature Specification: Optimización de hiperparámetros de OnTrack (etapa 6, Fase 2)

**Feature Branch**: `main` (el grupo trabaja solo en `main`)

**Created**: 2026-11-05

**Status**: Draft

**Input**: User description: "Etapa 6 · Optimización de hiperparámetros de los dos finalistas del torneo (LightGBM cuantil 0,75 y HistGradientBoosting cuantil 0,75), según especificaciones/06-optimizacion.md (cerrada el 5 nov 2026, el grupo aceptó Q1 a Q8). Carpeta: specs/004-hyperparameter-tuning."

> Fuente de las reglas: `especificaciones/06-optimizacion.md` (cerrada el 5 nov 2026) y la sección 7 de `especificacion-proyecto.md`. Si esta especificación y aquellas discrepan, mandan aquellas.
> Esta es la **Fase 2** (lo que corresponde a `tuning.py` del ejemplo CreditSight). Parte de los finalistas de `reports/05_torneo.md`.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Saber si optimizar vale la pena, con una regla escrita antes de mirar (Priority: P1)

Como analista del grupo, quiero comparar, en clientes que la búsqueda nunca vio, cada finalista con sus parámetros de partida (los del torneo) contra sus parámetros optimizados, con un intervalo del 95 % que sortea clientes completos, para decidir con la regla aprobada si uso los optimizados o me quedo con los de partida.

**Why this priority**: es la decisión central de la etapa y la pregunta 8 que hará el instructor. La mejor puntuación de una búsqueda es optimista; sin esta comprobación no se puede afirmar que optimizar ayudó.

**Independent Test**: con dos plazos recomendados conocidos sobre una tabla pequeña (por ejemplo, uno siempre un día mejor que el otro), la diferencia y su intervalo salen como se espera a mano, y la regla devuelve «optimizados» solo cuando todo el intervalo está por debajo de cero.

**Acceptance Scenarios**:

1. **Given** el conjunto de entrenamiento, **When** se separan los clientes reservados, **Then** se aparta el 25 % de los **clientes** (semilla 7), y ningún cliente queda a la vez en la reserva y en los datos de búsqueda.
2. **Given** un finalista con parámetros de partida y optimizados, **When** se ajustan los dos con todos los datos de búsqueda y se evalúan en la reserva, **Then** se informa la diferencia de costo 3 a 1 (optimizado − partida) por pedido y un intervalo del 95 % calculado con 1.000 sorteos de **clientes completos** (semilla 42).
3. **Given** el intervalo, **When** se aplica la regla, **Then** se usan los parámetros optimizados solo si su límite superior es menor que cero; si cruza el cero, toca el cero o queda por encima, se usan los de partida, y el motivo queda escrito.
4. **Given** los dos finalistas, cada uno con sus parámetros elegidos, **When** se comparan en la reserva con el mismo intervalo, **Then** gana el que queda con todo el intervalo a su favor; si el intervalo cruza el cero es un empate práctico y gana LightGBM cuantil 0,75.

---

### User Story 2 - Buscar parámetros dentro de un presupuesto y con rangos razonados (Priority: P1)

Como analista, quiero buscar los hiperparámetros de cada finalista con un número de ensayos fijado antes de empezar, rangos justificados por el conocimiento del algoritmo y la misma métrica que decidió el torneo, para no sobreajustar a la validación ni gastar tiempo de clase.

**Why this priority**: sin búsqueda no hay parámetros optimizados que comprobar.

**Independent Test**: con una tabla pequeña y 2 ensayos, la búsqueda termina, devuelve 2 ensayos con costos finitos, todos los parámetros quedan dentro de sus rangos, el percentil 75 sigue fijo y, con la misma semilla, se repiten los mismos parámetros.

**Acceptance Scenarios**:

1. **Given** los datos de búsqueda (75 % de los clientes), **When** se crean los pliegues, **Then** son 5 pliegues por cliente (semilla 42), los mismos para todos los ensayos y los dos finalistas, y ningún cliente cae en los dos lados de un pliegue.
2. **Given** un ensayo, **When** se proponen sus parámetros, **Then** cada uno está dentro del rango de la sección 4 de `especificaciones/06-optimizacion.md` y el objetivo del modelo sigue siendo el percentil 75 (no se puede cambiar desde los parámetros del ensayo).
3. **Given** el presupuesto, **When** termina la búsqueda, **Then** se hicieron exactamente 20 ensayos por finalista; si la búsqueda se interrumpe y se reanuda, solo se hacen los ensayos que faltan.
4. **Given** HistGradientBoosting cuantil 0,75, **When** se busca, **Then** la parada temprana está apagada y el número de árboles es un parámetro explícito; los parámetros de partida siguen siendo exactamente los del torneo, y se informa cuántos árboles usó en realidad el modelo de partida.

---

### User Story 3 - Dejar registrado cada ensayo para comparar, reproducir y auditar (Priority: P2)

Como grupo, queremos que cada ensayo quede registrado como una ejecución con sus parámetros, sus métricas (media y desviación entre pliegues) y etiquetas, para poder ordenar los ensayos, ver los mejores y mostrarle al instructor dónde están.

**Why this priority**: es parte de la rúbrica (tracking) y de la regla 7 del proyecto; sin registro, a mitad de la búsqueda ya no se sabe cuál era la buena.

**Independent Test**: tras una búsqueda corta, el experimento tiene una ejecución principal por finalista y una ejecución hija por ensayo, cada una con sus parámetros, el costo 3 a 1 con su desviación, las demás métricas y las etiquetas de etapa y modelo.

**Acceptance Scenarios**:

1. **Given** la conexión del proyecto, **When** se configura el seguimiento, **Then** se usa Databricks si la conexión está disponible y, si no, el registro local de respaldo, y se muestra en pantalla cuál quedó (sin imprimir tokens ni el contenido de `.env`).
2. **Given** un ensayo terminado, **When** se registra, **Then** tiene sus hiperparámetros, el costo 3 a 1 (media y desviación), MAE, RMSE, R², sesgo, porcentaje tarde, cobertura y tiempo de ajuste, más las etiquetas `etapa=06_optimizacion`, `modelo` y `metrica_optimizada`.
3. **Given** la búsqueda completa, **When** se exporta, **Then** queda una tabla con una fila por ensayo y un archivo con la decisión de cada finalista (parámetros de partida, optimizados, diferencia, intervalo, decisión y motivo).

---

### User Story 4 - Revisar el modelo elegido hacia adelante y por región, sin decidir con eso (Priority: P3)

Como grupo, queremos ver cómo se comporta el modelo elegido en compras posteriores (marzo a agosto de 2018) y por región del cliente, usando solo clientes reservados, para avisar si el optimizado empeora con el tiempo o en alguna región (principio VIII), sabiendo que la decisión ya la tomó la regla.

**Why this priority**: es una guardia informativa; el torneo mostró que la cobertura cambia de mes a mes (61 % a 93 %).

**Independent Test**: la partición hacia adelante entrena solo con clientes de búsqueda que compraron hasta el 2018-02-28 y valida solo con clientes reservados que compraron del 2018-03-01 al 2018-08-31; ningún cliente está en los dos lados.

**Acceptance Scenarios**:

1. **Given** el modelo de partida y el optimizado de cada finalista, **When** se evalúan hacia adelante, **Then** se informa el costo 3 a 1 y la cobertura mes a mes de cada uno, y se avisa por escrito si el optimizado queda peor que el de partida.
2. **Given** el modelo elegido, **When** se evalúa en la reserva por región del cliente, **Then** se informa el MAE y el porcentaje tarde por región, sin cambiar la decisión.

---

### Edge Cases

- **La optimización no mejora:** el intervalo cruza el cero. Es un resultado válido: se usan los parámetros de partida y se escribe que la mejora no se distingue del ruido.
- **El intervalo toca exactamente el cero** en su límite superior: cuenta como «no está por debajo de cero» y se usan los de partida.
- **Plazos idénticos** (partida = optimizado): diferencia 0 e intervalo [0; 0]; la regla devuelve «partida».
- **Entradas de distinto largo** (días reales, plazos y clientes): se rechazan con un mensaje claro.
- **Clientes vacíos** en la columna de clientes: se rechazan, porque no se pueden sortear clientes completos.
- **Un ensayo falla o da un costo no finito:** se registra como fallido y no puede quedar como el mejor.
- **La búsqueda se corta a la mitad:** al reanudar, se completan solo los ensayos que faltan hasta 20.
- **Databricks no conecta o se agotó la cuota:** se sigue con el registro local y se avisa en pantalla.
- **Los parámetros del ensayo traen otro percentil:** el modelo sigue en 0,75 o falla con un mensaje claro; nunca optimiza otro percentil sin avisar.
- **Mes sin pedidos de clientes reservados** en la partición hacia adelante: se informa el mes vacío; no se inventa cobertura.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Se DEBE usar solo `data/processed/train.parquet`; el conjunto de prueba final no se abre en ningún paso de la etapa (principio III).
- **FR-002**: Se DEBE separar el 25 % de los clientes como reserva (semilla 7), con una verificación de que la reserva y los datos de búsqueda no comparten clientes.
- **FR-003**: La búsqueda DEBE usar 5 pliegues por cliente (semilla 42) creados una sola vez sobre los datos de búsqueda e iguales para todos los ensayos y finalistas.
- **FR-004**: Se DEBEN optimizar exactamente dos finalistas, LightGBM cuantil 0,75 y HistGradientBoosting cuantil 0,75, con los 7 grupos de variables del torneo y sin la marca de fin de año, el historial por vendedor ni el plazo prometido por Olist.
- **FR-005**: Cada finalista DEBE tener un espacio de búsqueda con los parámetros y rangos de la sección 4 de `especificaciones/06-optimizacion.md`, y sus parámetros de partida DEBEN ser idénticos a los del torneo.
- **FR-006**: El objetivo de los dos modelos DEBE quedar fijo en el percentil 75 en todos los ensayos.
- **FR-007**: En la búsqueda, HistGradientBoosting DEBE tener la parada temprana apagada y el número de árboles como parámetro explícito; se DEBE informar cuántos árboles usó realmente su modelo de partida.
- **FR-008**: La búsqueda DEBE minimizar el costo 3 a 1 medio de los pliegues del plazo recomendado, con un muestreador reproducible (semilla 42) y un presupuesto fijo de 20 ensayos por finalista.
- **FR-009**: La búsqueda DEBE poder reanudarse desde lo guardado y hacer solo los ensayos que faltan.
- **FR-010**: Cada ensayo DEBE quedar registrado como ejecución hija dentro de una ejecución por finalista, con sus hiperparámetros, todas las métricas del torneo (media y desviación entre pliegues), el tiempo de ajuste y las etiquetas de etapa, modelo y métrica optimizada, en el experimento `ontrack-optimizacion`.
- **FR-011**: El seguimiento DEBE usar el módulo de seguimiento existente del proyecto, sin reescribirlo: Databricks si hay conexión y respaldo local si no, mostrando cuál quedó y sin exponer secretos.
- **FR-012**: La mejor puntuación de la búsqueda DEBE mostrarse marcada como optimista y no reportarse como resultado del modelo.
- **FR-013**: Para cada finalista, se DEBEN ajustar dos modelos (partida y optimizado) con todos los datos de búsqueda y compararlos en la reserva con la diferencia de costo 3 a 1 por pedido y un intervalo del 95 % que sortea **clientes completos** (1.000 sorteos, semilla 42).
- **FR-014**: La regla de decisión DEBE estar implementada y probada: «optimizados» solo si el límite superior del intervalo es menor que cero; en cualquier otro caso, «partida», siempre con su motivo.
- **FR-015**: El modelo elegido entre los dos finalistas DEBE salir de la misma comparación con intervalo en la reserva; si el intervalo cruza el cero, gana LightGBM cuantil 0,75.
- **FR-016**: Se DEBE informar, sin que cambie la decisión, el costo 3 a 1 y la cobertura mes a mes hacia adelante (entrenar con clientes de búsqueda hasta el 2018-02-28 y validar con clientes reservados del 2018-03-01 al 2018-08-31), y el MAE y el porcentaje tarde por región del modelo elegido en la reserva.
- **FR-017**: Todo lo que aprende de los datos (preprocesador y modelo) DEBE ajustarse dentro de un `Pipeline` solo con el entrenamiento de cada pliegue o de cada comparación.
- **FR-018**: Se DEBEN guardar `reports/06_best_params.json` (por finalista: parámetros de partida y optimizados, diferencia, intervalo, decisión y motivo, más el modelo elegido) y `reports/06_ensayos.csv` (una fila por ensayo), y redactar `reports/06_optimizacion.md` con cifras que salgan del código.
- **FR-019**: La lógica reutilizable DEBE vivir en `src/olistbr/models/` con pruebas en `tests/`; el notebook `06_optimizacion.ipynb` solo orquesta y lee Parquet (no `train.joblib`).
- **FR-020**: Se DEBEN rechazar con un mensaje claro las entradas de distinto largo y los clientes vacíos en el cálculo del intervalo.

### Key Entities

- **Datos de búsqueda**: el 75 % de los clientes de entrenamiento; con ellos se hacen los pliegues de la búsqueda y se ajustan los modelos que se comparan.
- **Reserva**: el 25 % de los clientes de entrenamiento; la búsqueda nunca los ve; solo sirven para la comprobación honesta, la elección entre finalistas y las revisiones informativas.
- **Finalista**: un modelo del torneo con su espacio de búsqueda, su constructor (con el percentil 75 fijo) y sus parámetros de partida.
- **Ensayo**: una combinación de hiperparámetros evaluada con los 5 pliegues de búsqueda; tiene sus métricas por pliegue y queda registrado.
- **Comprobación**: por finalista, la diferencia de costo (optimizado − partida) en la reserva con su intervalo del 95 % y la decisión de la regla.
- **Decisión final**: el modelo elegido entre los dos finalistas, con sus parámetros, la cifra que la respalda y su motivo.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: El experimento registrado tiene exactamente 20 ensayos por finalista (40 en total), cada uno con sus parámetros, métricas y etiquetas.
- **SC-002**: Reserva y datos de búsqueda comparten 0 clientes, y los pliegues de búsqueda comparten 0 clientes entre sus dos lados.
- **SC-003**: Para cada finalista hay una diferencia de costo con un intervalo del 95 % por cliente y una decisión («optimizados» o «partida») que coincide con la regla escrita antes de mirar.
- **SC-004**: Queda un único modelo elegido, con la cifra que lo respalda y el motivo escrito, listo para la validación final de la etapa 7.
- **SC-005**: La prueba final se abrió 0 veces (una búsqueda del nombre del archivo de prueba en el notebook y en el código de la etapa no da resultados).
- **SC-006**: Volver a correr el notebook con las mismas semillas da los mismos parámetros elegidos y las mismas cifras de costo (el tiempo de ajuste va aparte).
- **SC-007**: Todas las pruebas automáticas del proyecto pasan, incluidas las nuevas de la etapa.
- **SC-008**: Quien no escribió el código puede responder, con las cifras del informe, cuántos ensayos se hicieron y por qué, qué clientes se reservaron, qué regla se usó y qué resultado dio.

## Assumptions

- `data/processed/train.parquet` existe (se regenera con el notebook 03 a partir de los CSV de Olist en `data/raw/`); en la máquina donde se escribió esta especificación no estaba, así que hay que regenerarlo antes de correr el notebook.
- Los tiempos de la búsqueda (≈ 20 a 35 minutos en total) son estimaciones a partir de los tiempos del torneo, no mediciones.
- La lista de vendedores frecuentes de la etapa 3 se contó con todo el entrenamiento, incluida la reserva. Solo usa conteos de ítems (no el objetivo), así que se acepta como limitación anotada; lo revisa el `auditor-de-fuga`.
- La reserva se usa dos veces (partida contra optimizado, y luego entre finalistas). Sigue siendo validación, no la prueba; el resultado del modelo elegido en la reserva es algo optimista.
- Recalibrar con datos recientes, la marca de fin de año, el historial por vendedor y un indicador por región quedan fuera de esta etapa (pasan a la especificación de la etapa 7).
- El control de datos entre familias no aplica, porque los dos finalistas son de la familia B (percentil 75 directo, sin margen).
- Se reutilizan las funciones existentes del proyecto (partición por cliente, pliegues, pipeline, evaluación por pliegue, métricas y seguimiento) en lugar de duplicarlas.
