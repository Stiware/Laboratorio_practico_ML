# Feature Specification: Variables nuevas de OnTrack y su medición (etapa 4)

**Feature Branch**: `main` (el grupo trabaja solo en `main`)

**Created**: 2026-10-31

**Status**: Draft

**Input**: User description: "Etapa 4 · Variables (feature engineering) de OnTrack según `especificaciones/04-features.md` (cerrada). Crear las variables nuevas aprobadas (logaritmos, mismo estado, regiones, día de la semana, marca de temporada alta de fin de año, interacción distancia por mismo estado, historial por vendedor/categoría/ruta con TargetEncoder dentro del Pipeline, razones de flete y marca de varios vendedores), el preprocesador dentro de un Pipeline, y medir cada idea con 5 pliegues por cliente (GroupKFold por `customer_unique_id`), regla mejora/ruido > 1 y prueba hacia adelante (entrenar hasta 2018-02-28, validar 2018-03-01 a 2018-08-31), usando solo `data/processed/train.parquet` y LinearRegression. `promised_days` solo como referencia, nunca entra al modelo final. Código en `src/olistbr/features/engineering.py` con pruebas en `tests/test_features.py`, notebook 04 que solo orquesta y `reports/04_features.md` con la decisión y cifra de cada idea."

> Fuente de las reglas: `especificaciones/04-features.md` (cerrada el 31 oct 2026) y la sección 7 de `especificacion-proyecto.md`. Si esta especificación y aquellas discrepan, mandan aquellas.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Saber qué ideas de variables sirven y cuáles no (Priority: P1)

Como analista del grupo, quiero una tabla que compare cada idea de variable contra una referencia, con la misma validación y con su cifra, para decidir con números cuáles se conservan y cuáles se descartan.

**Why this priority**: es el entregable de la etapa (la «tabla de ideas medidas») y de ella depende qué variables llegan al torneo de modelos.

**Independent Test**: ejecutar la comparación y comprobar que cada idea aprobada tiene una fila con media y desviación entre pliegues del error en días, el costo 3 a 1, la columna `mejora / ruido`, el resultado de la prueba hacia adelante y la decisión (conservar o descartar) según la regla escrita.

**Acceptance Scenarios**:

1. **Given** el conjunto de entrenamiento y las variables base, **When** se mide la referencia, **Then** se obtiene el error medio en días y el costo medio 3 a 1 de cada uno de los 5 pliegues, con su media y desviación.
2. **Given** una idea de variable, **When** se mide con los mismos pliegues y el mismo modelo, **Then** se reporta la mejora respecto a la referencia (media de la diferencia), el ruido (desviación de esa diferencia entre pliegues) y su cociente.
3. **Given** una idea cuya mejora entre pliegues supera el ruido pero cambia de signo en la prueba hacia adelante, **When** se aplica la regla de decisión, **Then** se marca como descartada y se anota el motivo con ambas cifras.
4. **Given** el plazo prometido por Olist, **When** se mide, **Then** aparece como una fila de referencia («no usable en producción») y no entra a ningún conjunto de variables del modelo final.

---

### User Story 2 - Crear las variables aprobadas sin filtrar información (Priority: P1)

Como analista, quiero que cada variable aprobada se calcule con una función pequeña que use solo lo que se conoce al comprar, para que el modelo no «vea el futuro» y la app pueda calcularlas.

**Why this priority**: sin las variables no hay nada que medir; y calcularlas mal (con información del futuro o dividiendo por cero) invalida todo.

**Independent Test**: con tablas de ejemplo hechas a mano, cada variable da el valor esperado; división por cero y vacíos dan vacío (nunca infinito); la tabla de entrada no se modifica.

**Acceptance Scenarios**:

1. **Given** una distancia, un flete, un precio, un peso y un volumen, **When** se calculan sus versiones logarítmicas, **Then** coinciden con `log(1 + valor)` y un vacío sigue vacío.
2. **Given** un cliente y un vendedor, **When** se calcula «mismo estado», **Then** vale 1 si los estados son iguales y 0 si no.
3. **Given** el estado del cliente y del vendedor, **When** se calcula la región, **Then** cada uno de los 27 estados cae en una de 5 regiones según una tabla fija.
4. **Given** una fecha de compra, **When** se calcula el día de la semana y la marca de jueves a sábado, **Then** el lunes es 0 y jueves, viernes y sábado quedan marcados.
5. **Given** una fecha de compra, **When** se calcula la marca de temporada alta de fin de año, **Then** vale 1 desde el lunes de la semana de Black Friday hasta el 31 de diciembre, y 0 en el resto.
6. **Given** un precio de cero o vacío, **When** se calcula la razón flete/precio, **Then** el resultado es vacío.
7. **Given** un pedido con más ítems que los del vendedor principal, **When** se calcula «varios vendedores», **Then** vale 1.

---

### User Story 3 - Preparar los datos para el modelo sin fuga (Priority: P2)

Como analista, quiero que todo lo que depende de los datos (medianas, categorías, escalado y promedios históricos por vendedor, categoría y ruta) se ajuste solo con el entrenamiento de cada pliegue, para que la medición sea honesta.

**Why this priority**: es la regla central de la constitución (principio III); sin ella, las cifras de la tabla serían optimistas.

**Independent Test**: el preparador de datos se ajusta con un conjunto, transforma otro con categorías que no vio y deja una matriz sin vacíos; el promedio histórico de un cliente de validación no usa su propio objetivo.

**Acceptance Scenarios**:

1. **Given** vacíos en peso, volumen o distancia, **When** se ajusta el preparador con entrenamiento, **Then** los vacíos se completan con la mediana del entrenamiento y se agrega un indicador de vacío.
2. **Given** un vendedor, una categoría o una ruta que no aparece en entrenamiento, **When** se transforma el conjunto de validación, **Then** no falla y queda con un valor neutro.
3. **Given** las variables históricas por vendedor, categoría y ruta, **When** se calculan para el entrenamiento, **Then** cada fila se calcula sin usar su propio objetivo.

---

### User Story 4 - Comprobar que la mejora sirve hacia adelante (Priority: P2)

Como analista, quiero entrenar con las compras hasta febrero de 2018 y validar con marzo a agosto de 2018, para confirmar que una mejora no es solo la tendencia entre años.

**Why this priority**: el EDA mostró que la tendencia entre 2017 y 2018 puede confundirse con la temporada; esta guardia evita conservar variables que no sirven al predecir lo que viene.

**Independent Test**: la partición hacia adelante respeta las fechas (nada posterior a 2018-02-28 en entrenamiento), no comparte clientes entre los dos lados y no usa el conjunto de prueba final.

**Acceptance Scenarios**:

1. **Given** el conjunto de entrenamiento, **When** se arma la prueba hacia adelante, **Then** el entrenamiento solo tiene compras hasta 2018-02-28, la validación solo de 2018-03-01 a 2018-08-31 y ningún cliente aparece en los dos lados.
2. **Given** la marca de temporada alta de fin de año, **When** se evalúa hacia adelante, **Then** se informa que la validación no contiene esa temporada (los datos terminan en agosto) y que la conclusión descansa en una sola temporada observada.

---

### User Story 5 - Dejar constancia y poder repetirlo (Priority: P3)

Como grupo, queremos un notebook que solo orquesta y un reporte con la decisión y la cifra de cada idea (también las descartadas), para explicarlo en la presentación y en el informe.

**Why this priority**: obligatorio para el curso (principios IV y VI) pero no bloquea el trabajo técnico.

**Independent Test**: ejecutar el notebook desde cero y obtener la tabla y el reporte sin pasos manuales; el notebook no contiene lógica reutilizable.

**Acceptance Scenarios**:

1. **Given** `data/processed/train.parquet`, **When** se ejecuta el notebook 04, **Then** se genera la tabla de ideas con media, desviación, `mejora / ruido` y decisión.
2. **Given** las cifras de la tabla, **When** se revisa `reports/04_features.md`, **Then** cada idea (conservada o descartada) tiene su cifra y su motivo, y se anotan las limitaciones.

---

### Edge Cases

- Pedidos con distancia vacía (480 en la etapa 3): las variables derivadas de la distancia quedan vacías y el preparador las completa solo con la mediana del entrenamiento.
- Peso o volumen vacíos: sus razones y logaritmos quedan vacíos; nunca se calcula con cero.
- Vendedores agrupados como «otros»: el promedio histórico los trata como una sola categoría.
- Un pliegue de validación con un vendedor o ruta nuevos: valor neutro, sin error.
- Fechas de compra fuera del periodo 2017-01 a 2018-08: no deberían existir (ya se filtraron); si aparecieran, se informa en vez de ocultarlas.
- La marca de temporada alta no tiene ejemplos positivos en la prueba hacia adelante: se informa y no se usa esa prueba para descartarla.
- Una idea con ruido casi cero: el cociente `mejora / ruido` no debe dividir por cero; se informa «no evaluable».
- Pedidos del mismo cliente en la validación y el entrenamiento: nunca; se verifica con un `assert`.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: El sistema MUST leer solo `data/processed/train.parquet`; MUST NOT abrir `test.parquet` ni los CSV originales en esta etapa.
- **FR-002**: El sistema MUST crear las variables aprobadas de `especificaciones/04-features.md` §4.2 que no aprenden de los datos: logaritmos (distancia, flete, precio, peso, volumen), «mismo estado», región del cliente y del vendedor, día de la semana y marca de jueves a sábado, marca de temporada alta de fin de año, interacción distancia × mismo estado, razón flete/precio, densidad y marca de varios vendedores.
- **FR-003**: Cada variable MUST usar solo información que se conoce al comprar; el plazo prometido por Olist, las fechas de entrega, el estado del pedido, las reseñas y los identificadores MUST NOT entrar como entradas del modelo.
- **FR-004**: El cálculo de variables MUST devolver una copia (sin modificar la tabla de entrada) y MUST dar vacío, nunca infinito, ante división por cero o dato ausente.
- **FR-005**: La región MUST salir de una tabla fija de los 27 estados en 5 regiones (Norte, Nordeste, Centro-Oeste, Sudeste, Sur); un estado fuera de la tabla MUST quedar como vacío.
- **FR-006**: La marca de temporada alta MUST calcularse con calendario fijo (desde el lunes de la semana de Black Friday, el cuarto viernes de noviembre, hasta el 31 de diciembre), no con valores aprendidos de los datos.
- **FR-007**: Los promedios históricos por vendedor, categoría y ruta (estado del vendedor → estado del cliente) MUST ajustarse solo con el entrenamiento de cada pliegue y MUST calcular cada fila de entrenamiento sin usar su propio objetivo.
- **FR-008**: El preparador de datos MUST completar vacíos numéricos con la mediana del entrenamiento y agregar un indicador de vacío, escalar las numéricas y codificar las categóricas dejando agrupadas las poco frecuentes, todo ajustado solo con entrenamiento y tolerando categorías no vistas.
- **FR-009**: El sistema MUST permitir activar y desactivar cada grupo de ideas, para medirlas una por una contra la misma referencia.
- **FR-010**: El sistema MUST crear 5 pliegues por cliente (`customer_unique_id`) una sola vez y reutilizarlos en todas las comparaciones, verificando que ningún cliente aparece en los dos lados de un pliegue.
- **FR-011**: El sistema MUST medir la referencia y cada idea con el mismo modelo (regresión lineal) y reportar por idea: media y desviación entre pliegues del error absoluto medio en días y del costo medio 3 a 1, y la columna `mejora / ruido`.
- **FR-012**: El sistema MUST aplicar la regla de decisión escrita: una idea se conserva solo si se conoce al predecir, `mejora / ruido` es mayor que 1 con el costo 3 a 1 apuntando en la misma dirección, y la mejora no cambia de signo en la prueba hacia adelante.
- **FR-013**: El sistema MUST realizar la prueba hacia adelante dentro del conjunto de entrenamiento (entrenar con compras hasta 2018-02-28, validar con 2018-03-01 a 2018-08-31, sin compartir clientes).
- **FR-014**: El plazo prometido por Olist MUST medirse una sola vez como referencia y MUST NOT incluirse en el conjunto de variables del modelo final.
- **FR-015**: Todas las ideas, conservadas o descartadas, MUST quedar en `reports/04_features.md` con su cifra y su motivo, junto con las limitaciones conocidas (tendencia entre años y una sola temporada de fin de año observada).
- **FR-016**: Cada función de `src/olistbr/features/` MUST tener al menos una prueba en `tests/test_features.py` con un caso normal y uno sucio hechos a mano (incluyendo división por cero y vacíos).
- **FR-017**: El notebook `notebooks/04_features.ipynb` MUST solo llamar a funciones de `src/`, regenerar la tabla desde cero y no contener lógica reutilizable.
- **FR-018**: Cada decisión MUST reflejarse en `reports/bitacora_decisiones.md`.

### Key Entities *(include if feature involves data)*

- **Pedido**: una fila de `train.parquet`, con las variables base de la etapa 3 y el objetivo `dias_entrega`.
- **Idea de variable (hipótesis)**: una variable o grupo de variables con su definición, razón, momento en que se conoce y regla de decisión.
- **Grupo de ideas activas**: el interruptor que determina qué variables entran en una medición.
- **Pliegue**: una partición de validación cruzada, armada por cliente.
- **Resultado de una idea**: media y desviación del error y del costo, `mejora / ruido`, resultado hacia adelante y decisión.
- **Referencia**: la medición con las variables base, contra la que se compara cada idea.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Las 10 ideas aprobadas tienen una fila en la tabla de resultados, cada una con media, desviación, `mejora / ruido`, resultado hacia adelante y decisión escrita.
- **SC-002**: Ningún cliente aparece a la vez en el entrenamiento y la validación de ninguno de los 5 pliegues ni de la prueba hacia adelante.
- **SC-003**: La prueba final (`test.parquet`) no se abre en ningún punto de la etapa.
- **SC-004**: Ninguna variable de entrada del modelo final depende de información posterior a la compra, del plazo prometido por Olist ni de identificadores.
- **SC-005**: Todas las cifras del reporte se pueden reproducir ejecutando el notebook desde cero, y dos ejecuciones seguidas dan la misma tabla.
- **SC-006**: Cada idea descartada queda con su cifra y su motivo; el reporte también dice cuántas de las 10 ideas se conservaron.
- **SC-007**: `pytest tests` termina en verde y cada función de variables tiene su prueba.
- **SC-008**: Cada integrante del grupo puede explicar en palabras simples al menos una variable y por qué se conservó o se descartó.

## Assumptions

- `data/processed/train.parquet` existe (se regenera con el notebook 03) y tiene las columnas de la etapa 3.
- Las decisiones de la etapa 4 (Q1 a Q4) vienen de `especificaciones/04-features.md` y de la sección 7 de `especificacion-proyecto.md`; no se reabren aquí.
- La app recomienda la fecha prometida: el plazo de Olist se conoce solo como referencia y la fecha de compra sí se conoce al predecir.
- El modelo de comparación es la regresión lineal con parámetros fijos; el torneo con otros modelos es la etapa 5.
- Las métricas son el error absoluto medio en días (principal) y el costo medio 3 a 1 (control), definidas en `olistbr.models.metrics`.
- Black Friday cae en el cuarto viernes de noviembre (2017-11-24 en los datos); no hay compras de noviembre-diciembre de 2018 en el periodo.
- La tendencia entre años (2018 más rápido que 2017) no se modela; se reporta como limitación.
- Quedan fuera de alcance: entrenar otros modelos, optimizar hiperparámetros, tracking en Databricks, la app y la fecha recomendada con margen (etapas 5 a 7).
- Los textos y comentarios van en español; los nombres de funciones, columnas y archivos de código, en inglés.
