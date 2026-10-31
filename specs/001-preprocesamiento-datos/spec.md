# Feature Specification: Preprocesamiento de datos de OnTrack (etapa 3)

**Feature Branch**: `main` (sin rama propia; el grupo trabaja sobre `main`)

**Created**: 2026-10-31

**Status**: Draft

**Input**: User description: "Etapa 3 · Preprocesamiento de OnTrack: limpieza y partición de los datos de Olist según `especificaciones/03-preprocesamiento.md` (cerrada). Construir `dias_entrega`, filtrar periodo 2017-01 a 2018-08, excluir >100 días, agregar ítems por pedido, limpiar productos, distancia vendedor-cliente, quitar fuga y datos personales, partir 80/20 por `customer_unique_id` con semilla 42, agrupar vendedores raros solo con train, y guardar `data/processed/train` y `test`. Código en `src/olistbr/data/` con pruebas en `tests/` y notebook 03 que solo orquesta."

> Fuente de las reglas: `especificaciones/03-preprocesamiento.md` (aprobada por el grupo el 31 oct 2026) y las 10 decisiones de `especificaciones/02-eda.md`. Si esta especificación y aquellas discrepan, mandan aquellas.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Obtener una tabla limpia de pedidos con su objetivo (Priority: P1)

Como analista del grupo, quiero partir de los CSV originales de Olist y obtener una sola tabla con una fila por pedido entregado, con los días de entrega calculados y sin datos personales ni columnas que "ven el futuro", para poder entrenar modelos sin trampa.

**Why this priority**: sin esta tabla no hay etapa 4 ni torneo de modelos. Es el mínimo viable de la etapa.

**Independent Test**: ejecutar la limpieza sobre los CSV originales y comprobar que (a) hay una fila por `order_id`, (b) todas son pedidos entregados con `dias_entrega` entre 0 y 100, (c) todas las compras caen entre 2017-01-01 y 2018-08-31, (d) no aparecen columnas de reseñas ni de fechas posteriores a la compra.

**Acceptance Scenarios**:

1. **Given** los CSV originales en `data/raw/`, **When** se construye el objetivo, **Then** quedan solo pedidos con estado `delivered` y fecha de entrega, y `dias_entrega` es la diferencia en días entre entrega y compra.
2. **Given** pedidos con `dias_entrega` mayor a 100, **When** se aplica la regla de extremos, **Then** salen del conjunto de entrenamiento y se guardan aparte para reportarlos (64 en el EDA).
3. **Given** un pedido con varios ítems o varios vendedores, **When** se agregan los ítems, **Then** queda una sola fila con precio y flete sumados, conteo de ítems, peso y volumen sumados y el vendedor con más ítems (empate: el menor `seller_id`).
4. **Given** las tablas de reseñas y sus textos libres, **When** se cargan los datos, **Then** no llegan al resultado ni se imprimen.
5. **Given** la tabla limpia, **When** se revisan las columnas, **Then** no contiene `order_delivered_*`, `order_status` ni `review_*`; `order_approved_at` queda aparte y fuera de las entradas por defecto.

---

### User Story 2 - Partir entrenamiento y prueba sin fuga (Priority: P1)

Como analista, quiero un conjunto de entrenamiento (80 %) y uno de prueba cerrado (20 %), partidos por cliente, para que un mismo cliente nunca esté en los dos y la métrica final sea honesta.

**Why this priority**: es la regla de fuga más importante del proyecto (constitución, principio II) y de ella depende toda la validación posterior.

**Independent Test**: tras partir, comprobar que ningún `customer_unique_id` está en ambos conjuntos, que la suma de filas coincide con la tabla limpia y que repetir la ejecución con la semilla 42 da exactamente la misma partición.

**Acceptance Scenarios**:

1. **Given** clientes con varios pedidos (2.997 en el EDA, hasta 17), **When** se parte, **Then** todos los pedidos de un cliente van al mismo lado.
2. **Given** la misma tabla y la semilla 42, **When** se parte dos veces, **Then** el resultado es idéntico.
3. **Given** una partición hecha por fila (prueba de contraste), **When** se corre la verificación de no solapamiento, **Then** falla, lo que demuestra que la prueba sirve.
4. **Given** el conjunto de prueba guardado, **When** termina esta etapa, **Then** nadie lo mira ni lo usa para decidir nada hasta la etapa 7.

---

### User Story 3 - Variables derivadas de producto, distancia y categorías raras (Priority: P2)

Como analista, quiero la distancia entre vendedor y cliente en kilómetros, los productos limpios y los vendedores poco frecuentes agrupados, para entregar a la etapa 4 las variables más informativas ya listas.

**Why this priority**: mejora el poder predictivo (la distancia tiene correlación 0,39 con el objetivo) pero el pipeline de partición funciona sin ella.

**Independent Test**: con casos hechos a mano (dos códigos postales conocidos, un producto sin categoría, un peso 0, un vendedor con pocos ítems) verificar el valor esperado de cada regla.

**Acceptance Scenarios**:

1. **Given** varias filas de geolocalización para un mismo código postal, **When** se calculan centroides, **Then** hay un único punto (latitud y longitud promedio) por código postal.
2. **Given** vendedor y cliente con coordenadas, **When** se calcula la distancia, **Then** se obtiene en km; **Given** coordenadas ausentes, **Then** la distancia queda vacía y `missing_coords` indica que faltaba.
3. **Given** un producto sin categoría, **When** se limpia, **Then** su categoría es `"desconocida"`; las demás se traducen al inglés; un peso 0 queda como vacío.
4. **Given** vendedores con menos de 10 ítems **en el entrenamiento**, **When** se agrupan, **Then** pasan a `"otros"`, y la misma lista se aplica a la prueba sin recalcularla.
5. **Given** el periodo 2017-01 a 2018-08, **When** se filtra, **Then** se descartan los pedidos de 2016 y de 2018-09/10.

---

### User Story 4 - Dejar constancia de cada decisión (Priority: P3)

Como grupo, queremos que cada regla aplicada quede en `reports/` con su cifra y su motivo, y que el notebook 03 solo llame a las funciones, para poder explicarlas en la presentación y en el informe.

**Why this priority**: es obligatorio para el curso (principio IV) pero no bloquea el trabajo técnico.

**Independent Test**: borrar `data/processed/`, ejecutar el notebook 03 de principio a fin y comprobar que se regeneran `train` y `test` y que existe una sección de reporte con las filas antes y después de cada regla.

**Acceptance Scenarios**:

1. **Given** `data/processed/` vacía, **When** se ejecuta el notebook 03, **Then** se generan los archivos de entrenamiento y prueba sin pasos manuales.
2. **Given** cada regla de limpieza, **When** se revisa el reporte, **Then** muestra filas antes y después, la cifra y el motivo.

---

### Edge Cases

- Pedidos con estado `delivered` pero sin fecha de entrega (8 en el EDA): se excluyen.
- Pedidos sin ítems (775): no están entre los entregados, así que el filtro del objetivo ya los saca; si alguno llegara, no debe generar una fila con valores inventados.
- Pedidos con más de un vendedor (1.278): se usa el vendedor con más ítems; si hay empate, el de menor `seller_id`.
- Códigos postales sin coordenadas (278 de clientes, 7 de vendedores; 478 pedidos afectados): distancia vacía e indicador `missing_coords`; la imputación se hace en la etapa 4 con datos de entrenamiento.
- Vendedores que aparecen solo en prueba: se tratan como `"otros"` (no estaban en la lista aprendida del entrenamiento).
- Cliente con más de un estado o código postal (39 y 250 en el EDA): la partición usa solo `customer_unique_id`; no se intenta reparar esa ubicación en esta etapa.
- Peso 0 o vacío en productos (4 y 2): queda vacío; no se imputa aquí.
- Fechas guardadas como texto: se convierten antes de calcular días.
- Ejecutar dos veces seguidas no debe duplicar ni alterar los archivos de salida.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: El sistema MUST leer los CSV de `data/raw/` sin modificarlos y descartar los datos personales (textos libres de reseñas) y la tabla de reseñas al cargar, antes de cualquier impresión.
- **FR-002**: El sistema MUST conservar solo pedidos con estado `delivered` y fecha de entrega no vacía, y calcular `dias_entrega` en días (decisión 1; 96.470 pedidos en el EDA).
- **FR-003**: El sistema MUST conservar solo compras entre 2017-01-01 y 2018-08-31 (decisión 9).
- **FR-004**: El sistema MUST excluir los pedidos con `dias_entrega` mayor a 100 del entrenamiento y guardarlos aparte para reportarlos (decisión 2).
- **FR-005**: El sistema MUST producir una fila por `order_id` agregando ítems según la decisión 3 (suma de precio, flete, peso y volumen; conteo de ítems; vendedor principal con desempate por menor `seller_id`).
- **FR-006**: El sistema MUST limpiar productos según la decisión 7: categoría vacía a `"desconocida"`, traducción al inglés, peso 0 tratado como vacío.
- **FR-007**: El sistema MUST calcular la distancia vendedor–cliente en km a partir del promedio de coordenadas por código postal; si un punto cae fuera de Brasil, MUST reemplazarse por el promedio de las filas buenas del mismo código postal o, si no hay, por la mediana de la misma ciudad y estado; con la columna indicadora `missing_coords` cuando falte alguna coordenada (decisión 6).
- **FR-008**: El sistema MUST quitar de las entradas las columnas que se conocen después de la compra (`order_delivered_*`, `order_status`, `review_*`) y mantener `order_approved_at` aparte, fuera de las entradas por defecto (decisión 4).
- **FR-009**: El sistema MUST no usar ciudad, código postal ni ningún identificador como variable de entrada; `customer_unique_id` se conserva solo para agrupar la partición (decisiones 8 y 10).
- **FR-010**: El sistema MUST partir los datos en 80 % entrenamiento y 20 % prueba agrupando por `customer_unique_id`, con semilla 42, de forma reproducible (decisión 5).
- **FR-011**: El sistema MUST calcular la lista de vendedores con menos de 10 ítems **solo con el entrenamiento**, aplicarla a ambos conjuntos y asignar `"otros"` a los vendedores raros o no vistos (decisión 8).
- **FR-012**: El sistema MUST guardar `train` y `test` en `data/processed/`; el conjunto de prueba queda cerrado hasta la etapa 7.
- **FR-013**: El sistema MUST dejar la imputación de vacíos (peso, distancia) para la etapa 4, dentro de un `Pipeline` ajustado solo con entrenamiento.
- **FR-014**: Cada regla de limpieza y transformación MUST estar en una función de `src/olistbr/data/` con al menos una prueba en `tests/` con un caso normal y uno sucio hechos a mano.
- **FR-015**: Debe existir una prueba que verifique que ningún cliente aparece en ambas particiones y que no se pierde ninguna fila, y que falla si se parte por fila.
- **FR-016**: Debe existir una prueba que verifique que el resultado de cargar no contiene columnas de datos personales.
- **FR-017**: El notebook `notebooks/03_preprocesamiento.ipynb` MUST solo llamar a las funciones de `src/`, sin lógica reutilizable propia, y regenerar `data/processed/` desde cero.
- **FR-018**: Cada decisión MUST quedar documentada en `reports/` con filas antes y después, la cifra y el motivo, y reflejada en `reports/bitacora_decisiones.md`.

### Key Entities *(include if feature involves data)*

- **Pedido (order)**: unidad de análisis; una fila por `order_id` entregado, con `dias_entrega` como objetivo.
- **Cliente (customer)**: persona identificada por `customer_unique_id`; puede tener varios pedidos y es la entidad por la que se parte.
- **Ítem de pedido**: producto dentro de un pedido; se agrega para dejar una fila por pedido.
- **Producto**: categoría, peso y dimensiones; se limpia antes de agregarse al pedido.
- **Vendedor (seller)**: quien despacha; se agrupa como `"otros"` si es raro en el entrenamiento.
- **Código postal / centroide**: punto promedio (latitud, longitud) por código postal, base de la distancia.
- **Conjunto de entrenamiento y de prueba**: salidas de la etapa; la prueba permanece cerrada.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: La limpieza corre de principio a fin desde cero (con `data/processed/` borrada) sin pasos manuales y regenera `train` y `test`.
- **SC-002**: Cero clientes aparecen en entrenamiento y prueba a la vez, y la suma de filas de ambos es igual a la de la tabla limpia.
- **SC-003**: La proporción de prueba es 20 % de los clientes, con una tolerancia de 1 punto porcentual en filas, y repetir la ejecución con la semilla 42 produce archivos idénticos.
- **SC-004**: La tabla de salida tiene exactamente una fila por pedido, todas con `dias_entrega` entre 0 y 100 y compras entre 2017-01-01 y 2018-08-31.
- **SC-005**: Ninguna columna de reseñas, fechas posteriores a la compra, estado del pedido ni identificadores aparece como entrada del modelo.
- **SC-006**: `pytest tests` termina en verde y cada una de las 10 reglas de la sección 3 de la especificación tiene al menos una prueba.
- **SC-007**: Cada integrante del grupo puede explicar en palabras simples al menos una de las reglas y su motivo.
- **SC-008**: Los datos originales de `data/raw/` quedan intactos (sin cambios detectables en Git ni en sus tamaños).

## Assumptions

- Los CSV de Olist ya están en `data/raw/` (descargados de Kaggle; no se suben a GitHub por licencia).
- Los umbrales (100 días, periodo 2017-01 a 2018-08, 10 ítems, 20 % de prueba, semilla 42) vienen de la especificación aprobada y no se reabren en esta etapa.
- El número de pedidos esperado (96.470 entregados antes de los filtros de periodo y extremos) es una referencia del EDA; las cifras finales las producirá el código y se reportarán tal como salgan.
- La revisión del desempeño por fecha (últimos meses) se hace en la etapa 7 sobre el mismo conjunto de prueba; esta etapa no crea una segunda partición temporal.
- La imputación de medianas, la codificación de categorías y la decisión de usar `order_approved_at` pertenecen a la etapa 4.
- Quedan fuera de alcance: ingeniería de variables adicionales, entrenamiento de modelos, tracking en Databricks y la app.
- Los textos y comentarios se escriben en español; los nombres de funciones y archivos de código, en inglés.
- El orden de pasos aprobado es: cargar, quitar datos personales, objetivo, periodo, extremos, ítems, productos, distancia, fuga, partir y agrupar vendedores raros solo con train.
