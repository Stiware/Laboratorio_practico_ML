# Especificación · Etapa 2 · EDA

> Se llena **antes** de abrir `notebooks/02_eda.ipynb` (las secciones 1 y 2) y se completa **mientras**
> lo ejecutan (3 y 4). El resultado final, con las decisiones, va a `reports/01_eda.md`.
> Guía: `material-del-curso/kit/guia-eda-y-preprocesamiento.md` · Ejemplo: `material-del-curso/ejemplo-creditsight/reports/01_eda.md`.

**Grupo:** _______________   **Fecha:** 30 oct 2026   **Estado:** ☐ borrador  ☒ cerrada _(el grupo validó las 10 decisiones el 31 oct 2026)_

## 1. Qué esperamos encontrar (se escribe ANTES de mirar los datos)

Una hipótesis por pregunta. Si después no se cumple, es un hallazgo: anótenlo.

| Pregunta | Lo que creemos | Qué encontramos |
|---|---|---|
| ¿Qué representa una fila? | En `orders`, un pedido. En `order_items`, un producto dentro de un pedido. | Se cumple. `order_id` no se repite en `orders` (0 duplicados). |
| ¿Alguna entidad se repite en varias filas? | Sí: un cliente puede comprar varias veces y un pedido puede tener varios ítems. | Se cumple. 99.441 `customer_id` son 96.096 personas (`customer_unique_id`); 2.997 compraron más de una vez (máximo 17). 9.803 pedidos tienen más de un ítem. |
| ¿Qué variable **debería** ser estable y no cambiar? | La ubicación de una persona (`customer_unique_id` → estado y código postal). | Casi: 39 personas tienen más de un estado y 250 más de un código postal. |
| ¿Qué significaría un vacío en nuestras columnas clave? | Que el pedido no llegó a esa etapa (cancelado, en tránsito). | Se cumple en parte. 2.965 pedidos sin fecha de entrega, pero 8 están marcados `delivered`: esos son error de registro. |
| ¿Qué columna solo se conoce **después** de la decisión que queremos apoyar? | Fechas de despacho y entrega, estado del pedido y reseñas. | Se cumple. Ver sección 3, fugas. |

## 2. Qué vamos a medir

Marquen lo que aplica a su dataset. Lo que no aplique se tacha **con una razón** (no se deja en blanco).

| Chequeo (guía, sección 3) | ¿Aplica? | Columnas a revisar | Quién lo hace |
|---|---|---|---|
| 3.1 Números guardados como texto | ☒ sí ☐ no: | Todas las fechas de `orders` y `shipping_limit_date` vienen como texto en el CSV y hay que convertirlas | |
| 3.2 Valores centinela (`-1`, `9999`, `?`) | ☒ sí ☐ no: | `payment_type = not_defined`, `payment_installments = 0` | |
| 3.3 Valores imposibles (percentiles, negativos) | ☒ sí ☐ no: | `product_weight_g` = 0, `price` y `freight_value` (máximos), días de entrega (> 100), coordenadas fuera de Brasil | |
| 3.4 Constancia por entidad | ☒ sí ☐ no: | `customer_unique_id` → estado y código postal | |
| 3.5 Vacíos que significan algo | ☒ sí ☐ no: | `order_delivered_*`, `order_approved_at`, campos de producto | |
| 3.6 Duplicados | ☒ sí ☐ no: | `geolocation` (filas exactas), `order_reviews` y `order_payments` por `order_id` | |
| 3.7 Fechas y orden temporal | ☒ sí ☐ no: | Las 5 fechas de `orders` | |
| 3.8 Categorías raras / alta cardinalidad | ☒ sí ☐ no: | `customer_city`, `customer_zip_code_prefix`, `seller_id`, `product_category_name` | |
| 3.9 Identidad y datos personales | ☒ sí ☐ no: | `customer_unique_id`, `review_comment_title`, `review_comment_message` | |

## 3. Lo que medimos (se llena mientras corre el notebook)

- **Filas × columnas:** `orders` 99.441 × 8 · `order_items` 112.650 × 7 · `customers` 99.441 × 5 · `products` 32.951 × 9 · `sellers` 3.095 × 4 · `payments` 103.886 × 5 · `reviews` 99.224 × 7 · `geolocation` 1.000.163 × 5 · traducción de categorías 71 × 2.
- **Unidad de análisis (una frase):** un pedido entregado a un cliente.
- **Columna que identifica a la entidad (`ID_COL`):** `order_id` para el pedido; `customer_unique_id` para agrupar la partición.   **¿Se repite?** ☐ no ☒ sí, hasta 17 pedidos por persona (2.997 personas con más de uno)
- **Objetivo (`TARGET`):** `dias_entrega` = `order_delivered_customer_date` − `order_purchase_timestamp`, en días (se construye).   **Distribución:** 96.470 pedidos; mediana 10,2; media 12,6; desviación 9,5; p99 46,1; máximo 209,6. Cola larga a la derecha.
- **Línea base** (regla trivial) y su resultado: predecir siempre la mediana (10,2 días) da MAE ≈ 6,1 días. La fecha prometida por Olist da MAE ≈ 12,7 (promete 23,7 días en promedio, casi el doble de lo real). Una regresión lineal rápida con distancia y fecha prometida baja a MAE ≈ 5,4.
- **Vacíos en el objetivo:** 2.965 pedidos sin fecha de entrega al cliente (2.957 no están entregados + 8 entregados sin fecha).   **Qué haremos con esas filas:** se excluyen del entrenamiento, porque no tienen objetivo.
- **Datos personales encontrados:** los identificadores (`customer_unique_id`, `order_id`) ya vienen anonimizados con hash. Hay texto libre en `review_comment_title` (87.656 vacíos de 99.224) y `review_comment_message` (58.247 vacíos). Ciudad y código postal son de ubicación aproximada.
- **Columnas sospechosas de fuga:** `order_delivered_carrier_date`, `order_delivered_customer_date`, `order_status`, todas las `review_*` y, por precaución, `order_approved_at` (se miden con y sin ella en la etapa 4). **Sí se pueden usar:** `order_estimated_delivery_date`, `price`, `freight_value`, `shipping_limit_date` y datos de producto, cliente y vendedor, porque se conocen al comprar.

Otros hallazgos de calidad que respaldan las decisiones:
- **Fechas:** 2016 tiene solo 329 pedidos y 2018-09/10 solo 20 (la fecha de corte es 2018-10-17). El volumen estable va de 2017-01 a 2018-08. Hay 1.350 pedidos con despacho anterior a la aprobación y 23 con entrega anterior al despacho.
- **Productos:** 610 sin categoría ni descripción (1,9 %), 2 sin peso, 4 con peso 0, 13 categorías sin traducción. Dos columnas traen un error de ortografía de origen (`product_name_lenght`, `product_description_lenght`).
- **Geolocalización:** 261.831 filas duplicadas exactas, 42 puntos fuera de Brasil, 278 códigos postales de clientes y 7 de vendedores sin coordenadas.
- **Cardinalidad:** 4.119 ciudades y 14.994 códigos postales de clientes; 1.714 de 3.095 vendedores con menos de 10 ítems; el 1 % de vendedores concentra el 26,4 % de los ítems.
- **Qué mueve el objetivo:** distancia vendedor–cliente (correlación 0,39), fecha prometida (0,38), flete (0,17). Mismo estado vendedor–cliente: mediana 6,5 días contra 12,8 si es distinto. SP entrega en 7,2 días y AM en 25,9.
- **Otros:** 775 pedidos sin ítems, 1.278 con más de un vendedor, 6,8 % se entregó después de la fecha prometida.

## 4. Decisiones (mínimo 5)

> Cada una con **cifra**, acción **concreta** y **motivo**. Se copian tal cual a `reports/01_eda.md`.
> Cada decisión se convierte en una función de `src/olistbr/data/` y en una prueba (ver `03-preprocesamiento.md`).
> Las funciones están por crear; los nombres son una propuesta.

| # | Hallazgo (con cifra) | Decisión | Motivo | Función que la implementa |
|---|---|---|---|---|
| 1 | 2.965 de 99.441 pedidos no tienen fecha de entrega (8 de ellos con estado `delivered`) | Quedarse con pedidos `delivered` que tengan fecha de entrega y construir `dias_entrega`; quedan 96.470 | Sin fecha no hay objetivo que aprender | `build_target` |
| 2 | 64 pedidos tardaron más de 100 días (0,07 %); máximo 209,6 | Excluirlos del entrenamiento y reportarlos aparte | Distorsionan el MAE y el RMSE; no son lo que la app debe predecir. Se revisa en la etapa 4 | `drop_extreme_targets` |
| 3 | Un pedido puede tener varios ítems (9.803) y varios vendedores (1.278); 775 pedidos no tienen ítems | Una fila por pedido: sumar precio y flete, contar ítems, sumar peso y volumen, y tomar el vendedor principal. Los 775 pedidos sin ítems no están entre los entregados, así que el filtro de la decisión 1 ya los saca | La unidad de análisis es el pedido, no el ítem; si no, se duplican filas y se contamina la partición | `aggregate_items` |
| 4 | `order_delivered_*`, `order_status` y `review_*` se conocen después de la compra | Eliminarlas de las entradas. `order_approved_at` se mide con y sin ella | Si no, el modelo "ve el futuro" y la app no podría usarlas | `drop_leakage_columns` |
| 5 | 2.997 personas con más de un pedido (máx. 17) | Partir train/test por `customer_unique_id`, no por fila | Un mismo cliente en train y en test infla el resultado | `split_by_group` |
| 6 | La distancia entre vendedor y cliente tiene correlación 0,39 con el objetivo; hay 261.831 duplicados exactos en `geolocation` y 478 pedidos sin coordenadas | Promediar latitud y longitud por código postal, calcular la distancia en km y dejar un indicador cuando falte la coordenada | Es la variable más fuerte y la ubicación sola (ciudad, código postal) tiene demasiadas categorías | `zip_centroids`, `haversine_km` |
| 7 | 610 productos sin categoría (1,9 %), 2 sin peso, 4 con peso 0; 13 categorías sin traducción | Categoría "desconocida" y traducida al inglés; peso 0 se trata como vacío y se imputa con la mediana calculada solo con train | Evitar perder pedidos y evitar que la imputación use información de test | `clean_products` |
| 8 | 4.119 ciudades y 14.994 códigos postales de cliente; 1.714 vendedores con menos de 10 ítems | No usar ciudad ni código postal como variable; usar estado y distancia, y agrupar vendedores raros | Demasiadas categorías para pocos datos: el modelo memorizaría | `group_rare_categories` |
| 9 | 2016 tiene 329 pedidos y 2018-09/10 solo 20; el volumen estable va de 2017-01 a 2018-08 | Entrenar con 2017-01 a 2018-08 y evaluar también por fecha (últimos meses como prueba) | Los meses casi vacíos no representan la operación y la app predice hacia adelante | `filter_stable_period` |
| 10 | `review_comment_*` son texto libre (58–88 mil vacíos) y no se conocen al comprar | Descartar las tablas de reseñas y no usar identificadores como variables | Datos personales potenciales y fuga; los ids no generalizan | `drop_personal_data` |

## 5. Criterios de cierre

- ☒ `notebooks/02_eda.ipynb` corre de principio a fin sin errores (`ID_COL = order_id`, `TARGET = dias_entrega`).
- ☒ Unidad de análisis y `ID_COL` declarados por escrito (sección 3).
- ☒ Al menos **5 decisiones** con cifra, acción y motivo.
- ☒ Datos personales identificados y plan para eliminarlos al cargar.
- ☒ Valores imposibles, centinelas y significado de los vacíos revisados.
- ☒ Fuga sospechada: lista escrita.
- ☒ `reports/01_eda.md` completado.
- ☐ Quien **no** escribió una decisión sabe explicar por qué se tomó. _(lo verifica el grupo)_
