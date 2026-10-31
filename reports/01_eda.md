# Informe del EDA

> Resultado de la exploración de los 9 CSV de Olist en `data/raw/`. Cada hallazgo termina en una decisión,
> y cada decisión se implementa con pruebas en `src/olistbr/data/` (funciones por crear).
> Detalle completo: `especificaciones/02-eda.md`. Estado: **borrador**. Cifras verificadas ejecutando `notebooks/02_eda.ipynb`; falta que el grupo valide las decisiones.

## Qué son los datos

Nueve tablas relacionadas de una plataforma de comercio electrónico brasileña (Olist), de 2016 a 2018. La tabla central es `orders`; se une con clientes, ítems, productos, vendedores, pagos, reseñas y geolocalización.

- **Filas × columnas:** `orders` 99.441 × 8; `order_items` 112.650 × 7; `products` 32.951 × 9; `customers` 99.441 × 5; `sellers` 3.095 × 4; `geolocation` 1.000.163 × 5.
- **Unidad de análisis (qué es una fila):** un pedido entregado a un cliente.
- **¿Una entidad aparece en varias filas?** Sí. 99.441 `customer_id` son 96.096 personas; 2.997 compraron más de una vez (máximo 17). 9.803 pedidos tienen más de un ítem y 1.278 más de un vendedor.
- **Objetivo y distribución:** `dias_entrega` (entrega al cliente − compra), 96.470 pedidos. Mediana 10,2; media 12,6; p99 46,1; máximo 209,6. Cola larga a la derecha.
- **Línea base** (resultado de una regla trivial): predecir siempre la mediana da MAE ≈ 6,1 días. La fecha prometida por Olist da MAE ≈ 12,7 (promete 23,7 días en promedio).

## Calidad de los datos

- **Vacíos con significado:** 2.965 pedidos sin fecha de entrega (2.957 no entregados y 8 marcados `delivered`). 160 sin fecha de aprobación. 610 productos sin categoría ni descripción (1,9 %).
- **Fechas:** vienen como texto y hay que convertirlas. 2016 tiene solo 329 pedidos y 2018-09/10 solo 20. 1.350 pedidos con despacho anterior a la aprobación y 23 con entrega anterior al despacho.
- **Valores extremos o imposibles:** 64 pedidos con más de 100 días de entrega; 4 productos con peso 0; 42 coordenadas fuera de Brasil; `payment_type = not_defined` (3).
- **Duplicados:** 261.831 filas exactas en `geolocation`; 551 reseñas y 4.446 pagos repiten `order_id` (esperable: varias reseñas/pagos por pedido).
- **Variables que deberían ser estables:** 39 personas con más de un estado y 250 con más de un código postal.
- **Cardinalidad:** 4.119 ciudades y 14.994 códigos postales de cliente; 1.714 de 3.095 vendedores con menos de 10 ítems.
- **Datos personales:** ids anonimizados con hash; texto libre en `review_comment_title` y `review_comment_message`.

## Sospechas de fuga de información

- `order_delivered_carrier_date`, `order_delivered_customer_date`, `order_status` y todas las `review_*`: se conocen después de la compra.
- `order_approved_at`: se conoce horas después de comprar; se mide con y sin ella.
- **No son fuga:** `order_estimated_delivery_date` (se promete al comprar; correlación 0,38 con el objetivo), `price`, `freight_value`, `shipping_limit_date` y los datos de producto, cliente y vendedor.

## Decisiones

> Mínimo **5**. Cada una con una cifra que la respalda, una acción concreta y un motivo.

| # | Hallazgo (con cifra) | Decisión | Motivo |
|---|---|---|---|
| 1 | 2.965 de 99.441 pedidos sin fecha de entrega | Quedarse con `delivered` con fecha y construir `dias_entrega` (96.470) | Sin fecha no hay objetivo |
| 2 | 64 pedidos con más de 100 días (0,07 %) | Excluirlos del entrenamiento y reportarlos aparte | Distorsionan MAE y RMSE |
| 3 | 9.803 pedidos con varios ítems; 1.278 con varios vendedores | Una fila por pedido: agregar ítems | La unidad es el pedido, no el ítem |
| 4 | `order_delivered_*`, `order_status`, `review_*` son posteriores a la compra | Eliminarlas de las entradas; medir `order_approved_at` con y sin | Evitar fuga |
| 5 | 2.997 personas con varios pedidos | Partición por `customer_unique_id` | Evitar cliente repetido entre train y test |
| 6 | Distancia vendedor–cliente: correlación 0,39; 261.831 duplicados en geolocalización; 478 pedidos sin coordenadas | Centroide por código postal, distancia en km e indicador de faltante | Variable más fuerte y sin la cardinalidad de la ciudad |
| 7 | 610 productos sin categoría; 2 sin peso; 4 con peso 0 | Categoría "desconocida"; peso 0 como vacío; imputar con la mediana de train | No perder pedidos ni filtrar información de test |
| 8 | 14.994 códigos postales; 1.714 vendedores con menos de 10 ítems | Usar estado y distancia; agrupar vendedores raros | Evitar memorización |
| 9 | 2016: 329 pedidos; 2018-09/10: 20 | Entrenar con 2017-01 a 2018-08 y evaluar por fecha | La app predice hacia adelante |
| 10 | Texto libre en reseñas | Descartar `review_comment_*` e ids como variables | Datos personales y fuga |

## Qué queda abierto

- Validar en grupo las 10 decisiones (sobre todo la 2, excluir los pedidos de más de 100 días).
- ¿`order_approved_at` aporta señal real o es fuga? Se mide con y sin ella en la etapa 4.
- ¿Excluir los pedidos de más de 100 días o limitarlos? Depende de qué tan frecuentes sean en el periodo de prueba.
- ¿Una partición por fecha da un resultado distinto a la partición por persona? Se compara en la etapa 4.
- **Probabilidad de retraso (apartado secundario):** el modelo es una regresión lineal que predice días; el retraso (6,8 % de los pedidos) no es el objetivo. Falta decidir cómo calcular la probabilidad: (a) a partir de los errores del modelo, suponiendo que se reparten de forma simétrica (los errores aquí son asimétricos, hay que comprobar la calibración), o (b) con un modelo aparte de retraso sí/no. Se decide en la etapa de features/torneo.
