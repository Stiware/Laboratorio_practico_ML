# Especificación del proyecto

> Plantilla. Complétenla en el grupo y manténganla actualizada. Claude la lee antes de generar
> código: lo que escriban aquí manda sobre cualquier otra suposición.

## 1. Grupo

| Integrante | Rol (datos / modelado / app / negocio) |
|---|---|
| | |
| | |
| | |

Nombre de la "startup": _______________   Repositorio (público): _______________

## 2. Caso de negocio

- **Pregunta de negocio** (una frase, que alguien sin formación técnica entienda): ¿Cuántos días tardará en llegar un pedido y llegará antes o después de la fecha que le prometimos al cliente?
- **¿Quién usará el resultado y qué decisión tomará con él?** Equipos de logística y atención al cliente de un comercio electrónico: planificar la operación, avisar de posibles retrasos y ajustar la fecha prometida.
- **¿Qué cuesta equivocarse?** (cuál error es peor y por qué): Subestimar los días (prometer más rápido de lo real) genera retrasos y clientes molestos; sobrestimarlos alarga plazos y pierde ventas. Supuesto: subestimar cuesta 3 veces más (3 a 1); respaldado por las reseñas (nota 4,3 a tiempo vs 2,6 tarde) y evaluado con 1, 2, 3, 5 y 9 a 1. Ver `notebooks/01_caso_negocio.ipynb`.

## 3. Datos

- **Fuente** (enlace de Kaggle, empresa, etc.): *Brazilian E-Commerce Public Dataset by Olist* (Kaggle, publicado por Olist): https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce. 9 CSV en `data/raw/`.
- **Filas / columnas:** 99.441 pedidos × 8 columnas en `olist_orders_dataset.csv`; 96.478 están entregados. Las demás tablas se unen por `order_id`, `customer_id`, `product_id` y `seller_id`.
- **Variable objetivo** (columna que queremos predecir o describir): días de entrega = `order_delivered_customer_date` − `order_purchase_timestamp` (se construye; solo pedidos `delivered`). Mediana 10,2 días, media 12,6, p99 46,1, máximo 209,6.
- **Tipo de problema:** ☐ clasificación  ☒ regresión  ☐ clusterización
- **Modelo y alcance:** modelo de **regresión lineal**. Objetivo principal: predecir la **cantidad de días** de entrega. Apartado secundario: **probabilidad de retraso** del pedido (que llegue después de la fecha prometida). El retraso **no es la variable a predecir**; se deriva de la predicción en días.
- **¿Una misma entidad (cliente, paciente, empresa) aparece en varias filas?** ☐ no  ☒ sí → la partición
  será por: `customer_unique_id` (99.441 `customer_id` corresponden a 96.096 personas) o por fecha de compra; por decidir en el EDA.
- **Datos personales o confidenciales presentes:** ☐ no  ☒ sí → los identificadores ya vienen anonimizados (hash); se descartan los textos libres de reseñas (`review_comment_*`) y no se usan ids como variables.
- **Licencia o permiso de uso de los datos:** CC BY-NC-SA 4.0 (según la página de Kaggle): se debe citar a Olist, el uso es **no comercial** (válido para este curso) y cualquier obra derivada se comparte con la misma licencia. Los CSV no se suben al repo (`data/raw/` está en `.gitignore`); se descargan desde Kaggle.

## 4. Métrica

- **Métrica principal:** MAE (días), aceptada por el grupo.   **Por qué esa y no otra:** se lee directo en días; no la dominan los pedidos extremos (el 1 % peor aporta el 8,5 % del MAE y el 35 % del RMSE); es el caso 1 a 1 del costo por día; el MAPE se vuelve inestable en entregas cortas y el R² no dice días. Detalle: `notebooks/01_caso_negocio.ipynb`, sección 4.
- **Métricas secundarias:** costo medio por pedido (3 a 1, y sensibilidad con 2, 5 y 9 a 1) y RMSE. **Del apartado de probabilidad de retraso** (secundario): ROC-AUC (qué tan bien separa los pedidos tardíos de los puntuales), Brier score y revisión de calibración (si «30 %» de riesgo realmente se cumple 3 de cada 10 veces); retraso = entrega posterior a `order_estimated_delivery_date`.
- **Línea base** (qué resultado obtiene una regla simple o un modelo trivial): predecir siempre la mediana (10,2 días) da un MAE de ≈ 6,1 días. Usar la fecha prometida por Olist da ≈ 12,7 días (promete en promedio 23,7 días). Con costos 3 a 1 la referencia a vencer es el mejor plazo fijo (15,7 días): costo medio 12,1 por pedido.

## 5. Riesgos

- Fuga de información posible: `order_delivered_carrier_date`, `order_delivered_customer_date`, `order_status` y las reseñas (`review_*`) se conocen después de la compra; no pueden ser variables de entrada.
- Sesgos o equidad: el desempeño puede variar por estado/región del cliente; se revisa por segmento.
- Datos faltantes o de mala calidad: 2.965 pedidos sin fecha de entrega (cancelados, en tránsito, etc.); 775 pedidos sin ítems; 610 productos sin categoría; casos extremos de hasta 209 días.
- Límites de lo que el modelo puede decir: datos de 2016–2018 de una sola plataforma; no captura huelgas, clima ni feriados.

## 6. Entregables y fechas

| Hito | Contenido | Fecha |
|---|---|---|
| H1 | Ficha aprobada, EDA y preprocesamiento reproducible | Cierre de la clase 2 (vie 30 oct) |
| H2 | Torneo, optimización con tracking, modelo registrado, repo modular, app local | Cierre de la clase 4 (jue 5 nov) |
| H3 | App desplegada y presentación (≤ 5 diapositivas) | Clase 5 (vie 6 nov) |
| H4 | Informe técnico final (hasta 8 días tras la última clase) | Domingo 15 nov |

## 7. Decisiones tomadas (agréguenlas a medida que avancen)

| Fecha | Decisión | Motivo |
|---|---|---|
| 30 oct 2026 | El modelo es una **regresión lineal** que predice **días de entrega**. La **probabilidad de retraso** es un apartado secundario y no el objetivo del entrenamiento. | Solo el 6,8 % de los pedidos llega tarde, así que se entrena y evalúa por el error en días; el retraso se reporta como complemento para la app. En el torneo (etapa 5) la regresión lineal es el candidato principal. |
