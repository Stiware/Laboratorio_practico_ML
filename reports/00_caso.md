# Ficha de caso

> Instrucciones y un ejemplo completo (CreditSight): `material-del-curso/evaluacion/ficha-de-caso.md`.
> Estados: `borrador` → (`devuelta` → `borrador`)* → `aprobada`. Solo con la ficha **aprobada** se avanza al EDA.

- **case_id:** ontrack
- **business_question:** Los equipos de logística de un comercio electrónico necesitan saber, al recibir un pedido, cuántos días tardará en llegar al cliente y si llegará después de la fecha que se le prometió. Con esa estimación pueden planificar la operación, avisar a tiempo de posibles retrasos y mejorar la experiencia del cliente.
- **dataset_source:** Brazilian E-Commerce Public Dataset by Olist (Kaggle). Licencia: por confirmar en la página del dataset.
- **dataset_rows:** 99.441 pedidos (`olist_orders_dataset.csv`); 96.478 con estado `delivered`
- **dataset_columns:** 8 en la tabla de pedidos; 9 tablas relacionadas (clientes, ítems, productos, vendedores, pagos, reseñas, geolocalización, traducción de categorías)
- **target:** días entre la compra (`order_purchase_timestamp`) y la entrega al cliente (`order_delivered_customer_date`), columna que se construye
- **problem_type:** regresión
- **primary_metric:** MAE en días (propuesta del grupo; por confirmar), porque se interpreta directo: "el modelo se equivoca en promedio X días". Criterio de éxito: MAE menor que 6,1 días (siempre la mediana) y costo medio por pedido menor que 12,1 con 3 a 1 (mejor plazo fijo de 15,7 días), en datos no vistos; debe ganar también con 2, 5 y 9 a 1
- **business_cost_note:** Subestimar el tiempo (prometer más rápido de lo real) genera retrasos y clientes insatisfechos. Sobrestimarlo hace que la empresa prometa plazos más largos de lo necesario y pierda ventas. Supuesto de trabajo: subestimar cuesta 3 veces más que sobrestimar (3 a 1). Respaldo: la nota de la reseña baja de 4,3 a 2,6 si llega tarde y no cambia si llega antes; la plataforma actúa como si fuera de 5 a 9 a 1. No se puede medir el costo de prometer plazos largos, así que se evalúa también con 1, 2, 5 y 9 a 1. Detalle: `notebooks/01_caso_negocio.ipynb`.
- **status:** borrador
- **reviewer_notes:** <!-- las llena el instructor -->
