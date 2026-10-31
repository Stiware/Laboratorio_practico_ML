# Ficha de caso

> Instrucciones y un ejemplo completo (CreditSight): `material-del-curso/evaluacion/ficha-de-caso.md`.
> Estados: `borrador` → (`devuelta` → `borrador`)* → `aprobada`. Solo con la ficha **aprobada** se avanza al EDA.

- **case_id:** ontrack
- **business_question:** Los equipos de logística de un comercio electrónico necesitan saber, al recibir un pedido, cuántos días tardará en llegar al cliente y si llegará después de la fecha que se le prometió. Con esa estimación pueden planificar la operación, avisar a tiempo de posibles retrasos y mejorar la experiencia del cliente.
- **dataset_source:** *Brazilian E-Commerce Public Dataset by Olist* (Kaggle, publicado por Olist): https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce. Licencia: CC BY-NC-SA 4.0 (atribución, uso no comercial, compartir igual).
- **dataset_rows:** 99.441 pedidos (`olist_orders_dataset.csv`); 96.478 con estado `delivered`
- **dataset_columns:** 8 en la tabla de pedidos; 9 tablas relacionadas (clientes, ítems, productos, vendedores, pagos, reseñas, geolocalización, traducción de categorías)
- **target:** días entre la compra (`order_purchase_timestamp`) y la entrega al cliente (`order_delivered_customer_date`), columna que se construye
- **problem_type:** regresión (en la planeación inicial, un modelo de regresión lineal que predice los días de entrega; **desde el 31 oct 2026 el modelo final es el mejor del torneo**, ver `reports/05_torneo.md`). Apartado secundario: probabilidad de retraso frente a la fecha prometida; los retrasos no son el objetivo del entrenamiento.
- **primary_metric:** MAE en días (aceptada por el grupo). Por qué: se lee en el idioma del negocio ("el modelo se equivoca en promedio X días"); no la dominan unos pocos pedidos extremos (el 1 % peor aporta el 8,5 % del MAE pero el 35 % del RMSE); es el caso 1 a 1 del mismo costo por día del 3 a 1; y las alternativas fallan (el MAPE se dispara en entregas cortas y el R² no dice días). Su límite: no distingue subestimar de sobrestimar, por eso se acompaña del costo medio por pedido. Criterio de éxito: MAE menor que 6,1 días (siempre la mediana) y costo medio por pedido menor que 12,1 con 3 a 1 (mejor plazo fijo de 15,7 días), en datos no vistos; debe ganar también con 2, 5 y 9 a 1
- **business_cost_note:** Subestimar el tiempo (prometer más rápido de lo real) genera retrasos y clientes insatisfechos. Sobrestimarlo hace que la empresa prometa plazos más largos de lo necesario y pierda ventas. Supuesto de trabajo: subestimar cuesta 3 veces más que sobrestimar (3 a 1). Respaldo: la nota de la reseña baja de 4,3 a 2,6 si llega tarde y no cambia si llega antes; la plataforma actúa como si fuera de 5 a 9 a 1. No se puede medir el costo de prometer plazos largos, así que se evalúa también con 1, 2, 5 y 9 a 1. Detalle: `notebooks/01_caso_negocio.ipynb`.
- **status:** borrador
- **reviewer_notes:** <!-- las llena el instructor -->
