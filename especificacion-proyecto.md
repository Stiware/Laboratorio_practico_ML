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
- **Modelo y alcance:** _(actualizado el 31 oct 2026: el modelo final es **el mejor del torneo de modelos**, no necesariamente una regresión lineal; la lineal queda como referencia interpretable; ver sección 7)_ modelo de **regresión lineal** en la planeación inicial. Objetivo principal: predecir la **cantidad de días** de entrega. Apartado secundario: **probabilidad de retraso** del pedido (que llegue después de la fecha prometida). El retraso **no es la variable a predecir**; se deriva de la predicción en días.
- **¿Una misma entidad (cliente, paciente, empresa) aparece en varias filas?** ☐ no  ☒ sí → la partición
  será por: `customer_unique_id` (99.441 `customer_id` corresponden a 96.096 personas) o por fecha de compra; por decidir en el EDA.
- **Datos personales o confidenciales presentes:** ☐ no  ☒ sí → los identificadores ya vienen anonimizados (hash); se descartan los textos libres de reseñas (`review_comment_*`) y no se usan ids como variables.
- **Licencia o permiso de uso de los datos:** CC BY-NC-SA 4.0 (según la página de Kaggle): se debe citar a Olist, el uso es **no comercial** (válido para este curso) y cualquier obra derivada se comparte con la misma licencia. Los CSV no se suben al repo (`data/raw/` está en `.gitignore`); se descargan desde Kaggle.

## 4. Métrica

- **Métrica principal:** MAE (días), aceptada por el grupo. _(Desde el 31 oct 2026 el torneo de modelos lo decide el costo 3 a 1; ver sección 7.)_   **Por qué esa y no otra:** se lee directo en días; no la dominan los pedidos extremos (el 1 % peor aporta el 8,5 % del MAE y el 35 % del RMSE); es el caso 1 a 1 del costo por día; el MAPE se vuelve inestable en entregas cortas y el R² no dice días. Detalle: `notebooks/01_caso_negocio.ipynb`, sección 4.
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
| 31 oct 2026 | **La app recomienda la fecha prometida.** El modelo predice los días de entrega y **no usa el plazo que Olist ya prometió como entrada** (sería circular); ese plazo se mide una sola vez como referencia. El retraso (apartado secundario) se define respecto a la fecha recomendada. Con el costo 3 a 1, la fecha recomendada es la predicción más un margen (percentil 75 de los errores de clientes de calibración apartados dentro del entrenamiento; nunca de la validación) o, en los modelos que predicen el percentil 75 directamente, la propia predicción. | Usar la fecha de Olist para predecir sería circular si la app la fija; solo el 6,8 % de los pedidos llega tarde (clase poco frecuente), por eso no se entrena para retrasos. |
| 31 oct 2026 | **Variables (etapa 4):** se miden las ideas de `especificaciones/04-features.md` con validación cruzada por cliente y regla `mejora / ruido > 1`, más una prueba hacia adelante (entrenar hasta feb-2018, validar mar a ago-2018). El mes de compra suelto se descarta y se prueba una marca de temporada alta de fin de año. Modelo de comparación: `LinearRegression`. Variables históricas por vendedor, categoría y ruta solo con `TargetEncoder` dentro del `Pipeline`. | El patrón mensual no se repite entre 2017 y 2018 (correlación 0,14); el pico de fin de año solo se observa una vez; el día de la semana es estable (0,72). |
| 31 oct 2026 | **Flujo de ramas:** todo el trabajo va en `main`; no se crean más ramas. | Decisión del grupo para mantener un solo historial. |
| 31 oct 2026 | **Torneo de modelos (etapa 5):** el **costo 3 a 1 decide al ganador**; el MAE sigue siendo la métrica principal de exactitud y se reporta siempre, junto con RMSE, R², porcentaje de pedidos que llegan tarde y cobertura. | La app recomienda un plazo con costo asimétrico; un modelo con buen MAE no minimiza ese costo. El costo 3 a 1 equivale a 4 veces la pérdida cuantílica con τ = 0,75. |
| 31 oct 2026 | **Dos familias de modelos:** (A) valor típico + margen, donde el margen es el percentil 75 de los errores en clientes de calibración apartados dentro de cada entrenamiento; (B) modelos que predicen directamente el percentil 75 (boosting). La regresión lineal cuantílica de scikit-learn se descarta (no escala). | El plazo que minimiza el costo 3 a 1 es el percentil 75, no la mediana. |
| 31 oct 2026 | **Variables del torneo:** 7 grupos, **sin la marca de fin de año (F6)**, que se mide solo en los finalistas como sensibilidad. | F6 descansa en una sola temporada (2017) y no se puede validar hacia adelante. |
| 31 oct 2026 | **La regresión lineal siempre es candidata** y se reporta su diferencia contra el ganador (días de error por pedido y por cada 1.000 pedidos); no se inventa un valor en dinero. | Es el modelo del caso de negocio; si otro gana por más que su ruido, el grupo decide con esa diferencia a la vista. |
| 31 oct 2026 | **Validación del torneo:** los 5 pliegues por cliente de la etapa 4 (semilla 42) **más** la prueba hacia adelante, para todos los modelos. | En la etapa 4 una idea ganó por pliegues y perdió hacia adelante. |
| 31 oct 2026 | **Bosque aleatorio liviano** (100 árboles, profundidad 12, hoja mínima 20, 33 % de variables, 50 % de filas por árbol; 18,4 s por ajuste frente a 254 s). Decisión delegada a Claude. | Viable para el torneo y para optimizarlo en la etapa 6. |
| 31 oct 2026 | **El modelo final no tiene que ser una regresión lineal: se busca el mejor modelo.** En el torneo la regresión lineal quedó 7.ª (+0,298 de costo por pedido, 3,2 % más que el ganador, LightGBM cuantil 0,75). Esto reemplaza la decisión del 30 oct («el modelo es una regresión lineal»); la lineal se conserva como referencia interpretable. | El objetivo del proyecto es el mejor plazo recomendado; el modelo se entrega en `models/model.joblib`, así que no hay restricción de tipo de modelo. |
