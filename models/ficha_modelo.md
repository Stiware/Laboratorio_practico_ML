# Ficha del modelo

> Cifras de la prueba de `reports/07_metricas_test.json` (19.215 pedidos de clientes nunca vistos) y del registro de `reports/08_registro.json`.
> Detalle y límites en `reports/07_validacion.md` y `reports/08_registro.md`.

## Qué es

OnTrack recomienda **la fecha de entrega que conviene prometerle al cliente** de un pedido de Olist, a partir de lo que se sabe al momento de la compra.
Apoya a logística y atención al cliente: una promesa más realista que el plazo fijo, que no se queda corta demasiado seguido ni sobra demasiados días.

- **Modelo:** LightGBM cuantil 0,75 (predice el percentil 75 de los días de entrega), con los parámetros optimizados en la etapa 6 (418 árboles, tasa 0,049,
  70 hojas, mínimo 73 pedidos por hoja, `subsample` 0,95, `colsample_bytree` 0,65, `reg_lambda` 4,43), dentro de un `Pipeline` con su preprocesamiento.
- **Registro:** `workspace.default.ontrack_plazo_entrega` versión 1, alias `champion` (Databricks, Unity Catalog). La app usa `models/model.joblib`, exportado desde el champion.
- **Regla de decisión:** la fecha recomendada es la compra más el plazo del modelo, redondeada hacia arriba al día. Se usa el **percentil 75** porque se supuso que
  atrasarse cuesta **3 veces** más que prometer de más: ese es el plazo que minimiza el costo.

## Datos de entrenamiento

- **Fuente y licencia:** *Brazilian E-Commerce Public Dataset by Olist* (https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce); licencia CC BY-NC-SA 4.0 (uso no comercial)
- **Tamaño:** 76.924 pedidos entregados de 74.425 clientes, comprados de enero de 2017 a agosto de 2018 (sin los 64 pedidos de más de 100 días).
- **Partición:** por **cliente** (`customer_unique_id`): ningún cliente está en el entrenamiento y en la prueba; la prueba se abrió **una sola vez para medir**
  (2026-11-05 20:44:44) y se volvió a leer una vez, sin decidir nada, para comprobar el registro.
- **Datos personales:** las tablas de reseñas y de pagos no se leen; la ciudad y el código postal solo se usan para calcular la distancia y no quedan en la tabla
  final; el cliente y el pedido son códigos anónimos de Olist que solo sirven para partir los datos y **no entran al modelo** ni a la app. El vendedor tampoco se pide en la app (se usa «otros»).

## Rendimiento (prueba, entidades nunca vistas)

Costo 3 a 1 por pedido (1 unidad = un día prometido de más; un día de atraso cuesta 3). Intervalos del 95 % sorteando clientes completos (1.000 sorteos).

| Métrica | Valor | Intervalo del 95 % |
|---|---|---|
| Costo 3 a 1 del modelo (métrica principal) | **9,216** | [9,005 ; 9,422] |
| Baseline: plazo fijo del percentil 75 | 11,709 | [11,468 ; 11,946] |
| Diferencia (modelo − plazo fijo) | **−2,493** (21 % menos) | [−2,608 ; −2,373] → mejora con confianza |
| % de pedidos que llegan después del plazo del modelo (control) | 26,4 % (esperado ≈ 25 %) | [25,9 % ; 27,1 %] |
| Días que sobran cuando llega a tiempo / días que faltan cuando llega tarde | 5,3 / 6,7 (plazo fijo: 7,2 / 8,6) | — |
| Referencia: plazo prometido por Olist | costo 13,912; 8,1 % tarde | — |

## Variables que más pesan

Según SHAP (2.000 pedidos de la prueba), en orden: si vendedor y cliente están en el mismo estado (`same_state`), la distancia (`distance_km`), la combinación de
distancia y mismo estado, el estado del cliente SP, el logaritmo de la distancia, el flete, los estados MG y RJ, la región Nordeste y si la compra fue de jueves a
sábado. Es decir, la **geografía** pesa más que todo lo demás. SHAP explica al modelo, no al mundo, y no indica causas.

## Límites y usos no recomendados

- **Datos con los que se entrenó:** pedidos **entregados** de Olist en Brasil, de enero de 2017 a agosto de 2018. No se validó para otras fechas, otros países ni otras empresas.
- **Supuestos de negocio:** que atrasarse cuesta 3 veces más que prometer de más (con 9 a 1, el plazo prometido por Olist sería mejor que el modelo); que se conocen los
  datos del pedido al comprar (distancia, peso, volumen, flete); el modelo no conoce al vendedor concreto.
- **Atributos sensibles:** la **ubicación del cliente** (estado y región) cambia el plazo. El % de pedidos tarde por región en la prueba va de 23,1 % (Nordeste) a
  29,7 % (Norte, solo 306 pedidos). Revisar con el negocio que eso refleja la logística y no un trato injusto.
- **Qué no debe hacerse:** usar la fecha como una **promesa garantizada** (≈ 1 de cada 4 pedidos llega después; ese 26,4 % se midió con el plazo del modelo desde la
  hora de compra, y la app cuenta desde el día de compra y redondea al día, así que en la app es aproximado); usarla en temporadas de alta demanda sin vigilar la
  cobertura (de noviembre de 2017 a marzo de 2018 llegó tarde entre el 33 % y el 46 %); leer su error como el «error del día más probable» (es el del percentil 75).
- **Lo que no se evaluó:** el desempeño en meses posteriores a agosto de 2018 (la prueba es de clientes nuevos del mismo periodo); la marca de fin de año y el historial
  por vendedor; una probabilidad de retraso por pedido; otros percentiles para otras proporciones de costo. **Recomendación:** vigilar cada mes la cobertura y
  recalibrar si sale de una banda (por ejemplo, 70 % a 80 %).

## Reproducibilidad

Código en `src/`, pruebas en `tests/`, notebooks en `notebooks/` (generados con `scripts/generadores/`). La búsqueda de la etapa 6 se repitió y dio exactamente lo mismo.
`models/model.joblib` carga y predice igual en Python 3.12: 1.000 plazos idénticos y las pruebas de la inferencia y la app en verde (salida guardada en
`reports/09_verificacion_python312.txt`). El modelo se guarda con `joblib` (pickle): solo cargar
archivos del propio grupo.
