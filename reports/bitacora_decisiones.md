# Bitácora de cambios y decisiones · OnTrack

> Registro corrido para armar la presentación (máx. 5 diapositivas) y el informe técnico.
> Se agrega una línea por hito o decisión, con su motivo. Lo más reciente va abajo.
> Los detalles están en `reports/` y `especificaciones/`; aquí solo el resumen.

## Resumen en una frase

OnTrack predice cuántos días tardará en llegar un pedido de Olist (regresión lineal) para que logística y atención al cliente prometan fechas más realistas y avisen de posibles retrasos.

## Caso y métrica

| Fecha | Decisión | Motivo | Dónde |
|---|---|---|---|
| 30 oct 2026 | Objetivo: `dias_entrega` (compra → entrega al cliente). El retraso es un apartado secundario derivado, no el objetivo. | Solo 6,8 % llega tarde; se entrena por error en días. | `especificacion-proyecto.md` §3 |
| 30 oct 2026 | Métrica principal: MAE (días). Secundarias: costo medio y RMSE. | Se lee directo en días y no la dominan los extremos (el 1 % peor aporta 8,5 % del MAE y 35 % del RMSE). | `especificacion-proyecto.md` §4 |
| 30 oct 2026 | Costo de error 3 a 1 (subestimar cuesta 3 veces más), con sensibilidad 1/2/3/5/9. | Supuesto; respaldado por reseñas (nota 4,3 a tiempo vs 2,6 tarde). | `reports/00_caso.md` |
| 30 oct 2026 | Referencia a vencer: mejor plazo fijo (15,7 días, costo 12,1). Línea base: mediana 10,2 días, MAE ≈ 6,1. | Regla trivial contra la que se compara el modelo. | `reports/01_eda.md` |
| 30 oct 2026 | Licencia CC BY-NC-SA 4.0: uso no comercial, citar a Olist; los CSV no se suben al repo. | Cumplir la licencia del dataset. | `especificacion-proyecto.md` §3 |

## Etapa 2 · EDA (validada por el grupo el 31 oct 2026)

El grupo revisó las 10 decisiones y **acordó que todas son relevantes**. Son los puntos clave del proyecto:

| # | Decisión | Cifra que la respalda | Motivo corto |
|---|---|---|---|
| 1 | Quedarse con pedidos entregados y con fecha de entrega | 96.470 de 99.441 | Sin fecha no hay objetivo |
| 2 | Excluir entregas de más de 100 días | 64 pedidos (0,07 %) | Distorsionan MAE/RMSE |
| 3 | Una fila por pedido (agregar ítems) | 9.803 pedidos con varios ítems | La unidad es el pedido |
| 4 | Eliminar columnas que se conocen después de comprar | `order_delivered_*`, `order_status`, `review_*` | Evitar fuga de información |
| 5 | Partir train/test por cliente (`customer_unique_id`) | 2.997 clientes repiten (máx. 17) | Un cliente en train y test infla el resultado |
| 6 | Distancia vendedor–cliente en km | Correlación 0,39 con el objetivo | Variable más fuerte |
| 7 | Limpiar productos (categoría desconocida, peso 0 = vacío) | 610 sin categoría; 4 con peso 0 | No perder pedidos; imputar solo con train |
| 8 | No usar ciudad ni código postal; agrupar vendedores raros | 4.119 ciudades; 1.714 vendedores con < 10 ítems | Evitar memorización |
| 9 | Periodo 2017-01 a 2018-08 | 2016: 329 pedidos; 2018-09/10: 20 | Meses casi vacíos no representan la operación |
| 10 | Descartar reseñas e identificadores | 58–88 mil vacíos en texto libre | Datos personales y fuga |

## Etapa 3 · Preprocesamiento

| Fecha | Cambio / decisión | Motivo | Dónde |
|---|---|---|---|
| 31 oct 2026 | EDA marcado como cerrado (10 decisiones validadas). | Acuerdo del grupo. | `especificaciones/02-eda.md` |
| 31 oct 2026 | Borrador de `03-preprocesamiento.md` a partir de la tabla del EDA. Pendiente que el grupo lo apruebe. | Regla del proyecto: especificación llena antes del código. | `especificaciones/03-preprocesamiento.md` |

| 31 oct 2026 | Grupo aprueba la especificación de la etapa 3: prueba 20 % por `customer_unique_id` (semilla 42); revisión por fecha en la etapa 7 sobre el mismo conjunto de prueba; orden de pasos propuesto; vendedores raros (< 10 ítems) contados solo con train. | Evitar fuga y no crear una segunda partición. | `especificaciones/03-preprocesamiento.md` |

| 31 oct 2026 | Creada la especificación de Spec Kit de la etapa 3 (4 historias, 18 requisitos, 8 criterios de éxito), sin dudas pendientes. | Paso obligatorio antes de plan, tareas y código. | `specs/001-preprocesamiento-datos/spec.md` |

| 31 oct 2026 | Constitución de OnTrack completada (v1.0.0): se llenó el principio II y se agregó el VIII, «uso responsable del modelo» (estimación para planificar, no promesa; revisar desempeño por región). | Que el plan se revise contra reglas firmes. | `.specify/memory/constitution.md` |
| 31 oct 2026 | Plan de la etapa 3 (pendiente de aprobación): funciones en 6 módulos de `src/olistbr/data/`, salida en CSV, partición `GroupShuffleSplit`. Decisiones nuevas a confirmar: peso del pedido vacío si falta algún ítem; descartar coordenadas fuera de Brasil antes de promediar; vendedores raros contados por ítems del vendedor principal. | Mantener el orden aprobado y no agregar dependencias. | `specs/001-preprocesamiento-datos/plan.md` |

| 31 oct 2026 | Peso del pedido: queda vacío si algún ítem no tiene peso válido; se imputa en la etapa 4. El grupo delegó la elección. | Solo 22 de 96.478 pedidos entregados (0,02 %) lo tienen; sumar solo lo conocido subestimaría el peso sin avisar. | `specs/001-preprocesamiento-datos/research.md` D3 |
| 31 oct 2026 | Coordenadas erróneas (42 filas, 21 códigos postales fuera de Brasil): se reemplazan por el promedio de las filas buenas del mismo código postal o, si no hay, por la mediana de la misma ciudad y estado; sin punto si tampoco hay. | Un punto equivocado movería el promedio del código postal y la distancia es la variable más fuerte (correlación 0,39). | `specs/001-preprocesamiento-datos/research.md` D4 |
| 31 oct 2026 | Salida en Parquet (`train/test/extreme_orders.parquet`) y `pyarrow==25.0.1` agregado a `requirements.txt`. | El modelo se desplegará en Databricks; Parquet conserva tipos y fechas. | `requirements.txt`, `research.md` D1 |

| 31 oct 2026 | Plan aprobado por el grupo. Generada la lista de tareas (39 tareas en 7 fases; cada tarea de código con su prueba). Pendiente que el grupo la apruebe antes de escribir código. | Regla del proyecto: no hay código sin `plan.md` y `tasks.md` aprobados. | `specs/001-preprocesamiento-datos/tasks.md` |

| 31 oct 2026 | Lista de tareas aprobada y etapa 3 implementada: 6 módulos en `src/olistbr/data/`, 62 pruebas en verde (45 nuevas), notebook 03 ejecutado desde cero. | Regla del proyecto: código solo con plan y tareas aprobados; cada tarea se marca con su prueba. | `specs/001-preprocesamiento-datos/tasks.md` |

### Resultados de la etapa 3 (cifras reales del código)

| Dato | Valor |
|---|---|
| Pedidos entregados en la tabla final | 96.139 (una fila por pedido) |
| Pedidos de más de 100 días, aparte | 64 (máximo 209,6 días) |
| Fuera del periodo 2017-01 a 2018-08 | 267 |
| Entrenamiento / prueba | 76.924 / 19.215 pedidos (20,0 % de las filas); 74.425 / 18.607 clientes |
| Pedidos sin distancia (`missing_coords`) | 480 (0,5 %) |
| Coordenadas erróneas | 21 códigos postales: 16 corregidos con sus otras filas, 1 con ciudad y estado, 4 sin punto |
| Peso / volumen vacíos | 22 / 16 pedidos |
| Vendedores frecuentes (≥ 10 ítems en train) | 1.164; el resto es «otros» (6,8 % de los pedidos de train) |
| Reproducibilidad | dos ejecuciones del notebook generan Parquet idénticos |

**Hallazgo para contar:** el EDA hablaba de 2.965 pedidos sin fecha de entrega; el código quita 2.971 porque otros 6 son `canceled` pero traen fecha de entrega (verificado con los datos).

| 31 oct 2026 | Revisión con los subagentes `auditor-de-fuga` (sin fuga grave) y `revisor-de-codigo-ia` (aceptar con cambios). Se aplicaron los arreglos: el orden «partir y luego agrupar vendedores» pasó a `prepare_train_test` en `src/` con prueba; contraste de partición por fila sobre la función real; casos borde; textos del notebook sin cifras del EDA. | Que el paso más delicado contra fuga no dependa del notebook y que cada cifra sea la del código. | `reports/02_preprocesamiento.md` |
| 31 oct 2026 | **Para la etapa 4:** excluir `order_approved_at`, `customer_unique_id` y `order_id` de las entradas del modelo y medir con y sin `order_approved_at` y `order_estimated_delivery_date`. **Para la etapa 7:** reportar el MAE de la prueba y también incluyendo los 64 pedidos extremos, porque la prueba no los trae. | Hallazgos del auditor de fuga. | `reports/02_preprocesamiento.md` |

| 31 oct 2026 | Los datos procesados (`data/processed/`, incluida la prueba cerrada) **no se versionan**: se agregó al `.gitignore`. | La guía del curso (`kit/guia-eda-y-preprocesamiento.md` §4.4) lo dice y el ejemplo CreditSight lo repite; así la prueba no puede subirse por accidente. | `.gitignore` |
| 31 oct 2026 | Código modularizado siguiendo el ejemplo CreditSight: `config.py` central, `load_raw` y el encadenado en `data/cleaning.py`, `assert_disjoint_groups` en `data/split.py`, distancia y categorías raras en `features/`, costo del error 3 a 1 en `models/metrics.py` (antes dentro del notebook 01), pruebas por módulo. **Sin cambiar resultados**: Parquet con los mismos hashes y salidas de los notebooks 01 a 03 idénticas; 62 pruebas en verde. | Rúbrica B1 (15 %): `src/` con la lógica y notebooks que solo orquestan. | `reports/02_preprocesamiento.md` |

| 31 oct 2026 | Borrador de la especificación de la etapa 4 (variables) siguiendo `engineering.py` del ejemplo: 10 ideas de variables como hipótesis (transformaciones logarítmicas, mismo estado, región, plazo prometido, calendario de compra, interacción distancia × mismo estado, historial de vendedor/ruta, razones de flete), 5 descartadas de entrada y una regla de decisión escrita (`mejora / ruido > 1`). Quedan 4 preguntas para el grupo, la principal: ¿la app predice con o sin el plazo prometido por Olist? | Regla del proyecto: especificación antes del código; el plazo prometido puede ser circular si la app sirve para fijar ese plazo. | `especificaciones/04-features.md` |

| 31 oct 2026 | Respuestas a la etapa 4: **la app recomienda la fecha prometida** y no usa el plazo prometido por Olist como entrada (queda como referencia medida una vez); el retraso es secundario y se define respecto a la fecha recomendada (propuesta de Claude, el grupo debe confirmarla). Se aprobó medir variables históricas con `TargetEncoder` y usar `LinearRegression` como modelo fijo. | Usar la fecha de Olist para predecir sería circular si la app la fija; el retraso afecta solo al 6,8 % de los pedidos (clase desbalanceada). | `especificaciones/04-features.md` §1 |
| 31 oct 2026 | **Temporadas:** el mes de compra suelto se descarta y se prueba una marca de temporada alta de fin de año con calendario fijo. Se agrega una prueba hacia adelante (entrenar hasta feb-2018, validar mar a ago-2018). | El patrón mensual de ene–ago casi no se repite entre 2017 y 2018 (correlación 0,14: es tendencia, no temporada); el pico de fin de año solo se observa una vez (2017). El día de la semana sí es estable (correlación 0,72). | `especificaciones/04-features.md` Q3 |

| 31 oct 2026 | El grupo **confirmó** las decisiones de la etapa 4 (Q1 a Q4 y las ideas de variables) y las registró en la sección 7 de `especificacion-proyecto.md`. Especificación de la etapa 4 cerrada. | Requisito antes de pasar a Spec Kit. | `especificaciones/04-features.md` |
| 31 oct 2026 | **Flujo de trabajo:** todo en `main`, sin más ramas. La rama `etapa-3-preprocesamiento` se fusionó a `main` (avance directo, sin conflictos); queda pendiente borrarla del repositorio. | Decisión del grupo: un solo historial. | repositorio |

_(Las siguientes líneas se agregan al avanzar: etapa 4 y siguientes.)_
