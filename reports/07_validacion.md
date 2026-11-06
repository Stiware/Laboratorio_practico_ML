# Validación e interpretación

> Las cifras salen de `notebooks/07_validacion_interpretacion.ipynb` y de `reports/07_metricas_test.json`, `07_segmentos.csv` y `07_shap.csv`.
> Reglas: `especificaciones/07-validacion.md` (cerrada el 5 nov 2026) · Código: `src/olistbr/models/validation.py` · Pruebas: `tests/test_validation.py`.
> Unidades de costo: costo 3 a 1 por pedido (1 unidad = un día de sobrestimación). «Plazo» = días que recomendaría la app.

## 1. Decisiones congeladas (escritas ANTES de abrir la prueba)

| Decisión | Valor | Cómo se decidió (con qué datos) |
|---|---|---|
| Modelo final | LightGBM cuantil 0,75, reajustado con todo el entrenamiento (76.924 pedidos), 7 grupos de variables del torneo | Torneo (etapa 5) y comprobación en la reserva (etapa 6): ganó a HistGradientBoosting cuantil 0,75 con todo el intervalo bajo cero |
| Parámetros (optimizados o de partida) | **Optimizados**: 418 árboles, tasa 0,049, 70 hojas, mínimo 73 pedidos por hoja, `subsample` 0,95, `colsample_bytree` 0,65, `reg_lambda` 4,43; percentil 0,75 fijo | Regla de la etapa 6 en la reserva: −0,068 por pedido [−0,096 ; −0,041], todo bajo cero (`reports/06_best_params.json`). Advertencias de esa etapa: hacia adelante el optimizado costó lo mismo que la partida (+0,0002) y HistGradientBoosting optimizado quedó por debajo (informativo) y la ventaja sobre HistGradientBoosting tuvo un intervalo [−0,038 ; −0,002] que roza el cero |
| ¿Calibrar las probabilidades? | No aplica (no hay probabilidades). **No se recalibra el plazo**: la prueba se partió por cliente, no por fecha; la recalibración queda como regla de uso (vigilar la cobertura mensual) | Q2 de `especificaciones/07-validacion.md`; cobertura de la reserva 73,9 % y hacia adelante 64 % a 92 % según el mes |
| Métrica principal | Costo 3 a 1 del plazo recomendado (subestimar cuesta 3 veces más); además % tarde, cobertura, MAE y sesgo del plazo | Caso de negocio y Q1 del torneo |
| Regla de decisión | El plazo recomendado es el percentil 75 que predice el modelo (sin margen). **Mejora con confianza** si todo el intervalo del 95 % de `costo(modelo) − costo(plazo fijo del percentil 75)` está bajo cero (1.000 sorteos de clientes, semilla 42) | Q3 de `especificaciones/07-validacion.md` |
| Baselines | Principal: plazo fijo del percentil 75 del entrenamiento. Referencia: siempre la mediana y el plazo prometido por Olist (solo se mide; nunca entra al modelo) | Q3 |
| No evaluado | Marca de fin de año (F6), historial por vendedor (F8), probabilidad de retraso por pedido | Q7 y Q8 |
| **Fecha y hora en que se congelaron** | **2026-11-05 20:44:38 (UTC−5)**, según la marca que el propio notebook guardó en `reports/07_decisiones_congeladas.json` 6 segundos antes de abrir la prueba. Esa marca es la evidencia del orden; este informe se completó **después** de ver los resultados | `notebooks/07_validacion_interpretacion.ipynb`, celda de la sección 1 |

## Resumen

- **El modelo mejora al plazo fijo con confianza.** En la prueba (19.215 pedidos de 18.607 clientes que el modelo nunca vio), el costo 3 a 1 del modelo es
  **9,216** [9,005 ; 9,422] contra **11,709** [11,468 ; 11,946] del plazo fijo del percentil 75: **−2,493 por pedido** [−2,608 ; −2,373], un **21 % menos**.
  Todo el intervalo de la diferencia está bajo cero.
- **La cifra es coherente con la validación:** 9,083 en la reserva de la etapa 6 (algo optimista, como se esperaba) y 9,216 en la prueba.
- **La cobertura quedó un poco corta:** 73,6 % [72,9 % ; 74,1 %]; el intervalo no llega al 75 % esperado, así que llegan tarde algo más pedidos de los planeados (26,4 %).
- **El plazo funciona muy distinto según el mes de compra:** de noviembre de 2017 a marzo de 2018 llega tarde entre el 33 % y el 46 % de los pedidos; de junio a
  agosto de 2018, solo entre el 11 % y el 13 %. El modelo no capta el ritmo cambiante de las entregas.
- **Si atrasarse costara 9 veces más que sobrar** (y no 3), el plazo que promete Olist sería mejor que el modelo (18,0 contra 19,9).

## 2. Calibración (con entidades reservadas, no con la prueba)

- ¿Se evaluó? ☒ sí  ☐ no — **no hay probabilidades que calibrar** (el modelo da un plazo). El equivalente es la **cobertura** del plazo, que se midió en la reserva
  de la etapa 6 (73,9 % para el modelo elegido) y hacia adelante (de 64 % a 92 % según el mes).
- Resultado y decisión: **no se recalibró el modelo** (Q2), porque la prueba se partió por cliente y mezcla todos los meses: no podía evaluar una recalibración en el
  tiempo. La recalibración queda como **regla de uso** (ver «Conclusión»). La prueba confirma que hacía falta pensarla: la cobertura mensual va de 54 % a 89 %.

## 3. La prueba, abierta **una sola vez para medir**

- Fecha y hora en que se abrió (la marca está en `reports/prueba_abierta.txt`): **2026-11-05 20:44:44 (UTC−5)**, 6 segundos después de que el notebook guardara las decisiones congeladas
  (`reports/07_decisiones_congeladas.json`, 20:44:38). Esa marca y la constancia son la evidencia del orden.
- ¿Se cambió algo del modelo después de verla? ☒ no  ☐ sí.
- **Segunda lectura (declarada):** la etapa 8 volvió a leer la prueba una vez, a las 20:47, **después** de registrar el modelo y **sin decidir nada**, solo para comprobar que el
  modelo registrado predice igual (decisión del grupo, `especificaciones/08-registro.md`). Esa lectura no pasó por `open_test_once` y no dejó constancia propia; el resultado
  fue idéntico (costo 9,216262 en las dos). La prueba no se vuelve a leer en la etapa 9.

| Métrica (en la prueba) | Modelo final | Plazo fijo p75 (baseline principal) | Siempre la mediana | Plazo prometido por Olist |
|---|---:|---:|---:|---:|
| **Costo 3 a 1** [IC 95 % por cliente] | **9,216** [9,005 ; 9,422] | 11,709 [11,468 ; 11,946] | 14,169 [13,878 ; 14,479] | 13,912 [13,763 ; 14,068] |
| % de pedidos tarde [IC 95 %] | 26,4 % [25,9 ; 27,1] | 24,6 % [24,1 ; 25,2] | 50,2 % [49,5 ; 50,9] | 8,1 % [7,7 ; 8,5] |
| Cobertura (llega a tiempo) | 73,6 % [72,9 ; 74,1] | 75,4 % [74,8 ; 76,0] | 49,8 % | 91,9 % |
| MAE del plazo (días) | 5,66 | 7,50 | 5,98 | 12,54 |
| Sesgo (real − plazo, días) | −2,10 | −3,29 | +2,20 | −11,17 |

**¿Mejora al baseline con confianza?** **Sí.** `costo(modelo) − costo(plazo fijo p75)` = **−2,493** [−2,608 ; −2,373]: todo el intervalo bajo cero (1.000 sorteos
de clientes, semilla 42). El MAE y el sesgo son del **plazo** (el percentil 75, más alto que el valor típico a propósito); no son el error del «día más probable».

**Holgura y atraso** (en lugar de «carga de trabajo», que no aplica: aquí no hay revisión humana):

| Regla | % a tiempo | Días que sobran cuando llega a tiempo | Días que faltan cuando llega tarde |
|---|---:|---:|---:|
| Modelo final | 73,6 % | 5,3 | 6,7 |
| Plazo fijo p75 | 75,4 % | 7,2 | 8,6 |
| Plazo prometido por Olist | 91,9 % | 12,9 | 8,5 |

El modelo promete **casi 2 días menos de sobra** que el plazo fijo y, cuando se atrasa, se atrasa **casi 2 días menos**. Olist casi nunca llega tarde porque promete
13 días de sobra en promedio.

**Sensibilidad a la proporción del costo** (subestimar : sobrestimar; el plazo del modelo es siempre el percentil 75, no se reoptimiza):

| Regla | 1 a 1 (= MAE) | 2 a 1 | **3 a 1** | 5 a 1 | 9 a 1 |
|---|---:|---:|---:|---:|---:|
| Modelo final | 5,66 | 7,44 | **9,22** | 12,77 | 19,88 |
| Plazo fijo p75 | 7,50 | 9,61 | 11,71 | 15,92 | 24,33 |
| Siempre la mediana | 5,98 | 10,08 | 14,17 | 22,36 | 38,73 |
| Plazo prometido por Olist | 12,54 | 13,23 | 13,91 | 15,28 | 18,03 |

El modelo gana al plazo fijo con todas las proporciones y a Olist hasta 5 a 1. **Con 9 a 1 gana Olist**: si el negocio considerara que un atraso cuesta tanto, el plazo
debería ser otro percentil (≈ 0,9), algo que no se evaluó.

**Escenario pesimista con los 64 pedidos de más de 100 días** (19.279 pedidos, sin intervalo; los 64 son de 64 clientes distintos y **ninguno** está en el
entrenamiento, comprobado con `train.parquet`, aunque no se separaron con la partición por cliente): modelo 10,43 contra 12,95 del plazo fijo; % tarde 26,7 % contra 24,9 %;
MAE 6,06 contra 7,90. Los extremos encarecen a todos, y la ventaja del modelo se mantiene.

## 4. Errores y carga de trabajo

| Región del cliente | Pedidos | % tarde (esperado ≈ 25 %) | Cobertura | MAE del plazo | Días que sobran | Días que faltan |
|---|---:|---:|---:|---:|---:|---:|
| Centro-Oeste | 1.140 | 26,9 % | 73,1 % | 5,88 | 6,1 | 5,4 |
| Nordeste | 1.770 | 23,1 % | 76,9 % | 8,50 | 8,2 | 9,6 |
| Norte | 306 | 29,7 % | 70,3 % | 9,14 | 8,9 | 9,7 |
| Sudeste | 13.322 | 26,8 % | 73,2 % | 5,14 | 4,7 | 6,5 |
| Sul | 2.677 | 26,2 % | 73,8 % | 5,89 | 5,7 | 6,5 |

| Mes de compra | Pedidos | % tarde | Mes de compra | Pedidos | % tarde |
|---|---:|---:|---|---:|---:|
| 2017-01 | 153 | 24,2 % | 2017-11 | 1.430 | **38,1 %** |
| 2017-02 | 343 | 25,1 % | 2017-12 | 1.117 | **42,0 %** |
| 2017-03 | 554 | 24,7 % | 2018-01 | 1.446 | **32,5 %** |
| 2017-04 | 495 | 35,6 % | 2018-02 | 1.309 | **43,5 %** |
| 2017-05 | 714 | 18,5 % | 2018-03 | 1.352 | **45,6 %** |
| 2017-06 | 602 | 22,9 % | 2018-04 | 1.306 | 20,9 % |
| 2017-07 | 766 | 18,5 % | 2018-05 | 1.280 | 28,9 % |
| 2017-08 | 802 | 17,6 % | 2018-06 | 1.216 | 12,7 % |
| 2017-09 | 813 | 18,7 % | 2018-07 | 1.264 | 12,6 % |
| 2017-10 | 971 | 18,3 % | 2018-08 | 1.282 | 10,7 % |

- **¿Hay un segmento al que le va peor? ¿Por qué podría ser?:**
  - **Por mes, sí, y mucho.** Entre noviembre de 2017 y marzo de 2018 el plazo queda corto: entre 33 % y 46 % de pedidos tarde. También abril de 2017 (35,6 %, 495 pedidos).
    De junio a agosto de 2018 sobra: solo 11 % a 13 % tarde. El modelo no usa el mes de compra (se descartó en la etapa 4: `especificaciones/04-features.md` §4.3).
    **Hipótesis, no probada aquí:** el ritmo de entrega cambia con la demanda (fin de año, inicio de 2018) y el modelo aprende un ritmo promedio. Es el mismo patrón del
    torneo y de la optimización, ahora con clientes nuevos.
  - **Por región, poco.** El Norte queda en 29,7 %, pero con 306 pedidos la desviación de muestreo de ese porcentaje es de ≈ 2,5 a 2,6 puntos: no se puede afirmar que esté peor. El Norte y el
    Nordeste tienen el **mayor error absoluto** (MAE ≈ 9 y 8,5 días): sus entregas son más largas y variables.
- Si el modelo prioriza revisión humana: **no aplica** (no hay revisión humana). El costo de un plazo largo es prometer días de más: ver «Holgura y atraso».

## 5. Interpretación

Valores SHAP sobre 2.000 pedidos de la prueba (cuántos días sube o baja el plazo cada variable, en promedio y en valor absoluto):

| # | Variable | Peso medio (días, valor absoluto) | Efecto medio con signo |
|---:|---|---:|---:|
| 1 | `same_state` (vendedor y cliente en el mismo estado) | 1,93 | +0,06 |
| 2 | `distance_km` | 1,06 | +0,00 |
| 3 | `log_distance_x_same_state` | 0,66 | −0,00 |
| 4 | `customer_state_SP` | 0,57 | −0,14 |
| 5 | `log_distance` | 0,47 | −0,04 |
| 6 | `freight_sum` (flete) | 0,44 | +0,09 |
| 7 | `customer_state_MG` | 0,28 | +0,03 |
| 8 | `customer_state_RJ` | 0,26 | −0,04 |
| 9 | `customer_region_Nordeste` | 0,24 | −0,03 |
| 10 | `is_slow_weekday` (compra de jueves a sábado) | 0,23 | −0,00 |

- **Variables que más pesan:** la **geografía**: si el pedido se queda en el mismo estado y qué tan lejos viaja (las variables 1, 2, 3 y 5 son
  formas de la misma idea). Luego el **estado del cliente** (SP, MG, RJ y la región Nordeste) y el **flete**. Para el negocio: el modelo se apoya sobre todo en la ruta.
  El «efecto medio con signo» es poco interpretable en variables como `customer_state_SP` (promedia pedidos de SP y de otros estados); se reporta solo como referencia.
- **Límite de la explicación:** explica al modelo, no al mundo; no dice si el modelo acertó. Las variables de distancia están correlacionadas y se reparten el crédito
  (quitar una subiría el peso de otra). Una importancia alta no es una causa.
- **Atributos sensibles o de uso regulado:** la **ubicación del cliente** (`customer_state`, `customer_region`) aparece 4 veces entre las 10 primeras, y las de los puestos
  1 a 3 también dependen de la geografía. No es un dato
  personal, pero hace que el plazo cambie según dónde vive el cliente (este informe no midió cuánto cambia el plazo medio por región). Antes de usarlo en producción habría que
  revisar con el negocio que eso refleja la logística y no un trato injusto, y vigilar el % tarde por región.

## 6. Conclusión

- **¿El modelo sirve para la decisión del caso? ¿Con qué condiciones?:** **Sí, como herramienta para recomendar una fecha de entrega.** Reduce el costo del error un 21 %
  frente al mejor plazo fijo, con confianza, y promete casi 2 días menos de sobra. **Condiciones:** (1) el negocio debe aceptar que ≈ 1 de cada 4 pedidos llegue después
  de la fecha (es lo que implica el costo 3 a 1); (2) **vigilar la cobertura cada mes** y recalibrar si sale de una banda (por ejemplo, 70 % a 80 %), porque en los meses
  de alta demanda el plazo queda corto; (3) revisar el % tarde por región.
- **Qué NO debe hacerse con él:** usarlo como una **promesa garantizada** (es una estimación para planificar; principio VIII); usarlo en meses o temporadas fuera de
  2017-01 a 2018-08 sin volver a medir; usarlo si el costo de un atraso es mucho mayor que 3 veces el de sobrar (con 9 a 1 el plazo de Olist es mejor); usar su MAE
  como «error del día más probable» (es el error del percentil 75).
- **Riesgo técnico:** el modelo se ajustó y se registró con Python 3.14 (el `.venv` de esta máquina); el proyecto pide 3.12. Antes de exportarlo para la app (etapa 9)
  hay que comprobar que carga en el entorno de la app.
- **No evaluado:** la marca de fin de año (F6), el historial por vendedor (F8), una probabilidad de retraso por pedido, otros percentiles para otras proporciones de costo
  y el desempeño en meses posteriores a agosto de 2018 (la prueba es de clientes nuevos del mismo periodo, no del futuro).

## Revisiones

### Auditoría de fuga (`auditor-de-fuga`)

**Veredicto: fuga leve; ningún problema de datos ni de modelo.** Confirmó que las decisiones se congelaron antes de abrir la prueba (20:44:38 contra 20:44:44), que el modelo
se ajustó solo con el entrenamiento dentro del `Pipeline` y con los parámetros de la etapa 6, que los baselines salen del entrenamiento, que el plazo prometido no entra al
modelo, que los intervalos sortean clientes y que el ejemplo registrado no lleva identificadores.

| Hallazgo | Qué se hizo (sin cambiar el modelo ni reabrir la prueba) |
|---|---|
| La prueba se leyó dos veces; la segunda (etapa 8) se salta `open_test_once` | Declarado en la sección 3 («una sola vez para medir» y la segunda lectura con su hora y su motivo) |
| La evidencia del «congelado antes» depende de una marca escrita por el mismo notebook; los archivos no están en Git | La sección 1 cita esa marca como la evidencia y dice que el informe se completó después. **Recomendación al grupo:** hacer commit de `07_decisiones_congeladas.json`, `prueba_abierta.txt`, `07_metricas_test.json` y este informe |
| Los 64 extremos no se separaron por cliente | Comprobado con `train.parquet`: ninguno de sus 64 clientes está en el entrenamiento. Anotado en el escenario |
| Explicaciones causales sin respaldo («pico de fin de año», «ritmo promedio») y abril de 2017 sin mencionar | Marcadas como hipótesis; agregado abril de 2017 |

### Revisión de código (`revisor-de-codigo-ia`)

**Veredicto: aceptar con cambios.** 18 de 18 pruebas pasaban; todas las cifras cotejadas de este informe coinciden con los archivos. Ninguna corrección cambia una cifra.

| Hallazgo | Qué se hizo |
|---|---|
| «Abierta una sola vez» era literalmente falso | Corregido (ver arriba) |
| «Ya estaba escrita desde las 20:44» no se podía comprobar (el archivo se modificó después) | Quitado; la evidencia es la marca del notebook |
| Afirmaciones causales y omisión del resultado hacia adelante de la etapa 6 | Marcadas como hipótesis; agregadas las advertencias de la etapa 6 en la sección 1 |
| Python 3.14 en lugar de 3.12 | Declarado como riesgo técnico para la etapa 9 |
| Pruebas débiles: el intervalo de la cobertura podía estar invertido sin que nadie lo notara; la semilla distinta no se probaba | Pruebas nuevas (`test_coverage_interval_mirrors_late_interval`, `test_bootstrap_mean_changes_with_seed`) |
| Casos borde: valores no finitos en `slack_and_delay`, etiquetas vacías y largos distintos en `segment_table`, objetivo o fechas faltantes en `baseline_plans` | Ahora fallan con un mensaje claro o agrupan «sin dato» (pruebas nuevas). No cambian ninguna cifra: la prueba no tenía esos casos |
| Duplicación: `bootstrap_mean` repite la lógica de `bootstrap_cost_difference` (etapa 6); `segment_table` rehace parte de `metrics_by_group` | Se deja como deuda técnica declarada: la regla de esta pieza era no modificar módulos existentes |
| El escenario con extremos vive en el notebook, no en `src/` | Declarado: es un cálculo de tres líneas con funciones probadas (`mean_delivery_cost`, `mae_days`) |
| El nombre `test_report` obliga a un truco en las pruebas | Se mantiene (lo fija el contrato) |

**Preguntas que dejó el revisor para quien no escribió el código:** (1) ¿por qué el intervalo sería demasiado angosto si se sortearan pedidos sueltos? (2) ¿por qué la
segunda lectura de la prueba no cambia las cifras y qué garantiza que el modelo registrado es el evaluado? (3) ¿cómo puede el modelo costar 21 % menos que el plazo fijo
y a la vez llegar tarde en más pedidos (26,4 % contra 24,6 %)? Pista: el plazo fijo promete casi 2 días más de sobra a todos los pedidos.
