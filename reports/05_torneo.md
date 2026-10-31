# Etapa 5 · Torneo de modelos (Fase 1)

> Las cifras de las tablas y de cada decisión salen de `notebooks/05_torneo.ipynb` (ejecutado con la semilla 42, solo con `train.parquet`) y de
> `reports/05_torneo_por_pliegue.csv`, `05_torneo_resumen.csv`, `05_torneo_regiones.csv`, `05_torneo_cobertura_mensual.csv` y `05_torneo_tiempos.csv`.
> Las estadísticas mensuales de los días reales en el periodo de entrenamiento y el estado de los pedidos se calcularon aparte (con `train.parquet` y la tabla
> original de pedidos, solo lectura) y se marcan en el texto. Reglas: `especificaciones/05-torneo.md` · Código: `src/olistbr/models/` ·
> Pruebas: `tests/test_models.py`, `test_metrics.py`, `test_evaluation.py`.
> **Esto es la Fase 1 (torneo). La optimización de hiperparámetros (Fase 2) se hace en otra sesión.**

## Resumen

Se evaluaron 10 candidatos (2 líneas base, 5 modelos de la familia A —valor típico más un margen— y 3 de la familia B —percentil 75 directo—) con los mismos 5 pliegues
por cliente de la etapa 4 y la prueba hacia adelante. **El costo 3 a 1 del plazo recomendado decide al ganador.**

- **Gana LightGBM cuantil 0,75** (familia B) con un costo de **9,367 ± 0,040**, y también es el mejor hacia adelante (9,230).
  Es un **costo de selección**: se eligió el mejor de 10 con los mismos pliegues con los que se reporta, así que es algo optimista. La estimación honesta del modelo final sale de la prueba de la etapa 7.
- **Ventaja pequeña sobre el segundo.** Contra todos los demás gana en los 5 pliegues, pero contra HistGradientBoosting cuantil 0,75 (9,400) la diferencia es de 0,034 por pedido
  (0,36 % del costo del ganador), con diferencia / ruido 2,06. Ganar 5 de 5 por puro azar tendría probabilidad 1/32 (3 %) si los dos modelos fueran idénticos:
  es una señal, no una prueba.
- **Contra las líneas base:** el ganador cuesta 21 % menos que prometer siempre el percentil 75 (11,840) y 35 % menos que prometer la mediana (14,334).
  Los 8 modelos superan a las dos líneas base por más que su ruido.
- **Cobertura en los pliegues:** los plazos de los modelos de las familias A y B cubren entre 74,4 % y 75,3 % de los pedidos (esperado ≈ 75 %).
  **Hacia adelante cubren entre 82,3 % y 82,9 %, y por mes va de 61 % a 93 %** (ver «Hallazgo»).
- **La regresión lineal queda en la posición 7** (9,664): cuesta **0,298 más por pedido** que el ganador (3,2 % más) en los 5 pliegues
  (diferencia / ruido 20,37), y 0,203 más hacia adelante. La ventaja es consistente según la regla. **El grupo decidió el 31 oct 2026 que el modelo final no tiene que ser la regresión lineal** («al final se busca el mejor modelo»); la lineal queda como referencia interpretable.
- **Regiones:** los modelos que predicen el percentil 75 directamente (familia B) son mucho más parejos entre regiones que los de valor típico más un margen único (familia A): el rango del porcentaje de pedidos tarde entre regiones es de 0,9 a 4,9 puntos en la familia B y de 8,3 a 10,8 en la A. El ganador tiene la menor dispersión de los 8 modelos. Con la regresión lineal el Norte llega a 31,4 % de pedidos tarde y el Sudeste baja a 23,1 % (esperado ≈ 25 %).

## Cómo se midió

- **Datos:** `data/processed/train.parquet` (76.924 pedidos, 74.425 clientes), con 7 grupos de variables (logaritmos, mismo estado, regiones, día de la semana, distancia × mismo estado, razones de flete y varios vendedores). Sin la marca de fin de año ni el historial por vendedor; el plazo prometido por Olist nunca entra, ni tampoco el cliente ni el pedido (hay una prueba que lo verifica). La prueba final no se abrió.
- **Pliegues:** los 5 pliegues por cliente de la etapa 4 (semilla 42), iguales para todos los modelos; **prueba hacia adelante** (entrenar hasta 2018-02-28, validar del 2018-03-01 al 2018-08-31): una sola partición, sin ruido medido.
- **Plazo recomendado:** familia A = predicción + margen, donde el margen es el percentil 75 de los errores de un 20 % de clientes apartados dentro de cada entrenamiento (la familia A se ajusta con el 80 % de los datos); familia B = el modelo ya predice el percentil 75; líneas base sin margen.
- **Costo 3 a 1:** subestimar cuesta 3 veces más que sobrestimar; equivale a 4 veces la pérdida cuantílica de 0,75 (comprobado en las pruebas). 1 unidad de costo = un día de sobrestimación.
- **Parámetros fijos, sin optimizar** (los del benchmark de tiempos); el bosque aleatorio es el liviano de 100 árboles.

## Tabla del torneo: costo 3 a 1 (decide)

| Posición | Modelo | Fam. | Costo 3 a 1 (± entre pliegues) | Costo hacia adelante | % tarde | Cobertura | Cobertura hacia adelante | Margen (días) | Ajuste (s) | Supera mediana | Supera plazo fijo |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | LightGBM cuantil 0,75 | B | 9,367 ± 0,040 | 9,230 | 25,6 % | 74,4 % | 82,3 % | 0,00 | 3,1 | sí | sí |
| 2 | HistGradientBoosting cuantil 0,75 | B | 9,400 ± 0,049 | 9,257 | 25,4 % | 74,6 % | 82,5 % | 0,00 | 7,8 | sí | sí |
| 3 | LightGBM | A | 9,470 ± 0,033 | 9,373 | 24,7 % | 75,3 % | 82,7 % | 2,26 | 2,1 | sí | sí |
| 4 | XGBoost | A | 9,471 ± 0,036 | 9,416 | 24,7 % | 75,3 % | 82,9 % | 2,28 | 5,0 | sí | sí |
| 5 | XGBoost cuantil 0,75 | B | 9,528 ± 0,054 | 9,307 | 25,1 % | 74,9 % | 82,7 % | 0,00 | 8,0 | sí | sí |
| 6 | Bosque aleatorio liviano | A | 9,612 ± 0,055 | 9,387 | 24,8 % | 75,2 % | 82,5 % | 2,30 | 6,9 | sí | sí |
| 7 | Regresión lineal | A | 9,664 ± 0,044 | 9,433 | 24,7 % | 75,3 % | 82,7 % | 2,39 | 1,2 | sí | sí |
| 8 | Ridge | A | 9,664 ± 0,044 | 9,434 | 24,7 % | 75,3 % | 82,7 % | 2,40 | 0,9 | sí | sí |
| 9 | Línea base: plazo fijo (percentil 75) | base | 11,840 ± 0,087 | 11,804 | 25,0 % | 75,0 % | 83,0 % | 0,00 | 0,8 | no | no |
| 10 | Línea base: siempre la mediana | base | 14,334 ± 0,077 | 11,697 | 50,0 % | 50,0 % | 65,7 % | 0,00 | 0,9 | no | no |

`±` es la desviación entre los 5 pliegues. «Ajuste» es el tiempo medio por pliegue (cambia entre ejecuciones; está en `05_torneo_tiempos.csv`). «Supera» dice si vence a esa línea base por más que su ruido.
El margen de la familia A varía entre pliegues de 2,23 a 2,43 días (rango por modelo en `05_torneo_por_pliegue.csv`).

**Cambios de orden hacia adelante.** El primero y el segundo puesto se mantienen, pero el resto cambia: XGBoost cuantil 0,75 pasa del puesto 5 al 3; XGBoost pasa del puesto 4 al 6. Hacia adelante, además, la línea base de la mediana (11,697) cuesta menos que la del percentil 75 (11,804), otra señal de que el percentil 75 del pasado quedó largo.

## Exactitud: MAE, RMSE, R² y sesgo (solo comparables dentro de cada familia)

| Fam. | Modelo | MAE (días) | RMSE | R² | Sesgo | MAE hacia adelante |
|---|---|---:|---:|---:|---:|---:|
| A | LightGBM | 4,981 ± 0,010 | 7,491 | 0,290 | 0,002 | 5,506 |
| A | XGBoost | 4,990 ± 0,013 | 7,508 | 0,287 | -0,015 | 5,551 |
| A | Bosque aleatorio liviano | 5,055 ± 0,014 | 7,570 | 0,275 | -0,001 | 5,463 |
| A | Regresión lineal | 5,100 ± 0,016 | 7,606 | 0,268 | 0,002 | 5,546 |
| A | Ridge | 5,100 ± 0,016 | 7,606 | 0,268 | 0,002 | 5,546 |
| B | LightGBM cuantil 0,75 | 5,801 ± 0,014 | 7,854 | 0,219 | -2,236 | 6,725 |
| B | HistGradientBoosting cuantil 0,75 | 5,839 ± 0,031 | 7,883 | 0,214 | -2,278 | 6,768 |
| B | XGBoost cuantil 0,75 | 5,931 ± 0,020 | 7,944 | 0,201 | -2,334 | 6,787 |
| base | Línea base: siempre la mediana | 6,036 ± 0,025 | 9,173 | -0,065 | 2,262 | 6,007 |
| base | Línea base: plazo fijo (percentil 75) | 7,535 ± 0,039 | 9,459 | -0,132 | -3,231 | 8,829 |

**Cómo leerla.** Los modelos de la familia B predicen el **percentil 75**, no el valor típico: su sesgo es negativo (quedan por encima del valor típico a propósito) y su MAE, RMSE y R² salen peores **por diseño**, no porque sean peores modelos. Entre familias solo se compara el costo.
Dentro de la familia A, el de mejor MAE es LightGBM (4,981 días); la regresión lineal tiene 5,100.

## Comparación por pares contra el ganador

| Modelo | Pliegues que gana el mejor | Diferencia media | Diferencia / ruido | ¿Ventaja consistente? |
|---|---:|---:|---:|---|
| HistGradientBoosting cuantil 0,75 | 5 de 5 | 0,034 (0,36 %) | 2,06 | sí, según la regla |
| LightGBM | 5 de 5 | 0,104 (1,11 %) | 6,00 | sí, según la regla |
| XGBoost | 5 de 5 | 0,104 (1,12 %) | 6,01 | sí, según la regla |
| XGBoost cuantil 0,75 | 5 de 5 | 0,162 (1,72 %) | 6,84 | sí, según la regla |
| Bosque aleatorio liviano | 5 de 5 | 0,245 (2,62 %) | 8,16 | sí, según la regla |
| Regresión lineal | 5 de 5 | 0,298 (3,18 %) | 20,37 | sí, según la regla |
| Ridge | 5 de 5 | 0,298 (3,18 %) | 20,33 | sí, según la regla |
| Línea base: plazo fijo (percentil 75) | 5 de 5 | 2,473 (26,40 %) | 39,47 | sí, según la regla |
| Línea base: siempre la mediana | 5 de 5 | 4,967 (53,03 %) | 95,66 | sí, según la regla |

Una ventaja es **consistente según la regla** si el ganador gana en al menos 4 de 5 pliegues y la diferencia supera 1 vez su ruido. Ningún candidato queda en empate práctico con el ganador, pero la ventaja sobre HistGradientBoosting cuantil 0,75 es de apenas 0,36 % del costo.
El «ruido» es la desviación de solo 5 diferencias y los pliegues comparten gran parte del entrenamiento: **no es una prueba de significancia**, y no se midió cuánto cambian los resultados con otra semilla o con otro sorteo de los clientes de calibración.

## Hallazgo: la calibración no se mantiene en el tiempo

En los pliegues los plazos cubren ≈ 75 % de los pedidos. En la prueba hacia adelante (compras de marzo a agosto de 2018) **la cobertura del ganador va de 61,1 % en 2018-03 a 93,1 % en 2018-08**:

| Mes de compra | Pedidos | Mediana de días reales | Percentil 75 de días reales | Cobertura · LightGBM cuantil 0,75 | Cobertura · HistGradientBoosting cuantil 0,75 | Cobertura · LightGBM | Cobertura · Regresión lineal |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2018-03 | 5.650 | 13,4 | 21,6 | 61,1 % | 61,2 % | 61,5 % | 61,7 % |
| 2018-04 | 5.490 | 9,4 | 14,4 | 82,9 % | 83,1 % | 83,0 % | 82,2 % |
| 2018-05 | 5.467 | 9,1 | 15,7 | 77,1 % | 77,5 % | 77,4 % | 77,5 % |
| 2018-06 | 4.879 | 8,0 | 11,6 | 92,2 % | 92,2 % | 92,5 % | 92,5 % |
| 2018-07 | 4.892 | 7,5 | 10,9 | 91,0 % | 91,1 % | 91,4 % | 91,5 % |
| 2018-08 | 5.069 | 7,0 | 9,8 | 93,1 % | 93,3 % | 94,1 % | 94,4 % |

- Marzo queda **corto** (más pedidos tarde de lo esperado) y de junio en adelante el plazo queda **más largo de lo necesario** (más del 90 % llega a tiempo). El cambio ocurre dentro de la propia ventana de validación.
- Los modelos se mueven juntos: el problema no es de un modelo, es que el ritmo de entrega cambia mes a mes (mediana de 13,4 días en 2018-03 a 7,0 en 2018-08).
- Antes del corte del entrenamiento hacia adelante hubo meses lentos (*calculado aparte desde `train.parquet`*):

| Mes de compra | Mediana de días | Percentil 75 |
|---|---:|---:|
| 2017-11 | 12,6 | 18,6 |
| 2017-12 | 13,1 | 20,0 |
| 2018-01 | 11,8 | 17,7 |
| 2018-02 | 14,3 | 21,6 |

**Posibles explicaciones (hipótesis; con estos datos no se pueden separar del todo):**
1. *Cambio del ritmo de la operación:* las entregas se aceleran de marzo a agosto de 2018.
2. *Mezcla de temporadas en el entrenamiento hacia adelante:* incluye el pico de noviembre-diciembre de 2017 y los meses lentos de enero a marzo de 2018, que elevan el percentil 75 aprendido; la validación no contiene esa temporada.
3. *Pedidos lentos que aún no se habían entregado al cortar los datos* (solo hay pedidos ya entregados): *calculado aparte con la tabla original de pedidos*, el porcentaje de pedidos todavía sin cerrar por mes de compra va de 0,75 % a 2,29 %
   y es mayor en 2018-03, no hacia 2018-08. Por eso no parece ser la causa principal.

- No cambia al ganador: sigue siendo el mejor hacia adelante. **Sí es una advertencia para la app:** un plazo aprendido una vez se desajusta cuando cambia el ritmo de entrega; conviene vigilar la cobertura y recalibrar con datos recientes (ver «Para decidir en grupo»).

## Revisión por región del cliente

Porcentaje de pedidos que llega **después** del plazo recomendado (esperado ≈ 25 %), con predicciones fuera de muestra, para todos los modelos:

| Región | Pedidos | LightGBM cuantil 0,75 | HistGradientBoosting cuantil 0,75 | LightGBM | XGBoost | XGBoost cuantil 0,75 | Bosque aleatorio liviano | Regresión lineal | Ridge |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Centro-Oeste | 4.467 | 25,6 % | 25,3 % | 27,0 % | 27,2 % | 25,3 % | 26,8 % | 27,2 % | 27,2 % |
| Nordeste | 7.223 | 26,4 % | 26,5 % | 30,6 % | 31,0 % | 27,8 % | 31,1 % | 30,3 % | 30,3 % |
| Norte | 1.480 | 26,4 % | 27,1 % | 33,3 % | 32,1 % | 29,5 % | 33,9 % | 31,4 % | 31,5 % |
| Sudeste | 52.666 | 25,5 % | 25,3 % | 23,1 % | 23,0 % | 24,6 % | 23,2 % | 23,1 % | 23,1 % |
| Sul | 11.088 | 25,7 % | 25,1 % | 26,1 % | 26,5 % | 25,2 % | 26,5 % | 26,5 % | 26,5 % |
| **Rango (máx − mín), puntos** | | **0,9** | **2,0** | **10,2** | **9,1** | **4,9** | **10,8** | **8,3** | **8,4** |

- Frente a los **modelos de la familia A** (valor típico + margen único), el ganador y los demás de la familia B son claramente más parejos: con un margen único para todo el país el **Norte y el Nordeste** quedan por encima de lo esperado y el Sudeste por debajo (la regresión lineal: 31,4 %, 30,3 % y 23,1 %).
- Frente a los otros modelos de la familia B (por ejemplo HistGradientBoosting cuantil 0,75) **no se puede ordenar**: la diferencia de dispersión es de un punto, y el error de muestreo del Norte (1.480 pedidos) es de unos ±1,1 puntos.
- Una explicación posible, no probada: los modelos que predicen el percentil 75 aprenden una incertidumbre distinta por tipo de pedido, mientras que un margen único es el mismo para todos. Un margen por región sería otra forma de resolverlo.
- El error absoluto es mayor en el Norte y el Nordeste (ver `05_torneo_regiones.csv`; solo comparable dentro de cada familia): son las regiones donde conviene mirar con más cuidado en la validación final (principio VIII).

## Finalistas propuestos (los confirma el grupo)

Reglas aprobadas, en orden: (1) consistencia, (2) costo, (3) tiempo y simplicidad —**solo ante un empate práctico**—, (4) interpretabilidad, (5) métricas técnicas y cobertura.

1. **LightGBM cuantil 0,75** (familia B): el menor costo, gana en los 5 pliegues a todos los demás, también es el mejor hacia adelante y se ajusta en 3,1 s. No necesita margen.
2. **Segundo finalista según la regla: HistGradientBoosting cuantil 0,75** (familia B, costo 9,400). Es el segundo por costo y vence a LightGBM de valor típico —el siguiente— en 5 de 5 pliegues
   (diferencia media 0,070, diferencia / ruido 2,43), es decir, por la regla es una ventaja consistente. Se parece mucho al ganador (misma familia y misma pérdida) y es más lento (7,8 s contra 3,1 s).
3. **Alternativa fuera de la regla, si el grupo la prefiere: LightGBM** (familia A, costo 9,470, MAE 4,981, ajuste 2,1 s). No es una conclusión del torneo sino una preferencia: aportaría un modelo de otra familia (con margen) y la mejor exactitud del valor típico,
   útil si la app muestra además un tiempo típico, pero hay que decir que **pierde frente a HistGradientBoosting cuantil 0,75 en los 5 pliegues**. Hacia adelante HistGradientBoosting cuantil 0,75 también queda mejor (9,257 contra 9,373).
4. XGBoost de valor típico queda en empate práctico con LightGBM de valor típico (gana 3 de 5, diferencia / ruido 0,10); gana el más rápido.
5. **Descartados:** Ridge (idéntico a la regresión lineal: 9,664 contra 9,664), el bosque liviano (9,612) y las dos líneas base. XGBoost cuantil (9,528) queda 5.º por pliegues pero sube al 3.º hacia adelante:
   no se avanza por la regla (consistencia en los pliegues), pero se anota el cambio de orden.
   La regresión lineal **no se descarta del reporte**: queda como referencia interpretable con la diferencia que se muestra abajo.

**Por qué el torneo no elige un campeón.** Los parámetros son fijos y no están optimizados: la optimización puede cambiar el orden entre los cuatro primeros (sus costos están entre 9,367 y 9,471), la ventaja entre los dos primeros es del 0,36 % y solo la prueba final de la etapa 7 dice qué tan bueno es el modelo elegido. El torneo descarta; no corona.

## La regresión lineal contra el ganador

| | Regresión lineal | LightGBM cuantil 0,75 | Diferencia |
|---|---:|---:|---:|
| Costo 3 a 1 por pedido (pliegues) | 9,664 | 9,367 | **+0,298** |
| Por cada 1.000 pedidos (unidades de costo) | 9.664 | 9.367 | **+298** |
| Costo 3 a 1 por pedido (hacia adelante) | 9,433 | 9,230 | **+0,203** |
| % de pedidos que llegan tarde en el Norte | 31,4 % | 26,4 % | |

- La diferencia no se convierte en dinero: el costo 3 a 1 es un supuesto en unidades relativas (1 unidad = un día de sobrestimación). Se expresa por pedido y por cada 1.000 pedidos.
- Dentro de la familia A, la regresión lineal con margen tiene un MAE de 5,100 días y LightGBM con margen, 4,981: los árboles también son más exactos para el valor típico. El MAE del ganador no se compara con el de la lineal (distinta familia).

## Limitaciones

1. **Parámetros fijos:** ningún modelo está optimizado; la distancia entre los cuatro primeros es pequeña y la etapa 6 puede reordenarlos.
2. **Datos distintos entre familias:** la familia A se ajusta con el 80 % de cada entrenamiento (se aparta un 20 % de clientes para el margen) y la B con el 100 %. El titular «gana la familia B» puede deberse en parte a eso y a que el margen de la A es el mismo para todos los pedidos. **No hay un control que lo aísle** (queda para la Fase 2).
3. **Costo de selección:** el costo del ganador es el del mejor de 10 elegido con los mismos pliegues; es optimista.
4. **Prueba hacia adelante:** es una sola partición, sin ruido medido. Entrena con compras hasta 2018-02-28, y algunos de esos pedidos se entregaron después del 1 de marzo (una fuga temporal pequeña: ese día todavía no se conocía su plazo de entrega).
5. **El ruido de 5 pliegues no es significancia estadística** y los pliegues comparten datos de entrenamiento; no se midió la variación por semilla ni por sorteo de calibración.
6. **La calibración depende del periodo** (ver el hallazgo): la cobertura hacia adelante va de 61 % a 93 % según el mes.
7. **Tendencia entre años** (heredada de la etapa 4): los modelos no pueden anticipar cambios posteriores a agosto de 2018. Que el costo hacia adelante sea menor que el de los pliegues en todos los modelos es coherente con entregas más rápidas, pero no se probó.
8. **Variables:** sin la marca de fin de año (una sola temporada) y sin el historial por vendedor; se evalúan después en los finalistas.
9. **Tiempos:** son del equipo del grupo con carga variable y cambian entre ejecuciones.

## Para decidir en grupo

1. **Segundo finalista:** la regla da HistGradientBoosting cuantil 0,75; ¿lo aceptan o prefieren LightGBM por razones que no son de la regla (otra familia, mejor exactitud)?
2. **Regresión lineal — DECIDIDO (31 oct 2026):** el ganador la supera de forma consistente (+0,298 por pedido, 3,2 % más) y el grupo aceptó que el modelo final no sea la regresión lineal, porque el objetivo es el mejor modelo. Queda como referencia interpretable. (Pregunta Q4 de `especificaciones/05-torneo.md`.)
3. **Regiones:** el Norte y el Nordeste tienen el mayor error absoluto y, con la regresión lineal, más pedidos tarde de lo esperado; ¿se agrega un indicador por región en la validación final?
4. **Calibración con el tiempo:** la cobertura se desplaza de un mes a otro. ¿Se recalibra el margen o el modelo con los meses más recientes antes de la validación final y se anota como parte del uso de la app (vigilar la cobertura)?
5. **Control de datos entre familias:** ¿se agrega en la Fase 2 una comparación con los mismos datos de ajuste para la familia B?

## Revisiones

### Auditoría de fuga (`auditor-de-fuga`)

**Veredicto: sin fuga en el código del torneo.** El margen de la familia A sale solo de clientes apartados dentro del entrenamiento del pliegue (nunca de la validación) y ningún cliente está a la vez en el ajuste y en la calibración; los pliegues y la partición hacia adelante son por cliente; todo lo que aprende está dentro del pliegue;
la prueba final no se abre; `promised_days`, la marca de fin de año y el historial no entran; `customer_unique_id` no llega al modelo.
Puntos que se atendieron: (1) el costo del ganador es un costo de selección (ahora lo dice el resumen); (2) la causa de la sobrecobertura estaba escrita como un hecho y ahora se presenta como hipótesis con la evidencia mes a mes (cambio de ritmo, mezcla de temporadas y pedidos sin entregar, esta última con un dato en contra);
(3) faltaba una prueba automática de que el cliente, el pedido y el plazo prometido no llegan al modelo (ahora existe). Notas sin acción: la lista de vendedores frecuentes viene de la etapa 3 (solo conteos) y la pequeña fuga temporal de la partición hacia adelante (limitación 4).

### Revisión de código (`revisor-de-codigo-ia`)

**Veredicto: aceptar con cambios.** El código y las pruebas eran sólidos; los errores estaban en el reporte y en algunas pruebas.

| Hallazgo | Qué se hizo |
|---|---|
| Porcentajes mal calculados («cuesta 21 % más») | Corregido: el ganador cuesta 21 % y 35 % *menos* que las líneas base |
| Explicación de la sobrecobertura demasiado segura, palabras invertidas («sobrepromete») y falsa afirmación de que el orden no cambia | Reescrita como hipótesis; se agregó la cobertura mes a mes al notebook; se lista cómo cambia el orden hacia adelante |
| Segundo finalista contradecía las reglas aprobadas | Corregido: la regla da HistGradientBoosting cuantil 0,75 (vence a LightGBM de valor típico en 5 de 5 pliegues); LightGBM de valor típico queda como alternativa fuera de la regla |
| «El ganador es el más parejo entre regiones» solo estaba verificado contra 3 modelos | La revisión por región se calculó para los 8 modelos y el texto solo afirma lo que respaldan los datos |
| «Ventaja consistente» con más certeza de la debida | Se agregó el tamaño de la ventaja (0,36 %), la probabilidad de ganar 5 de 5 por azar (1/32) y la advertencia de que no hay variación por semilla |
| Conversión a «unidades de costo por mes» con bases frágiles | Eliminada; solo por pedido y por 1.000 pedidos |
| Comparación de MAE entre familias impresa en el notebook | Eliminada |
| Pruebas que pasarían aunque el código estuviera mal | Se fortalecieron: el margen es exactamente el percentil 75 (con errores conocidos), el modelo del margen no ve a los clientes de calibración, las predicciones regionales son fuera de muestra (un árbol que memoriza daría error cercano a 0 si no lo fueran), reproducibilidad y que no se modifican las entradas |
| Casos borde (clientes vacíos, longitud distinta, modelo inexistente, falta la región) | Ahora fallan con un mensaje claro |
| Duplicación de `predicción + margen` | Reutiliza `DeadlineModel.predict_with_deadline` |
| Aviso de «pocos clientes de calibración» oculto en el notebook | Ya no se silencia |
| Documentos desactualizados (119 pruebas, margen «de validación», contrato) | Corregidos |
| Control de la desventaja de datos de la familia A | Pendiente para la Fase 2 (limitación 2) |
