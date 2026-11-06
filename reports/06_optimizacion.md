# Etapa 6 · Optimización de hiperparámetros (Fase 2)

> Las cifras salen de `notebooks/06_optimizacion.ipynb` (ejecutado el 2026-11-05 con las semillas de abajo, solo con `train.parquet`) y de
> `reports/06_best_params.json`, `06_ensayos.csv`, `06_adelante_mensual.csv` y `06_regiones.csv`. Reglas: `especificaciones/06-optimizacion.md`
> (cerrada el 5 nov 2026) · Código: `src/olistbr/models/tuning.py`, `src/olistbr/data/split.py` · Pruebas: `tests/test_tuning.py`, `tests/test_split.py`.
> **La prueba final no se abrió.** Las unidades de costo son las del costo 3 a 1 (1 unidad = un día de sobrestimación).

## Resumen

- **Pasa a la validación final: LightGBM cuantil 0,75 con los parámetros optimizados.**
- **La regla escrita antes de mirar eligió los optimizados en los dos finalistas:** en los clientes reservados, todo el intervalo del 95 % de
  `costo(optimizado) − costo(partida)` quedó por debajo de cero.
  - LightGBM cuantil 0,75: **−0,068** por pedido, intervalo [−0,096 ; −0,041] (9,151 → 9,083).
  - HistGradientBoosting cuantil 0,75: **−0,099** por pedido, intervalo [−0,131 ; −0,070] (9,202 → 9,103).
- **Entre los dos (ya optimizados), LightGBM cuesta 0,020 menos por pedido**, intervalo [−0,038 ; −0,002]: todo bajo cero, así que gana por la
  cifra y no por la regla de desempate. La ventaja es **pequeña** (0,2 % del costo) y el límite superior está muy cerca de cero.
- **Cuidado al leer las cifras:** el mejor costo de la búsqueda (9,392) es **optimista** y no se reporta como resultado; el de la reserva (9,083) es
  más honesto pero **no** es comparable con el 9,367 del torneo (otros datos de ajuste y otros clientes) ni es la estimación final, que sale de la
  prueba en la etapa 7.
- **Informativo:** hacia adelante, el LightGBM optimizado cuesta **lo mismo** que su partida (8,853 contra 8,852) y HistGradientBoosting optimizado queda
  0,012 por debajo de él (8,840): el orden entre finalistas se invierte, en una sola partición. La cobertura mes a mes sigue desplazándose (60 % en marzo de 2018,
  92 % en agosto). Optimizar no corrige eso.

## Diseño de la búsqueda

- **Finalistas optimizados:** LightGBM cuantil 0,75 e HistGradientBoosting cuantil 0,75 (los dos que dio la regla del torneo, Q1). LightGBM de valor
  típico no entra: no lo respalda la regla. Como los dos son de la familia B (percentil 75 directo, sin margen), el control de datos entre familias pendiente del torneo no aplica aquí.
- **Métrica optimizada:** costo 3 a 1 del plazo recomendado, media de 5 pliegues por cliente (Q2). MAE, RMSE, R², sesgo, tarde y cobertura se registran, pero no deciden.
- **Variables:** las mismas 7 del torneo (sin marca de fin de año ni historial); el plazo prometido por Olist nunca entra.
- **El percentil 75 queda fijo** en todos los ensayos: si un ensayo intenta cambiarlo, el código falla (prueba `test_quantile_cannot_be_overridden`).
- **Espacio de búsqueda** (partida = parámetros del torneo):

| Finalista | Parámetro | Rango | Escala | Partida | Motivo |
|---|---|---|---|---:|---|
| LightGBM | `n_estimators` | 150 a 800 | entera | 300 | Con tasas bajas hacen falta más árboles; 800 acota el tiempo |
| LightGBM | `learning_rate` | 0,01 a 0,2 | logarítmica | 0,05 | Rango típico; se compensa con el número de árboles |
| LightGBM | `num_leaves` | 15 a 127 | entera | 31 | Complejidad de cada árbol |
| LightGBM | `min_child_samples` | 20 a 300 | entera | 20 | Un percentil necesita más pedidos por hoja que una media para ser estable |
| LightGBM | `subsample` | 0,6 a 1,0 | continua | 1,0 | Muestrear filas reduce el sobreajuste |
| LightGBM | `colsample_bytree` | 0,5 a 1,0 | continua | 1,0 | ≈ 100 columnas codificadas, muchas casi constantes |
| LightGBM | `reg_lambda` | 0,001 a 10 | logarítmica | 0 | Regularización L2 de las hojas |
| HistGB | `max_iter` | 150 a 800 | entera | 300 | Número de árboles, explícito (sin parada temprana en la búsqueda, Q5) |
| HistGB | `learning_rate` | 0,01 a 0,2 | logarítmica | 0,05 | Igual que LightGBM |
| HistGB | `max_leaf_nodes` | 15 a 127 | entera | 31 | Equivalente a `num_leaves` |
| HistGB | `min_samples_leaf` | 20 a 300 | entera | 20 | Equivalente a `min_child_samples` |
| HistGB | `l2_regularization` | 0,001 a 10 | logarítmica | 0 | Equivalente a `reg_lambda` |
| HistGB | `max_features` | 0,5 a 1,0 | continua | 1,0 | Proporción de columnas por división |

- **Presupuesto:** **20 ensayos por finalista**, fijado antes de empezar (Q3; la guía sugiere 12 a 20). Un ensayo fallido cuenta dentro del presupuesto. No falló ninguno.
  Tiempo real: 84 minutos en total; un ajuste de LightGBM tardó 3,2 s de media y uno de HistGradientBoosting 41,5 s (sin parada temprana y con más árboles que en el torneo).
- **Clientes reservados:** **25 % de los clientes** de entrenamiento (`split_by_group`, semilla 7): 19.225 pedidos de 18.607 clientes. La búsqueda usó el 75 % restante
  (57.699 pedidos de 55.818 clientes) con **5 pliegues nuevos por cliente** (semilla 42; ≈ 11.500 pedidos de validación por pliegue). Ningún cliente está en la búsqueda y en la reserva (`assert` en el notebook y prueba en `tests/test_split.py`).
- **Muestreador:** TPE de Optuna con semilla 42. El estudio está guardado en `data/processed/optuna_optimizacion.db` (no se versiona) y se puede reanudar.

## Seguimiento (tracking)

- **Dónde quedó registrado:** MLflow en **Databricks**, experimento `ontrack-optimizacion` (conexión del `.env`, que no se abrió ni se imprimió).
- **¿Está cada ensayo como una ejecución con parámetros, métrica y etiquetas?** ☒ sí ☐ no — una ejecución principal por finalista (`optuna-lightgbm-cuantil`,
  `optuna-histgradientboosting-cuantil`) con 20 ejecuciones hijas cada una (`lightgbm-cuantil-ensayo-00` … `-19`), cada hija con sus 6 o 7 hiperparámetros,
  `costo_media`, `costo_desv`, las demás métricas, el tiempo de ajuste y las etiquetas `etapa=06_optimizacion`, `modelo` y `metrica_optimizada=costo`.
  Además, dos ejecuciones `comprobacion-…` con el resultado de la comprobación honesta. Total: 44 ejecuciones en la primera corrida; al volver a
  ejecutar el notebook (para recuperar sus salidas, reutilizando los 40 ensayos guardados) se registraron otras 2 `comprobacion-…` con las mismas cifras: **46 en total**.
- **Mejor ejecución de cada búsqueda (optimista):** LightGBM, ensayo 15, costo 9,392 ± 0,103; HistGradientBoosting, ensayo 17, costo 9,411 ± 0,100.
- **Sin datos personales:** MLflow imprimía el enlace del workspace y el correo de la cuenta en las salidas del notebook; se apagó (`MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT`) y el notebook muestra solo el nombre del experimento.

## Resultados de la búsqueda

| Finalista | Mejor costo (optimista) | ± entre pliegues | Mediana de los 20 | Peor de los 20 | Ensayos a menos de 0,02 del mejor |
|---|---:|---:|---:|---:|---:|
| LightGBM cuantil 0,75 | 9,392 | 0,103 | 9,447 | 9,592 | 5 |
| HistGradientBoosting cuantil 0,75 | 9,411 | 0,100 | 9,442 | 9,522 | 7 |

- **Los mejores ensayos casi no se distinguen entre sí:** las diferencias entre los primeros puestos (≈ 0,02) son mucho menores que la variación entre pliegues (≈ 0,10).
  Por eso el mejor de la búsqueda es optimista y la decisión se toma en la reserva.
- **Parámetros optimizados:**
  - LightGBM: 418 árboles, tasa 0,049, 70 hojas, mínimo 73 pedidos por hoja, `subsample` 0,95, `colsample_bytree` 0,65, `reg_lambda` 4,43.
    Frente a la partida: árboles más grandes pero con hojas más pobladas y más regularización, y cada árbol mira el 65 % de las columnas.
  - HistGradientBoosting: 530 árboles, tasa 0,025, 110 hojas, mínimo 129 pedidos por hoja, `l2_regularization` 0,014, `max_features` 0,53.
- No se midió qué parámetro influyó más (no hay un análisis de importancia de hiperparámetros); con 20 ensayos sería poco confiable.

## Comprobación honesta (entidades reservadas)

> Cada modelo se ajustó con **toda** la búsqueda (75 % de los clientes) y se evaluó en la reserva (25 % de los clientes, que la búsqueda no vio).
> Intervalo del 95 % con 1.000 sorteos de **clientes completos** (semilla 42), no de pedidos: los pedidos de un mismo cliente no son independientes.

| Modelo | Costo con parámetros de partida | Costo con parámetros optimizados | Diferencia (optimizado − partida) | Intervalo 95 % |
|---|---:|---:|---:|---|
| LightGBM cuantil 0,75 | 9,151 | 9,083 | −0,068 | [−0,096 ; −0,041] |
| HistGradientBoosting cuantil 0,75 | 9,202 | 9,103 | −0,099 | [−0,131 ; −0,070] |

| Modelo | Cobertura partida | Cobertura optimizado | Árboles que usó la partida | Árboles del optimizado |
|---|---:|---:|---:|---:|
| LightGBM cuantil 0,75 | 74,7 % | 73,9 % | 300 | 418 |
| HistGradientBoosting cuantil 0,75 | 75,1 % | 73,9 % | **163** (de 300) | 530 |

**Regla de decisión escrita antes de mirar:** «Uso los parámetros optimizados **solo si todo el intervalo está por debajo de cero**. Si el intervalo cruza el
cero (o queda por encima), la mejora no se distingue del ruido y uso los de partida.» Está en código probado (`decide_params`).

**Resultado de la regla** (¿todo el intervalo por debajo de cero?): **sí en los dos finalistas** → se usan los parámetros optimizados.

- **En la partida del torneo, HistGradientBoosting tenía la parada temprana activada sin saberlo:** con más de 10.000 filas se activa sola, aparta un 10 % de
  **pedidos** (no de clientes) y aquí paró en 163 de 300 árboles. Parte de su mejora puede venir solo de apagarla y usar más árboles.
- **La cobertura de los optimizados bajó un punto** (≈ 75 % → 74 %): el plazo queda un poco más corto en promedio. El costo baja igual; una explicación posible
  (no medida aquí) es que los plazos optimizados se ajustan mejor a cada pedido, con menos días de sobra, y eso compensa el punto adicional de pedidos tarde.
  Vale la pena vigilar la cobertura en la etapa 7.

**Elección entre los dos finalistas** (con los parámetros optimizados de cada uno, en la reserva): `costo(LightGBM) − costo(HistGradientBoosting)` = **−0,020** por pedido,
intervalo [−0,038 ; −0,002]. Todo el intervalo está por debajo de cero → **gana LightGBM cuantil 0,75** (`choose_finalist`). Si el intervalo hubiera cruzado el cero,
también habría ganado LightGBM por la regla de desempate (Q7). Además fue unas 13 veces más rápido de ajustar **en promedio durante la búsqueda** (3,2 s contra 41,5 s por ajuste; no es el tiempo del modelo final).

## Revisiones informativas (no cambian la decisión)

**Hacia adelante con clientes reservados** (entrenar con las compras de la búsqueda hasta el 2018-02-28 **ya entregadas antes del 1 de marzo**: 31.988 pedidos;
validar con las compras de clientes reservados del 2018-03-01 al 2018-08-31: 7.937 pedidos). Una sola partición, sin ruido medido. Cifras de `reports/06_adelante.csv`.

| Modelo | Costo hacia adelante | Cobertura | MAE (del percentil 75) |
|---|---:|---:|---:|
| LightGBM cuantil 0,75 · partida | 8,852 | 80,5 % | 6,246 |
| LightGBM cuantil 0,75 · **optimizado** | 8,853 | 79,7 % | 6,180 |
| HistGradientBoosting cuantil 0,75 · partida | 8,851 | 81,2 % | 6,280 |
| HistGradientBoosting cuantil 0,75 · optimizado | **8,840** | 80,1 % | 6,181 |

- **El LightGBM optimizado no mejora hacia adelante:** cuesta 0,0002 más que su partida (prácticamente lo mismo). Frente a la mejora de −0,068 en la reserva, la
  ganancia de optimizar puede no trasladarse a meses posteriores; con una sola partición no se puede saber si es ruido.
- **El orden entre finalistas se invierte:** hacia adelante, HistGradientBoosting optimizado cuesta 0,012 menos que LightGBM optimizado, al revés que en la reserva
  (donde LightGBM ganó por 0,020 con un intervalo [−0,038 ; −0,002]). La decisión la toma la reserva (regla escrita); esto es una advertencia sobre lo ajustada que fue la elección.
- **Los cuatro modelos quedan a menos de 0,02 de distancia** hacia adelante, lo que es compatible con que el cambio de ritmo de las entregas pese más que
  los hiperparámetros (no se midió; es una sola partición).
- _Antes de la revisión del `auditor-de-fuga`, el entrenamiento hacia adelante incluía compras de fines de febrero entregadas en marzo (34.189 pedidos); las cifras de
  esa versión (por ejemplo, LightGBM optimizado 8,964 contra 8,958 de su partida) quedaron reemplazadas por estas._

**Cobertura mes a mes** (mismo corte; esperado ≈ 75 %; `reports/06_adelante_mensual.csv`):

| Mes de compra | Pedidos | Mediana de días reales | LightGBM partida | LightGBM optimizado | HistGB partida | HistGB optimizado |
|---|---:|---:|---:|---:|---:|---:|
| 2018-03 | 1.472 | 12,9 | 61,2 % | 60,3 % | 62,1 % | 60,9 % |
| 2018-04 | 1.373 | 9,3 | 80,6 % | 79,2 % | 81,0 % | 79,8 % |
| 2018-05 | 1.358 | 9,5 | 73,6 % | 73,1 % | 74,6 % | 73,1 % |
| 2018-06 | 1.237 | 8,1 | 90,5 % | 89,7 % | 91,3 % | 90,3 % |
| 2018-07 | 1.217 | 7,4 | 89,2 % | 88,5 % | 89,8 % | 89,1 % |
| 2018-08 | 1.280 | 7,0 | 92,3 % | 91,6 % | 92,2 % | 91,7 % |

Es el mismo patrón del torneo: marzo queda corto y de junio en adelante el plazo sobra. Optimizar no lo corrige; queda para la etapa 7 (recalibrar con datos recientes y vigilar la cobertura).

**Por región del cliente** (modelo elegido, en la reserva; esperado ≈ 25 % de pedidos tarde):

| Región | Pedidos | MAE (del percentil 75) | % tarde |
|---|---:|---:|---:|
| Centro-Oeste | 1.128 | 5,99 | 27,6 % |
| Nordeste | 1.805 | 8,31 | 24,9 % |
| Norte | 380 | 8,73 | 29,2 % |
| Sudeste | 13.122 | 5,06 | 26,3 % |
| Sul | 2.790 | 6,04 | 24,8 % |

- El **Norte** queda por encima de lo esperado (29,2 %), pero con 380 pedidos la desviación de muestreo de ese porcentaje es de ≈ 2,2 puntos (el intervalo del 95 % sería ≈ ± 4,4): no se puede afirmar que el Norte esté peor. El Centro-Oeste también sube (27,6 %, 1.128 pedidos, ≈ 1,3 puntos de desviación).
- El error absoluto es mayor en el Norte y el Nordeste, como en el torneo (principio VIII): son las regiones a revisar en la validación final.

## Decisión

- **Parámetros que se usarán a partir de ahora:** los **optimizados** de **LightGBM cuantil 0,75** (418 árboles, tasa 0,049, 70 hojas, mínimo 73 pedidos por hoja,
  `subsample` 0,95, `colsample_bytree` 0,65, `reg_lambda` 4,43; percentil 0,75 fijo). Motivo: la regla escrita antes de mirar (todo el intervalo bajo cero en la
  reserva) y la comparación entre finalistas (también bajo cero). Detalle completo en `reports/06_best_params.json`.
- **Lo que no pudimos concluir:**
  1. Si la mejora se mantiene en el tiempo: hacia adelante el LightGBM optimizado cuesta lo mismo que su partida (+0,0002), y HistGradientBoosting optimizado
     queda por debajo de él (el orden entre finalistas se invierte), en una sola partición.
  2. Cuánto de la mejora de HistGradientBoosting viene solo de apagar la parada temprana.
  3. Si la ventaja de LightGBM sobre HistGradientBoosting (0,2 %) se mantendría con otra semilla de reserva: el límite superior del intervalo está a 0,002 de cero.

## Limitaciones

1. **La reserva se usó tres veces:** dos comparaciones de partida contra optimizado, una entre finalistas (las tres deciden) y la prueba hacia adelante (informativa).
   Sigue siendo validación, pero el costo del elegido en la reserva (9,083) es algo optimista, y las tres comparaciones con intervalos del 95 % no tienen
   corrección por comparaciones múltiples: la elección entre finalistas, con el límite superior en −0,002, es la más frágil.
2. **Costos no comparables con el torneo:** la búsqueda ajusta con ≈ 46.000 pedidos por pliegue (≈ 61.500 en el torneo) y la reserva es otro grupo de clientes.
3. **Lista de vendedores frecuentes:** viene de la etapa 3 y se contó con todo el entrenamiento, incluida la reserva. Solo usa conteos de ítems (no el objetivo); riesgo bajo.
4. **20 ensayos por finalista:** la búsqueda no garantiza haber encontrado el mejor punto del espacio; solo que el encontrado es mejor que la partida en la reserva.
5. **Calibración en el tiempo:** la cobertura cambia de 60 % a 92 % según el mes (heredado del torneo); se trata en la etapa 7.
6. **Reproducibilidad: comprobada.** Una segunda ejecución completa en modo local (sin Databricks), con el estudio en otra carpeta, dio exactamente los mismos
   40 ensayos, parámetros, comprobación honesta, elección y tablas por región y por mes (los tiempos y los códigos de MLflow no se comparan).
7. **La prueba hacia adelante no es limpia en el tiempo para los hiperparámetros:** separa clientes, pero los parámetros se eligieron con pliegues de todos los
   meses de la búsqueda (incluidos los de marzo a agosto de 2018). Por eso es solo informativa.

## Revisiones

### Auditoría de fuga (`auditor-de-fuga`)

**Veredicto: fuga leve, ninguna grave; ninguna invalida la decisión.** Confirmó que la reserva no entra en la búsqueda (ni en los pliegues ni en el preprocesador),
que los pliegues son por cliente, que el intervalo sortea clientes completos, que todo lo que aprende está en el `Pipeline`, que `test.parquet` no se lee y que el
plazo prometido, la marca de fin de año y el historial no entran.

| Hallazgo | Qué se hizo |
|---|---|
| Prueba hacia adelante: el entrenamiento traía compras de fines de febrero entregadas en marzo (su tiempo de entrega no se conocía al corte) | **Corregido:** `reserved_forward_split` solo usa pedidos ya entregados antes del 1 de marzo (compra + días de entrega); prueba nueva. Las cifras hacia adelante se recalcularon |
| Lista de vendedores frecuentes calculada con todo el entrenamiento, incluida la reserva | Se mantiene como limitación (3): son conteos de ítems, sin el objetivo; riesgo bajo |
| La reserva se usa tres veces, no dos | Corregido en «Limitaciones» (1), con la nota de comparaciones múltiples |
| No había prueba de que el plazo prometido no llegue al modelo de esta etapa | Prueba nueva: ni el plazo prometido, ni el cliente, ni el pedido entran al preprocesador |

### Revisión de código (`revisor-de-codigo-ia`)

**Veredicto: aceptar con cambios.** El código cumple el contrato y las reglas; las cifras contrastables coinciden con los archivos; los módulos de etapas anteriores no cambiaron.

| Hallazgo | Qué se hizo |
|---|---|
| **Grave (no es de código):** el `.gitignore` tenía comentadas las líneas que excluyen `data/processed/`, así que un `git add .` subiría la prueba cerrada | Avisado al grupo para que lo restaure (ver bitácora) |
| Cifras del informe sin archivo que las respalde (comprobación, hacia adelante, conteos, duración) | El notebook ahora guarda `06_comprobacion.csv` y `06_adelante.csv`, y el JSON anota los pedidos hacia adelante y los minutos |
| Hacia adelante, el orden entre finalistas se invierte | Agregado al texto y a «Lo que no pudimos concluir» |
| «13 veces más rápido» y «pesa mucho más» iban más allá de los datos | Reescritos: tiempo medio de la búsqueda; «compatible con» |
| `catch=(ValueError,)` escondía cualquier error como ensayo fallido | Excepción propia `NonFiniteCost`: solo un costo no finito marca un ensayo como fallido; otro error detiene la búsqueda (prueba nueva) |
| Un ensayo fallido no quedaba en MLflow | Ahora se registra con la etiqueta `estado=FAIL` (prueba nueva) |
| Entradas vacías o con valores no finitos en el intervalo y en las reglas | Ahora fallan con un mensaje claro (pruebas nuevas) |
| Faltaba garantizar que la comprobación mida igual que el torneo | Prueba nueva: mismo costo que `evaluate_candidate_holdout` |
| Prueba circular de los parámetros de partida | Prueba nueva con los números del torneo escritos a mano |
| Otros parámetros fijos (`random_state`, `n_jobs`, …) podían chocar con un ensayo | Ahora también se rechazan con un mensaje claro |
| Notebook: los días reales de la reserva salían de la última vuelta de un bucle | Ahora se definen explícitamente y se verifica que son los mismos para los dos finalistas |

**Preguntas que dejó el revisor para quien no escribió el código:** (1) ¿por qué no se reporta 9,392 y qué cifra se usa en su lugar? (2) si el intervalo hubiera
sido [−0,05 ; +0,01], ¿qué parámetros se usarían, y por qué se sortean clientes y no pedidos? (3) ¿qué tan firme es la elección de LightGBM y qué parte de la mejora
de HistGradientBoosting puede venir solo de apagar la parada temprana?

## Para la clase siguiente

- **Modelo que pasa a la validación final (etapa 7):** LightGBM cuantil 0,75 con los parámetros optimizados, ajustado con todo el conjunto de entrenamiento.
  La prueba (`test.parquet`) se abre **una sola vez** ahí.
- Llevar a la especificación de la etapa 7: recalibración con datos recientes y vigilancia de la cobertura; revisión por región (Norte y Nordeste); la marca de fin de año (F6) y el historial por vendedor (F8) como sensibilidades; reportar también el MAE con los 64 pedidos extremos.
