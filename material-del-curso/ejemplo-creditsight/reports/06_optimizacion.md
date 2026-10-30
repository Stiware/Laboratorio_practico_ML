# 06 · Optimización con seguimiento (tracking)

**Qué se hizo.** Búsqueda bayesiana de hiperparámetros (Optuna, muestreador TPE) sobre los **dos finalistas del torneo**,
`lightgbm` y `random forest`, optimizando el **costo esperado** con la matriz de costos. Cada ensayo quedó registrado como
una ejecución anidada en MLflow, con el tablero completo de métricas (negocio y técnicas, con su desviación entre pliegues).

**Dónde corrió.** En un *Git folder* de Databricks (el repositorio clonado), con el notebook `material-del-curso/ejemplo-creditsight/notebooks/06_optimizacion.ipynb`;
el seguimiento quedó en el experimento `creditsight-optimizacion` del workspace. El cómputo local no alcanzaba: la etapa se
había cortado dos veces por falta de memoria.

## Cómo se buscó

- **Datos:** el 75 % de los clientes del entrenamiento (la búsqueda trabaja con ellos); el otro 25 % queda **reservado** y la
  búsqueda nunca lo ve. La prueba final sigue cerrada. La validación cruzada (3 pliegues) es **por cliente**.
- **Presupuesto acotado a propósito:** 15 ensayos para `lightgbm` y 10 para `random forest` (el bosque cuesta unas 10 veces más por
  ajuste: unos 24 s por pliegue frente a unos 2,4 s). Es una decisión de cómputo, no de método.
- **Espacio de búsqueda con conocimiento del algoritmo:** tasa de aprendizaje en escala logarítmica, número de hojas, muestra
  mínima por hoja, submuestreo y regularización para `lightgbm`; profundidad, hoja mínima y fracción de variables por división
  para el bosque (200 árboles fijos: se afina su forma, no su cantidad).

## Resultados de la búsqueda (validación cruzada, 25 ensayos distintos)

| Modelo | Ensayos | Mejor costo | Peor costo | Mejor AUC | Mejor ensayo |
|---|---|---|---|---|---|
| `lightgbm` | 15 | **0,4856** | 0,5488 | 0,8538 | n.º 10 |
| `random forest` | 10 | **0,4795** | 0,5008 | 0,8535 | n.º 9 |

Mejores configuraciones (guardadas en `06_mejores_parametros.json`; todos los ensayos, en `06_optimizacion_tabla.csv`):

- `lightgbm`: `n_estimators` 185, `learning_rate` 0,0265, `num_leaves` 29, `min_child_samples` 129, `subsample` 0,956,
  `colsample_bytree` 0,696, `reg_lambda` 0,670.
- `random forest`: `max_depth` 15, `min_samples_leaf` 12, `max_features` 0,237.

## Qué se observó

- **El rango de resultados es angosto en el bosque** (costo 0,4795 a 0,5008) **y ancho en `lightgbm`** (0,4856 a 0,5488):
  `lightgbm` es más sensible a sus parámetros, y un mal ensayo sale caro.
- **Costo y AUC se mueven casi juntos en esta búsqueda** (correlación de −0,97 entre los 25 ensayos): los ensayos con mejor AUC
  fueron también los de menor costo. No es una ley: aquí ocurre porque se varían solo parámetros de forma del mismo modelo.
- **El recall de `Good` sigue siendo muy bajo** (0 a 19 % según el ensayo) y el de `Poor` alto (76 a 82 %): la regla de menor
  costo esperado es prudente por diseño, porque aprobar a un `Poor` cuesta 10 y rechazar a un `Good`, 2. Ninguna configuración
  cambió eso.
- **La mejor puntuación de una búsqueda es optimista** (se eligió el máximo entre varios ensayos con ruido). Por eso la mejora
  **no se da por buena** hasta comprobarla en clientes reservados: eso se hace en la sección 1 del notebook 07, con intervalo
  remuestreando clientes. Si la mejora no se distingue del ruido, se queda la configuración de partida.

## Una nota honesta sobre el registro

El experimento tiene **86 ejecuciones y no 25**: el notebook se lanzó varias veces (4 corridas de `lightgbm` y 2 de `random forest`)
y cada lanzamiento volvió a registrar sus ensayos. Los ensayos repetidos dan **exactamente los mismos resultados**
(misma semilla), así que no cambian ninguna conclusión; para la tabla se conservó **una sola copia** de cada ensayo
(la primera registrada). Lección para el equipo: la búsqueda de Optuna es reanudable (`storage=` en `tune`) y conviene
usarla cuando el cómputo es de otro, porque una ejecución que se reintenta duplica el tracking.
