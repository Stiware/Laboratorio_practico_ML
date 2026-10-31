# Etapa 4 · Variables (feature engineering)

> Las cifras de las tablas y de cada decisión salen de `notebooks/04_features.ipynb` (ejecutado con la semilla 42, solo con `train.parquet`)
> y están en `reports/04_features_tabla.csv`. Las cifras de la exploración previa (correlaciones, medianas por año) vienen de
> `especificaciones/04-features.md`; las de la sección «La referencia usa poco al vendedor» se comprobaron aparte con `train.parquet`
> durante la revisión. Reglas: `especificaciones/04-features.md` · Código: `src/olistbr/features/`, `src/olistbr/models/`
> · Pruebas: `tests/test_features.py`, `test_split.py`, `test_evaluation.py`, `test_models.py`.

## Resumen

Se midieron 9 ideas de variables contra una referencia (las variables base) con el mismo modelo (regresión lineal), 5 pliegues por
cliente y una prueba hacia adelante (entrenar hasta 2018-02-28, validar 2018-03-01 a 2018-08-31). **Aplicando la regla escrita antes de
medir se conservan 8 ideas y se descarta 1** (F8 historial vendedor / categoría / ruta).

- Juntas, las conservadas bajan el error medio de 5,211 a 5,033 días: **0,178 días menos (3,4 %)**.
  Es una **estimación de validación, no de la prueba final**, y algo optimista porque las ideas se eligieron mirando los mismos pliegues.
- Para dar contexto: predecir siempre la mediana da 6,036 días; la referencia (variables base) ya baja a 5,211 (0,825 días menos), de modo que las ideas de esta etapa agregan una mejora pequeña sobre una base que ya aporta bastante.
- **Mejoras claras** (mejora / ruido ≥ 5 y mejora ≥ 0,05 días): F1, F2, F6, F7 (F6 con reserva: ver su sección). **Pequeñas** (mejora / ruido entre 2 y 5): F3, F5.
  **Prácticamente indistinguibles de cero**: F9, F10 (cumplen la regla por muy poco; ver «Para decidir en grupo»).
  Ojo: el «ruido» es la desviación de solo 5 diferencias que comparten gran parte del entrenamiento; **no es una prueba de significancia**.
- El plazo prometido por Olist se mide solo como referencia y **no se usa por diseño** (la app fija la fecha, decisión Q1). Su mejora
  (0,142 días) es menor que la del conjunto en la validación por cliente, pero hacia adelante casi empatan
  (0,366 contra 0,373 días). No se midió el plazo sumado al conjunto.
- Las mejoras son pequeñas en términos de negocio (el conjunto baja el error unas 4,3 horas por pedido). Con una regresión
  lineal el techo parece modesto; el torneo de modelos (etapa 5) probará si modelos no lineales lo aprovechan mejor.

## Cómo se midió

- **Datos:** `data/processed/train.parquet` (76.924 pedidos, 74.425 clientes). La prueba final no se abrió.
- **Pliegues:** `GroupKFold` por `customer_unique_id`, 5 pliegues, semilla 42, creados una vez; ningún cliente en los dos lados de un pliegue.
- **Hacia adelante:** entrenamiento hasta 2018-02-28 (44.850 pedidos, sin los clientes que aparecen en la validación) y validación del 2018-03-01 al 2018-08-31 (31.447 pedidos). Es **una sola partición**, sin ruido medido.
- **Métricas:** error absoluto medio en días (principal) y costo medio 3 a 1 (control).
- **Mejora / ruido:** para cada pliegue, error de la referencia menos error de la idea; la mejora es la media y el ruido la desviación estándar de esas 5 diferencias.
- **Regla de decisión (escrita antes de medir):** conservar solo si se conoce al predecir, `mejora / ruido > 1`, el costo 3 a 1 también mejora y la mejora no cambia de signo hacia adelante. Se aplica literalmente: una mejora hacia adelante de +0,0009 días pasa, una de −0,085 no.

### La referencia usa poco al vendedor

El preprocesador agrupa como «poco frecuentes» las categorías con menos del 1 % de las filas. Con `train.parquet`, **solo 11 de los 1.165 vendedores**
alcanzan ese 1 % (cubren el 21 % de los pedidos), así que la referencia casi no conoce al vendedor. El preprocesador tiene 66 columnas en la referencia
y 103 con todos los grupos. Esto importa para leer F8: su ganancia en validación por cliente puede deberse sobre todo a «tener información del vendedor» y no
a «tener su historial».

## Tabla de ideas medidas

| Idea | Error medio (días) | Mejora (días) | Mejora / ruido | Mejora del costo 3 a 1 | Mejora hacia adelante (días) | Error hacia adelante (días) | Decisión |
|---|---:|---:|---:|---:|---:|---:|---|
| Referencia (variables base) | 5,211 ± 0,017 | — | — | — | — | 5,676 | referencia |
| F1 logaritmos | 5,149 ± 0,010 | 0,062 | 7,33 | 0,124 | 0,047 | 5,629 | **conservar** |
| F2 mismo estado | 5,151 ± 0,017 | 0,060 | 14,24 | 0,119 | 0,095 | 5,580 | **conservar** |
| F3 regiones | 5,208 ± 0,017 | 0,003 | 2,39 | 0,007 | 0,006 | 5,670 | **conservar** |
| F5 día de la semana | 5,196 ± 0,019 | 0,015 | 4,47 | 0,031 | 0,018 | 5,658 | **conservar** |
| F6 temporada alta de fin de año | 5,142 ± 0,011 | 0,069 | 6,73 | 0,138 | — | — | **conservar** |
| F7 distancia × mismo estado | 5,120 ± 0,012 | 0,091 | 15,01 | 0,182 | 0,140 | 5,536 | **conservar** |
| F8 historial vendedor / categoría / ruta | 4,972 ± 0,025 | 0,239 | 8,85 | 0,488 | -0,085 | 5,761 | **descartar** |
| F9 razones de flete y densidad | 5,211 ± 0,017 | 0,0004 | 1,02 | 0,0009 | 0,0009 | 5,675 | **conservar** |
| F10 varios vendedores | 5,206 ± 0,020 | 0,005 | 1,19 | 0,011 | 0,012 | 5,664 | **conservar** |
| F4 plazo prometido por Olist (referencia) | 5,069 ± 0,011 | 0,142 | 13,33 | 0,284 | 0,366 | 5,310 | referencia (no usable) |
| Línea base trivial (siempre la mediana) | 6,036 ± 0,025 | -0,825 | -41,16 | -3,911 | -0,331 | 6,007 | línea base |
| Conjunto conservado | 5,033 ± 0,012 | 0,178 | 12,60 | 0,357 | 0,373 | 5,303 | comprobación |
| Conjunto conservado sin la marca de fin de año (sensibilidad) | 5,096 ± 0,017 | 0,115 | 12,55 | 0,230 | 0,133 | 5,543 | sensibilidad |

`±` es la desviación del error entre los 5 pliegues. «Mejora» = referencia − idea (positiva = ayuda). «Error hacia adelante» es el error absoluto de la prueba hacia adelante.

## Decisión por idea

- **F1 logaritmos → conservar.** Mejora 0,062 días, mejora / ruido 7,33, hacia adelante 0,047 días. Explicación plausible, no medida por separado: en un modelo lineal los logaritmos de distancia, flete, precio, peso y volumen pueden capturar un efecto creciente pero curvo (el grupo se midió junto, sin aislar cada logaritmo).
- **F2 mismo estado → conservar.** Mejora 0,060 días, mejora / ruido 14,24, hacia adelante 0,095 días. Consistente con el EDA: dentro del mismo estado el envío tarda menos.
- **F3 regiones → conservar, aporte pequeño.** Mejora de 0,003 días (unos 5 minutos), mejora / ruido 2,39.
- **F5 día de la semana → conservar.** Mejora 0,015 días, mejora / ruido 4,47, hacia adelante 0,018 días. El patrón de jueves a sábado se repite entre 2017 y 2018 (correlación 0,72 entre años, de la exploración previa).
- **F6 temporada alta de fin de año → conservar, con reserva (decide el grupo).** Mejora 0,069 días, mejora / ruido 6,73, **no evaluable hacia adelante**. Aunque se llama «temporada alta», la marca vale 1 solo entre el 20 de noviembre y el 31 de diciembre de **2017**: es un único periodo, mezclado con la tendencia de la operación de esas semanas, y la validación por cliente mezcla fechas, así que el modelo puede estar aprendiendo «noviembre-diciembre de 2017 fue lento». Es el mismo mecanismo que probablemente afecta a F8, y la regla no pudo protegerla porque la prueba hacia adelante no contiene ese periodo. Sin esta marca, el conjunto conservado mejora 0,115 días en vez de 0,178 (la marca aporta 0,063 días al conjunto en la validación por cliente). **Hacia adelante la diferencia es aún mayor:** el conjunto mejora 0,133 días sin la marca y 0,373 con ella. La marca no se evalúa directamente hacia adelante (vale 0 en toda la validación), pero entrenar con ella cambia los coeficientes de las demás variables. *Hipótesis sin probar:* la marca absorbe un periodo anómalo del entrenamiento (noviembre-diciembre de 2017) y eso mejora las predicciones de marzo a agosto de 2018; no demuestra que la marca sirva en un fin de año futuro.
- **F7 distancia × mismo estado → conservar.** La mayor mejora entre las ideas conservadas: 0,091 días, mejora / ruido 15,01, hacia adelante 0,140 días. Incluye los logaritmos de distancia y «mismo estado», así que solapa con F1 y F2 (no se suman).
- **F8 historial vendedor / categoría / ruta → descartar.** En la validación por cliente es la que más mejora (0,239 días, mejora / ruido 8,85, costo 0,488), pero **hacia adelante empeora (-0,085 días)** y la regla la descarta. Ese valor es de una sola partición, sin ruido medido. *No sabemos la causa; hipótesis sin probar:* (a) los promedios históricos de vendedores y rutas cambian con el tiempo y la validación por cliente mezcla fechas; (b) 2018 es más rápido que 2017 en general, y esos promedios quedan altos; (c) el entrenamiento hacia adelante es más chico y no incluye a los clientes que repiten; (d) la referencia casi no usa al vendedor, así que parte de la ganancia sería solo «tener información del vendedor». Separarlas pide una validación temporal por bloques (etapa 5).
- **F9 razones de flete y densidad → conservar, aporte despreciable.** Mejora 0,0004 días (menos de un minuto) y mejora / ruido 1,02: supera el umbral por apenas un 2 %, un margen que el redondeo podría mover.
- **F10 varios vendedores → conservar, aporte despreciable.** Mejora 0,005 días (unos 8 minutos), mejora / ruido 1,19.
- **F4 plazo prometido por Olist → referencia, nunca entra al modelo.** Mejora 0,142 días (mejora / ruido 13,33), hacia adelante 0,366 días. Se mide para saber cuánto se pierde sin él. La razón para no usarlo es de diseño (la app recomienda la fecha), no un resultado de esta tabla. Que hacia adelante aporte mucho más que en los pliegues podría reflejar la tendencia de 2018 (no comprobado).

### Descartadas antes de medir (con su evidencia)

- `purchase_hour`: correlación 0,004 con el objetivo en entrenamiento; sin hipótesis de negocio.
- `purchase_month` (mes suelto): el patrón mensual de enero a agosto casi no se repite entre 2017 y 2018 (correlación 0,14); mezcla temporada con tendencia entre años.
- `approval_delay_hours`: se conoce horas después de comprar (la mediana es 0,34 h pero el 10 % peor tarda más de 34 h).

## Conjuntos conservados (comprobaciones)

- **Conjunto conservado** (logs + same_state + regions + calendar + peak_season + distance_interaction + ratios + multi_seller): error medio 5,033 ± 0,012 días; mejora 0,178 días (mejora / ruido 12,60), costo 3 a 1 0,357 menos y 0,373 días de mejora hacia adelante (error hacia adelante 5,303 contra 5,676 de la referencia).
- **Sin la marca de fin de año (sensibilidad):** error medio 5,096; mejora 0,115 días; hacia adelante 0,133 días.
- La idea descartada F8 mejora más que el conjunto en la validación por cliente (0,239 días), pero no hacia adelante.

## Limitaciones

1. **Tendencia entre años.** En 2018 las entregas son más rápidas que en 2017 (mediana de julio: 7,5 días contra 10,1). Ninguna variable puede anticipar cambios de la operación posteriores a agosto de 2018; el modelo los desconoce.
2. **Una sola temporada de fin de año** (2017): F6 no se puede validar hacia adelante ni confirmar que se repita.
3. **La validación por cliente mezcla fechas**, lo que puede inflar ideas que capturan comportamientos de un periodo concreto (F6 y, posiblemente, F8).
4. **Historial (F8):** `TargetEncoder` usa un separador interno que no agrupa por cliente. No contamina la validación (los pliegues exteriores sí son por cliente), pero dentro del entrenamiento pedidos del mismo cliente pueden informarse entre sí.
5. **Lista de vendedores frecuentes:** se calculó en la etapa 3 con todo `train.parquet` (solo conteos de ítems, no el objetivo), por lo que incluye las filas de validación de cada pliegue. El efecto probable es pequeño; no se midió.
6. **Estimación optimista:** las ideas se eligieron con los mismos pliegues con los que se mide el conjunto, y el entrenamiento hacia adelante es más chico y sin los clientes que repiten.
7. **Solapamiento entre ideas:** F7 incluye los logaritmos de distancia y «mismo estado», por lo que no se suma a F1 ni a F2. Cada idea se midió sola, no en todas las combinaciones.

## Para decidir en grupo

1. **F6 (marca de fin de año).** La regla escrita la conserva, pero sin validación hacia adelante y apoyada en una sola temporada. Opciones: (a) mantenerla «con reserva» (el conjunto mejora 0,178 días en validación por cliente y 0,373 hacia adelante); (b) dejarla fuera hasta tener otra temporada (0,115 y 0,133 días). Si se mantiene, conviene entender por qué ayuda hacia adelante (¿absorbe el periodo anómalo de 2017?) antes de confiar en ella. Cambiar la regla para que una idea sin prueba hacia adelante quede «pendiente» sería una decisión nueva, a escribir antes de volver a medir.
2. **F3, F9 y F10.** Cumplen la regla pero casi no aportan (0,003, 0,0004 y 0,005 días). No piden columnas extra a la app, pero añaden complejidad. Se puede proponer **para la etapa 5** un umbral mínimo de mejora en días además de `mejora / ruido > 1`.
3. **F8.** Retomarla en la etapa 5 con una validación que respete las fechas (por bloques) y, quizá, una referencia que use más al vendedor.

## Revisiones

### Auditoría de fuga (`auditor-de-fuga`)

**Veredicto: sin fuga grave.** Las variables usan solo lo que se conoce al comprar, `add_features` es de verdad sin estado, todo lo que aprende está dentro del `Pipeline`, ningún cliente cae en los dos lados de un pliegue, no se abre la prueba final y el plazo prometido no entra al modelo final.
Matices que se incorporaron: F6 se conserva sin validación temporal (ahora con reserva y como decisión del grupo, con la cifra de sensibilidad); la lista de vendedores frecuentes (limitación 5); `TargetEncoder` (limitación 4); el 5,033 es una estimación de validación y el plazo casi empata con el conjunto hacia adelante. Se agregó una prueba de que `promised_days` no llega al modelo final.

### Revisión de código (`revisor-de-codigo-ia`)

**Veredicto: aceptar con cambios.** Las cifras del reporte coinciden con la tabla. Cambios aplicados:

| # | Hallazgo | Qué se hizo |
|---|---|---|
| 1 | F6 con trato más blando que F8 | Reescrita en este reporte; se midió el conjunto sin la marca (sensibilidad), lo que mostró que hacia adelante la marca pesa mucho en el conjunto; queda como decisión del grupo |
| 2 | Explicación de F8 dicha con demasiada seguridad | Se quitó «más probable» y se listan cuatro hipótesis sin probar |
| 3 | La referencia casi no usa al vendedor | Documentado arriba; se corrigió el plan (66 columnas, no 1.200) |
| 4 | «Claras frente al ruido» para mejoras que no lo son | Reescrito con una clasificación descriptiva; se aclara que el ruido no es significancia |
| 5 | «El modelo no lo necesita» sobre el plazo prometido | Reescrito: la razón es de diseño; hacia adelante casi empatan |
| 6 | Faltaba el error absoluto hacia adelante | Nueva columna `adelante_mae` en la tabla |
| 7 | Explicaciones causales sin prueba | Marcadas como plausibles/no medidas |
| 8 | «Se mantiene hacia adelante» con +0,0009 | El motivo ahora dice «no empeora hacia adelante» |
| 9 | Lógica de decisión en el notebook | `forward_check_column`, `kept_groups` y `results_table` pasaron a `src/` con pruebas |
| 10 | Pruebas débiles o con código muerto | Aserción más estricta en la prueba del historial; línea sobrante eliminada; prueba nueva de que una fila de validación recibe el promedio de entrenamiento |
| 11 | Casos borde sin cubrir | Infinitos de entrada y `forward_split` con fechas superpuestas o validación vacía. **Pendientes (improbables con estos datos):** identificadores de cliente vacíos en `make_group_folds`; columnas 100 % vacías en un pliegue |
| 12 | Texto desactualizado sobre `GroupKFold` | Corregido en `especificaciones/04-features.md` |
| — | Faltaba una línea base trivial | Nueva fila «siempre la mediana» |
