# Especificación · Etapa 7 · Validación final e interpretación

> Se llena **antes** de pedirle código a Claude. Borrador propuesto por Claude el 5 nov 2026 a partir de la guía
> `material-del-curso/kit/guia-validacion-registro-y-app.md` (secciones 0 a 4), la plantilla `notebooks/07_validacion_interpretacion.ipynb`,
> `reports/06_optimizacion.md` y los pendientes que dejaron las etapas 3 a 6.
> **La guía está escrita para clasificación** (probabilidades, umbral, calibración); OnTrack es **regresión** que recomienda un plazo, así que
> cada pieza se adapta (sección 1). **Esta etapa abre `data/processed/test.parquet` una sola vez**: todo lo de la sección 2 se decide antes.

**Grupo:** _______________   **Fecha:** 5 nov 2026   **Estado:** ☐ borrador  ☒ cerrada _(el grupo aceptó todas las propuestas de Q1 a Q8 el 5 nov 2026)_

## 1. De la guía (clasificación) a OnTrack (regresión)

| Pieza de la guía | En clasificación | En OnTrack |
|---|---|---|
| Decisiones congeladas | modelo, parámetros, calibración, métrica, umbral | modelo, parámetros, **recalibración en el tiempo**, métrica (costo 3 a 1) y regla (el plazo es el percentil 75 que predice el modelo) |
| ¿Calibrar las probabilidades? | `CalibratedClassifierCV` con entidades reservadas | No hay probabilidades. El equivalente es la **cobertura**: ¿el plazo cubre ≈ 75 % de los pedidos? Ya se midió en la reserva (73,9 %) y cambia mes a mes (Q2) |
| Métrica en la prueba con intervalo por entidad | pérdida por fila, `bootstrap_por_entidad` | costo 3 a 1 por pedido, intervalo sorteando **clientes** (la función ya existe: `bootstrap_cost_difference`; falta la de una sola media) |
| Baseline | el del torneo | las líneas base del torneo: **plazo fijo del percentil 75** y siempre la mediana (Q3) |
| Errores por segmento | % marcado contra % real | % de pedidos tarde contra el 25 % esperado, por región y por mes (Q5) |
| Carga de trabajo | casos que el modelo manda a revisión | No aplica (no hay revisión humana). El análogo: **días de holgura** (cuánto sobra el plazo) frente a pedidos tarde (Q5) |
| SHAP | qué variables mueven la probabilidad | qué variables mueven el **plazo** (Q6) |

## 2. Preguntas y decisiones del grupo (5 nov 2026)

**Decisión:** el grupo aceptó **todas** las propuestas de Claude (columna «Propuesta») el 5 nov 2026.

| # | Pregunta | Por qué importa | Propuesta de Claude |
|---|---|---|---|
| Q1 | **¿Qué modelo se congela?** | Si se elige mirando la prueba, la prueba deja de medir generalización | **LightGBM cuantil 0,75 con los parámetros optimizados** (`reports/06_best_params.json`), reajustado con **todo** el entrenamiento (76.924 pedidos), percentil 0,75 fijo, mismas 7 variables del torneo. Se escribe con fecha y hora en `reports/07_validacion.md` **antes** de abrir la prueba. Se guarda entero en `models/final_pipeline.joblib` |
| Q2 | **¿Se recalibra el plazo con datos recientes antes de la prueba?** (pendiente 2 del torneo) | La cobertura cambió de 61 % a 93 % según el mes. Pero la prueba se partió **por cliente, no por fecha**: mezcla todos los meses, así que una recalibración temporal no se puede evaluar bien con ella | **No se recalibra el modelo de la etapa 7.** Se mide la cobertura mes a mes **en la prueba** y se reporta. La recalibración pasa a ser una **regla de uso** para la ficha y la app: vigilar la cobertura cada mes y recalibrar si sale de una banda (por ejemplo, 70 % a 80 %). Alternativa: medir en la reserva una corrección simple con los últimos 3 meses y adoptarla solo si el intervalo queda bajo cero (más trabajo, y con una sola ventana de tiempo) |
| Q3 | **¿Contra qué baseline se compara en la prueba?** | Un modelo solo sirve si mejora una regla simple con confianza | **Principal: plazo fijo del percentil 75 del entrenamiento** (≈ 15,7 días, el «mejor plazo fijo» del caso de negocio). Diferencia `costo(modelo) − costo(plazo fijo)` con intervalo del 95 % por cliente; el modelo «mejora con confianza» si todo el intervalo queda bajo cero. Además, como referencia, siempre la mediana y **el plazo que prometió Olist** (`order_estimated_delivery_date`; se mide una vez, nunca entra al modelo, decisión Q1 de la etapa 4) |
| Q4 | **¿Qué métricas se reportan en la prueba?** | Una sola cifra esconde cosas | **Costo 3 a 1** (con intervalo) · **% tarde** y **cobertura** (con intervalo) · MAE y sesgo del plazo (aclarando que es el percentil 75, no el valor típico) · **sensibilidad del costo** con proporciones 1, 2, 3, 5 y 9 a 1 (`COST_RATIOS`, caso de negocio) · las mismas cifras **incluyendo los 64 pedidos extremos** (pendiente de la etapa 3), como escenario pesimista |
| Q5 | **¿Qué segmentos y qué «carga»?** | Un buen promedio puede esconder un grupo al que le va mal (principio VIII) | Segmentos: **región del cliente** (5) y **mes de compra** (cobertura mes a mes). En lugar de «carga de trabajo»: **días de holgura media** (plazo − real cuando llega antes) y **días de atraso medio** (cuando llega después), en total y por región |
| Q6 | **¿Cómo se interpreta?** | La guía pide SHAP con sus límites | **SHAP** (`shap==0.54.1`, ya en `requirements.txt`) sobre una muestra de **2.000 pedidos de la prueba** (semilla 42): las 10 variables que más mueven el plazo y su dirección. Se declara que explica al modelo, no al mundo, y que la ubicación del cliente (estado y región) es un atributo **sensible** (puede tratar distinto a regiones) |
| Q7 | **¿Se miden las sensibilidades pendientes** (marca de fin de año F6, historial por vendedor F8)? | Si se miden mirando la prueba, sería decidir con la prueba | **No en esta etapa.** Se declaran como **no evaluadas** en el reporte y la ficha. Medirlas exigiría reabrir la búsqueda con la reserva, antes de la prueba |
| Q8 | **¿Probabilidad de retraso** (apartado secundario del caso)? | El caso la menciona; el modelo solo da un percentil | **No se calcula por pedido.** Por diseño, el plazo deja ≈ 25 % de pedidos tarde; se reporta el % tarde real en la prueba (en total y por región). Una probabilidad por pedido necesitaría varios percentiles; queda como trabajo futuro |

## 3. Reglas escritas antes de abrir la prueba

1. **Todo lo de la sección 2 queda decidido y escrito** (con fecha y hora) en `reports/07_validacion.md` **antes** de leer `test.parquet`.
2. El modelo final se ajusta **solo con `train.parquet`** y se guarda antes de abrir la prueba.
3. La prueba se abre **una sola vez**, en una función que deja constancia en `reports/prueba_abierta.txt` (fecha y hora) y **se niega a abrirla de nuevo** si la constancia ya existe.
4. **Nada del modelo se cambia después de ver la prueba.** Si algo no gusta, se anota como límite.
5. Los intervalos sortean **clientes completos** (1.000 sorteos, semilla 42).
6. El plazo prometido por Olist solo se usa como referencia en la prueba; nunca entra al modelo.

## 4. Código previsto

En un módulo nuevo `src/olistbr/models/validation.py`, que reutiliza `bootstrap_cost_difference`, `regression_metrics`, `metrics_by_group` y `coverage_by_month`:

| Función | Qué hace | Pruebas (`tests/test_validation.py`) |
|---|---|---|
| `open_test_once(path, mark_path)` | lee la prueba y escribe la constancia; si la constancia existe, falla con un mensaje claro | `test_test_set_opens_only_once` (con archivos temporales) |
| `bootstrap_mean(values, groups, n_boot, seed)` | media por pedido con intervalo del 95 % sorteando clientes | casos a mano (constante, un cliente grande, reproducible) |
| `test_report(real, plan, baselines, groups)` | tabla de métricas del modelo y de cada baseline, con intervalos y diferencias contra el baseline principal | casos a mano |
| `cost_sensitivity(real, plan, ratios)` | costo con cada proporción de `COST_RATIOS` | con 1 a 1 es igual al MAE |
| `slack_and_delay(real, plan)` | días de holgura y de atraso medios | casos a mano |
| `top_shap(pipeline, X, n)` | las n variables con más peso (media del valor SHAP absoluto) y su signo medio | modelo de juguete |

## 5. Entregables

- `notebooks/07_validacion_interpretacion.ipynb` (plantilla del curso; **solo orquesta**), generado por `scripts/generadores/generar_notebook_07.py`.
- `models/final_pipeline.joblib` (modelo ajustado con todo el entrenamiento).
- `reports/prueba_abierta.txt` (constancia), `reports/07_metricas_test.json`, `reports/07_segmentos.csv`, `reports/07_shap.csv`.
- `reports/07_validacion.md` (plantilla existente) completado **después** de ver las cifras.
- Línea en `reports/bitacora_decisiones.md` y en la sección 7 de `especificacion-proyecto.md`.

## 6. Criterios de cierre

- ☒ Q1 a Q8 respondidas y confirmadas por el grupo (5 nov 2026).
- ☒ Decisiones congeladas con fecha y hora **antes** de abrir la prueba.
- ☒ La prueba se abrió una sola vez para medir (constancia 2026-11-05 20:44:44) y no se cambió nada después. _(La etapa 8 la releyó una vez, sin decidir nada, para la ida y vuelta: declarado en `reports/07_validacion.md` §3.)_
- ☒ Costo 3 a 1 del modelo y del baseline principal, con la diferencia y su intervalo por cliente.
- ☒ Segmentos (región y mes), holgura y atraso, sensibilidad del costo y escenario con los 64 extremos.
- ☒ SHAP con sus límites y el atributo sensible declarado.
- ☒ `pytest tests` en verde; revisión con `auditor-de-fuga` y `revisor-de-codigo-ia`.

## 7. Riesgos

- **La reserva de la etapa 6 dio 9,083**, que es algo optimista; es normal que la prueba dé un costo distinto. No es motivo para cambiar nada.
- **La ventaja de LightGBM sobre HistGradientBoosting fue chica** (intervalo hasta −0,002). Si en la prueba se viera al revés, no se cambia de modelo (regla 4).
- **El conjunto de prueba no es temporal**: la prueba mide generalización a **clientes nuevos** del mismo periodo, no a meses futuros. La limitación temporal se declara.
