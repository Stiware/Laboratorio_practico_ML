# Research: Optimización de hiperparámetros de OnTrack (etapa 6, Fase 2)

No quedaron dudas abiertas en el contexto técnico. Aquí se registran las decisiones de diseño y su motivo. Las verificaciones se hicieron en el `.venv`
el 2026-11-05 (scikit-learn 1.9.1, Optuna 5.0.0, MLflow 3.17.1, SQLAlchemy 2.1.8).

## D1. Reutilizar la evaluación del torneo en vez de escribir otra

- **Decisión:** el objetivo de cada ensayo llama a `evaluate_candidate_folds` (etapa 5) con un `Candidate(modelo, "B", False)` armado con los parámetros del ensayo.
  Esa función ya devuelve, por pliegue, costo 3 a 1, MAE, RMSE, R², sesgo, tarde, cobertura, margen y tiempo de ajuste, y arma el pipeline con `TOURNAMENT_GROUPS`.
- **Motivo:** los ensayos se miden exactamente igual que el torneo (mismo preprocesador, mismas variables, mismo plazo recomendado). Una segunda función de evaluación
  podría diferir sin que nadie lo note.
- **Alternativa descartada:** usar `evaluate_folds` del ejemplo (CreditSight) adaptada: duplicaría lógica que ya existe y tiene pruebas.

## D2. El percentil 75 no se puede cambiar desde un ensayo

- **Decisión:** `make_lgbm_quantile(params)` y `make_hgb_quantile(params)` fijan `objective="quantile", alpha=0.75` y `loss="quantile", quantile=0.75`.
  Si `params` trae alguna de esas claves (`objective`, `alpha`, `loss`, `quantile`), lanzan `ValueError` con un mensaje claro.
- **Motivo:** si un ensayo optimizara otro percentil, el costo 3 a 1 dejaría de medir el plazo correcto sin avisar (caso borde de la especificación).
- **Alternativa descartada:** sobrescribir en silencio: oculta un error de configuración.

## D3. Parámetros de partida = los del torneo, tomados de `candidates()`

- **Decisión:** la partida se obtiene con `candidates()[nombre]` (no se copian los números a mano). Una prueba verifica que es así.
- **Motivo:** la comparación «partida contra optimizado» debe ser contra lo que realmente se midió en el torneo, incluida la parada temprana por defecto de
  HistGradientBoosting (Q5).
- **Hecho verificado:** con `early_stopping="auto"` y más de 10.000 filas, HistGradientBoosting activa la parada temprana (`do_early_stopping_ = True`). En una tabla sintética
  de 20.000 filas usó 98 de 300 árboles. Con los datos reales se mide en el notebook (`n_trees_used`) y se reporta.

## D4. Parada temprana apagada en la búsqueda de HistGradientBoosting

- **Decisión:** `make_hgb_quantile` fija `early_stopping=False`; el número de árboles es `max_iter` (150 a 800).
- **Motivo:** (a) con la parada encendida, `max_iter` no controla el número de árboles; (b) la parada aparta un 10 % de **filas**, no de clientes, para decidir cuándo parar;
  (c) LightGBM no la tiene, y los dos finalistas deben buscarse en igualdad.
- **Alternativa descartada:** parada temprana con un apartado por cliente: no la ofrece scikit-learn y agregaría código propio sin necesidad.

## D5. Reserva y pliegues de búsqueda

- **Decisión:** `split_by_group(train, test_size=0.25, seed=7)` (ya existe y ya verifica que no hay clientes compartidos) devuelve `(busqueda, reserva)`.
  Sobre `busqueda` se crean **una vez** 5 pliegues con `make_group_folds(seed=42)`.
- **Motivo:** la guía (§3.4) pide el 25 % de las entidades con semilla 7. Los pliegues de la etapa 4 incluyen a los clientes reservados, así que no sirven tal cual.
- **Consecuencia:** cada pliegue de búsqueda ajusta con ≈ 46.000 pedidos (en el torneo ≈ 61.500). Los costos de la búsqueda **no** se comparan uno a uno con los del torneo.

## D6. Búsqueda con Optuna, reproducible y reanudable

- **Decisión:** `optuna.create_study(direction="minimize", sampler=TPESampler(seed=42), study_name="ontrack-<finalista>-costo", storage=..., load_if_exists=True)`.
  El estudio se guarda en `data/processed/optuna_optimizacion.db` (carpeta ya ignorada por Git). Al reanudar se hacen solo `n_trials − ensayos terminados`.
- **Verificado:** un estudio guardado en SQLite se reabre con sus ensayos (prueba con 2 ensayos); `optuna.trial.FixedTrial` existe y sirve para probar los espacios.
- **Ensayos fallidos:** si el costo no es finito, el objetivo lanza `ValueError`; `study.optimize(..., catch=(ValueError,))` lo marca como fallido y no puede ser el mejor.
  Un ensayo fallido **cuenta** dentro del presupuesto de 20 (si no, el presupuesto no estaría fijado de antemano). Se informa cuántos fallaron.
- **Limitación:** si la búsqueda se corta y se reanuda, el TPE no garantiza la misma secuencia que una corrida sin cortes. La reproducibilidad (SC-006) se exige para una corrida completa sin cortes.

## D7. Seguimiento: módulo existente, Databricks con respaldo local

- **Decisión:** `configure_mlflow()` (sin `force_local`), `set_experiment("ontrack-optimizacion")` y `log_run(..., nested=True)` por ensayo dentro de un `mlflow.start_run`
  por finalista. Etiquetas: `etapa=06_optimizacion`, `modelo=<finalista>`, `metrica_optimizada=costo`.
- **Métricas por ensayo:** media y desviación (`ddof=1`) entre pliegues de costo, MAE, RMSE, R², sesgo, tarde y cobertura, más el tiempo medio de ajuste.
- **Motivo:** regla 7 de `CLAUDE.md` (MLflow en Databricks) y guía §4. El módulo no se reescribe (prompts de la etapa 6). `.env` no se abre ni se imprime.
- **En las pruebas:** `tune(..., log=False)`; el seguimiento ya tiene sus pruebas en `test_tracking.py`.

## D8. Intervalo de la diferencia remuestreando clientes

- **Decisión:** `bootstrap_cost_difference(real, plan_a, plan_b, groups, n_boot=1000, seed=42)`:
  1. costo por pedido con `delivery_cost` (3 a 1) para cada plazo; diferencia por pedido = costo(A) − costo(B);
  2. suma y número de pedidos **por cliente**;
  3. 1.000 sorteos con reemplazo de clientes; en cada uno, diferencia media = suma de sumas / suma de pedidos;
  4. devuelve `(diferencia_total, percentil 2,5, percentil 97,5)`.
- **Motivo:** los pedidos de un mismo cliente no son independientes; sortear filas daría un intervalo demasiado angosto (guía §3.6). Es la misma forma que el ejemplo.
- **Validaciones:** largos distintos o clientes vacíos → `ValueError`.

## D9. Reglas de decisión en código

- **`decide_params(low, high)`:** `"optimizados"` solo si `high < 0`; en cualquier otro caso `"partida"`. El motivo distingue «todo el intervalo bajo cero», «cruza o toca el cero»
  y «todo el intervalo sobre cero (el optimizado es peor)».
- **`choose_finalist(low, high, name_a, name_b, tie_winner)`:** con la diferencia A − B; `high < 0` → A; `low > 0` → B; si no, empate práctico → `tie_winner`
  (LightGBM cuantil 0,75, Q7). Devuelve el nombre y el motivo.
- **Motivo:** que la regla aprobada esté escrita en código probado y no se aplique «a ojo» en el notebook.

## D10. Prueba hacia adelante con clientes reservados

- **Decisión:** `reserved_forward_split(busqueda, reserva)` en `data/split.py`: entrena con las compras de `busqueda` hasta el 2018-02-28 y valida con las compras de `reserva`
  del 2018-03-01 al 2018-08-31 (mismas fechas de `config.py`). Como `busqueda` y `reserva` no comparten clientes, la partición tampoco; igual se verifica con `assert_disjoint_groups`.
  Se mide con las funciones existentes `evaluate_candidate_holdout` y `coverage_by_month`.
- **Motivo:** la prueba hacia adelante del torneo incluye en su validación a clientes que la búsqueda vio (Q6). Es informativa: no cambia la decisión.

## D11. Revisión por región en la reserva

- **Decisión:** se ajusta el modelo elegido con `busqueda`, se predice `reserva` y se agrupa con `metrics_by_group` (ya existe) por `customer_region`.
- **Motivo:** principio VIII. Solo se reporta.

## D12. Notebook generado por script, como en las etapas 3 a 5

- **Decisión:** `scripts/generadores/generar_notebook_06.py` arma `notebooks/06_optimizacion.ipynb` sobre la plantilla del curso, leyendo `train.parquet` (no `train.joblib`).
- **Motivo:** mismo flujo que los notebooks 03 a 05 (`scripts/generadores/README.md`); el notebook solo orquesta.
