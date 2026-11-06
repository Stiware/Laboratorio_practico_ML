"""Escribe notebooks/06_optimizacion.ipynb (etapa 6): solo orquesta funciones de src/olistbr/. Se ejecuta desde la raíz del repo."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

nb.cells = [
code('''import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
# MLflow imprime el enlace al workspace en cada ejecución: no debe quedar en las salidas del notebook (regla 2, sin datos personales)
os.environ["MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT"] = "true"
# Permite importar el paquete de src/ sin instalarlo
sys.path.insert(0, str(Path.cwd().resolve().parent / "src"))'''),

md('''# 06 · Optimización con seguimiento (tracking)

Especificación: `especificaciones/06-optimizacion.md` · Plan: `specs/004-hyperparameter-tuning/plan.md`.
Este notebook **solo orquesta**: la lógica está en `src/olistbr/models/tuning.py` y `src/olistbr/data/split.py`, con pruebas.

**Qué se decide aquí.** Si los parámetros optimizados de cada finalista (LightGBM cuantil 0,75 e HistGradientBoosting cuantil 0,75) son
**mejores que los de partida** (los del torneo), y cuál de los dos pasa a la validación final.

**Cómo, en cinco pasos:**
1. Se reserva el **25 % de los clientes** (semilla 7): la búsqueda nunca los ve.
2. Con el 75 % restante, Optuna prueba **20 combinaciones por finalista** (presupuesto fijado antes) y minimiza el **costo 3 a 1**
   medio de 5 pliegues por cliente. Cada ensayo queda registrado en MLflow.
3. La mejor puntuación de la búsqueda es **optimista** (es el máximo de muchas estimaciones con ruido): no se reporta como resultado.
4. **Comprobación honesta:** partida y optimizado se ajustan con toda la búsqueda y se comparan en la reserva, con un intervalo del 95 %
   que sortea **clientes completos**. Regla escrita antes de mirar: *se usan los optimizados solo si todo el intervalo está por debajo de cero*.
5. Entre finalistas, el mismo intervalo; si cruza el cero es un empate práctico y gana LightGBM cuantil 0,75.

**Reglas de esta etapa:** la prueba final (`test`) sigue cerrada; mismas variables del torneo (sin marca de fin de año ni historial);
el plazo prometido por Olist nunca entra.'''),

code('''import json
import time
from datetime import date

STARTED = time.perf_counter()  # para anotar en el JSON cuánto tardó la etapa

import mlflow
import numpy as np
import pandas as pd

from olistbr.config import (
    BOOTSTRAP_ROUNDS, GROUP_COL, PROCESSED_DIR, RANDOM_STATE, REPORTS_DIR, RESERVE_SEED, RESERVE_SIZE, TARGET, TIE_WINNER,
    TOURNAMENT_GROUPS, TUNING_EXPERIMENT, TUNING_TRIALS,
)
from olistbr.data.split import assert_disjoint_groups, make_group_folds, reserved_forward_split, split_by_group
from olistbr.features.engineering import add_features
from olistbr.models.evaluation import coverage_by_month, evaluate_candidate_holdout
from olistbr.models.metrics import coverage, mean_delivery_cost, metrics_by_group
from olistbr.models.tuning import (
    FINALIST_NAMES, FINALISTS, bootstrap_cost_difference, choose_finalist, decide_params, fit_and_recommend, n_trees_used,
    starting_candidate, trials_table, tune, tuned_candidate,
)
from olistbr.tracking.mlflow_utils import configure_mlflow, log_run, set_experiment

import logging
logging.getLogger("mlflow").setLevel(logging.WARNING)  # después de importar MLflow: sin mensajes que muestren la ruta del experimento

pd.set_option("display.width", 230)
pd.set_option("display.max_columns", 40)
assert not {"peak_season", "history"} & set(TOURNAMENT_GROUPS), "la optimización usa las mismas variables del torneo"
print("finalistas:", " y ".join(FINALIST_NAMES), f"| {TUNING_TRIALS} ensayos cada uno")'''),

md('''## 1. Conexión con el seguimiento

`configure_mlflow()` usa Databricks si `.env` tiene la conexión y responde; si no, cae solo a `mlflow_local.db` y lo avisa.
El `.env` no se imprime. Para ver el registro local: `mlflow ui --backend-store-uri sqlite:///mlflow_local.db`.'''),

code('''where = configure_mlflow()
experiment = set_experiment(TUNING_EXPERIMENT)
# La ruta completa del experimento en Databricks incluye el correo del usuario: se muestra solo el nombre
print("MLflow:", where.split(" (")[0], "| experimento:", experiment.rsplit("/", 1)[-1])'''),

md('''## 2. Datos: reserva de clientes y pliegues de búsqueda

`add_features` solo crea columnas fila por fila (no aprende de los datos), así que puede ir antes de partir. La reserva se hace con
`split_by_group` (por cliente); los 5 pliegues de búsqueda se crean **una vez** sobre el 75 % de búsqueda y valen para los dos finalistas.'''),

code('''df = add_features(pd.read_parquet(PROCESSED_DIR / "train.parquet"))
search, reserved = split_by_group(df, test_size=RESERVE_SIZE, seed=RESERVE_SEED)
assert_disjoint_groups(search, reserved)  # ningún cliente en la búsqueda y en la reserva

folds = make_group_folds(search, seed=RANDOM_STATE)  # verifica por dentro que cada pliegue no comparte clientes
X_search, y_search = search.drop(columns=[TARGET]), search[TARGET]

print(f"entrenamiento: {len(df):,} pedidos | búsqueda: {len(search):,} pedidos ({search[GROUP_COL].nunique():,} clientes) | "
      f"reserva: {len(reserved):,} pedidos ({reserved[GROUP_COL].nunique():,} clientes)")
print("pedidos de validación por pliegue:", [len(valid) for _, valid in folds])'''),

md('''## 3. La búsqueda (un ensayo = una ejecución registrada)

El estudio se guarda en `data/processed/optuna_optimizacion.db` (ignorado por Git): si la ejecución se corta, al volver a correr
solo se hacen los ensayos que faltan. **El mejor costo de la búsqueda es optimista.**'''),

code('''STORAGE = PROCESSED_DIR / "optuna_optimizacion.db"
studies, trials = {}, []
for name in FINALIST_NAMES:
    studies[name] = tune(name, X_search, y_search, folds, n_trials=TUNING_TRIALS, storage=STORAGE)
    trials.append(trials_table(studies[name], name))
    study = studies[name]
    failed = sum(t.state.name == "FAIL" for t in study.trials)
    print(f"{name}: {len(study.trials)} ensayos ({failed} fallidos) | mejor costo de la búsqueda (OPTIMISTA): {study.best_value:.4f}")
trials = pd.concat(trials, ignore_index=True)'''),

code('''# Los 5 mejores ensayos de cada finalista (costo medio de los 5 pliegues de búsqueda ± su desviación)
columns = ["modelo", "ensayo", "costo_media", "costo_desv", "cobertura_media", "fit_seconds_media"]
trials[columns].sort_values(["modelo", "costo_media"]).groupby("modelo").head(5).round(4)'''),

code('''# Mejores parámetros de cada finalista, junto a los de partida (los del torneo)
def starting_params(name):
    """Los parámetros de partida que corresponden al espacio de búsqueda (más la parada temprana de HistGradientBoosting)."""
    keys = list(studies[name].best_params)
    params = starting_candidate(name).model.get_params()
    extra = ["early_stopping"] if "max_iter" in keys else []
    return {k: params[k] for k in keys + extra}

for name in FINALIST_NAMES:
    print(name)
    print("  partida:    ", starting_params(name))
    print("  optimizados:", {k: round(v, 4) if isinstance(v, float) else v for k, v in studies[name].best_params.items()})'''),

md('''## 4. La comprobación honesta (clientes reservados)

Para cada finalista se ajustan **dos** modelos con toda la búsqueda (partida y optimizado) y se evalúan en la reserva.
`diferencia = costo(optimizado) − costo(partida)` por pedido; negativo = el optimizado cuesta menos. Intervalo del 95 % con
1.000 sorteos de clientes completos.

**Regla escrita antes de mirar:** uso los parámetros optimizados **solo si todo el intervalo está por debajo de cero**. Si cruza
el cero, la mejora no se distingue del ruido y uso los de partida.'''),

code('''real = reserved[TARGET].to_numpy(dtype=float)  # días reales de la reserva: los mismos para los dos finalistas
checks, chosen = [], {}
for name in FINALIST_NAMES:
    start, tuned = starting_candidate(name), tuned_candidate(name, studies[name].best_params)
    real_check, plan_start, model_start = fit_and_recommend(start, search, reserved)
    assert np.array_equal(real_check, real)
    _, plan_tuned, model_tuned = fit_and_recommend(tuned, search, reserved)
    diff, low, high = bootstrap_cost_difference(real, plan_tuned, plan_start, reserved[GROUP_COL], n_boot=BOOTSTRAP_ROUNDS)
    decision, motive = decide_params(low, high)
    checks.append({
        "modelo": name, "costo_partida": mean_delivery_cost(real, plan_start), "costo_optimizado": mean_delivery_cost(real, plan_tuned),
        "diferencia": diff, "ic_inferior": low, "ic_superior": high, "decision": decision,
        "cobertura_partida": coverage(real, plan_start), "cobertura_optimizado": coverage(real, plan_tuned),
        "arboles_partida": n_trees_used(model_start), "arboles_optimizado": n_trees_used(model_tuned), "motivo": motive,
    })
    chosen[name] = {
        "candidate": tuned if decision == "optimizados" else start,
        "params": studies[name].best_params if decision == "optimizados" else starting_params(name),
        "plan": plan_tuned if decision == "optimizados" else plan_start,
        "model": model_tuned if decision == "optimizados" else model_start,
    }
    log_run(f"comprobacion-{name.split(' 0,')[0].lower().replace(' ', '-')}", {"decision": decision},
            {k: float(checks[-1][k]) for k in ["costo_partida", "costo_optimizado", "diferencia", "ic_inferior", "ic_superior"]},
            tags={"etapa": "06_optimizacion", "modelo": name, "paso": "comprobacion_honesta"})
checks = pd.DataFrame(checks)
checks.drop(columns="motivo").round(4)'''),

code('''for row in checks.itertuples():
    print(f"{row.modelo}: {row.decision} — {row.motivo}")
    print(f"  árboles que usó realmente la partida: {row.arboles_partida}")'''),

md('''## 5. Elección entre los dos finalistas

Con los parámetros que eligió la regla para cada uno, se comparan en la reserva: `diferencia = costo(LightGBM) − costo(HistGradientBoosting)`.
Si todo el intervalo queda de un lado del cero gana ese modelo; si cruza el cero es un **empate práctico** y gana LightGBM cuantil 0,75
(más rápido y ganador del torneo).'''),

code('''name_a, name_b = FINALIST_NAMES
diff_ab, low_ab, high_ab = bootstrap_cost_difference(real, chosen[name_a]["plan"], chosen[name_b]["plan"], reserved[GROUP_COL],
                                                     n_boot=BOOTSTRAP_ROUNDS)
winner, winner_motive = choose_finalist(low_ab, high_ab, name_a, name_b, tie_winner=TIE_WINNER)
print(f"costo({name_a}) − costo({name_b}) = {diff_ab:+.4f}  [IC 95 %: {low_ab:+.4f} ; {high_ab:+.4f}]")
print(f"modelo elegido: {winner} — {winner_motive}")'''),

md('''## 6. Revisiones informativas (no cambian la decisión)

**Hacia adelante:** se entrena con las compras de la búsqueda hasta el 2018-02-28 **ya entregadas antes del 1 de marzo** (lo que se
conocía al corte) y se valida con las compras de los **clientes
reservados** del 2018-03-01 al 2018-08-31. Es una sola partición, sin ruido medido. **Por región:** el modelo elegido en la reserva
(principio VIII).'''),

code('''fwd_train, fwd_valid = reserved_forward_split(search, reserved)
print(f"hacia adelante: entrena con {len(fwd_train):,} pedidos de búsqueda ya entregados al corte y valida con {len(fwd_valid):,} de clientes reservados")
forward_rows, monthly = [], []
for name in FINALIST_NAMES:
    for label, candidate in [("partida", starting_candidate(name)), ("optimizado", tuned_candidate(name, studies[name].best_params))]:
        tag = f"{name} · {label}"
        forward_rows.append(evaluate_candidate_holdout(tag, candidate, fwd_train, fwd_valid))
        monthly.append(coverage_by_month(tag, candidate, fwd_train, fwd_valid))
forward = pd.DataFrame(forward_rows)
monthly = pd.concat(monthly, ignore_index=True)
forward[["modelo", "costo", "cobertura", "mae"]].round(4)'''),

code('''for name in FINALIST_NAMES:
    cost = forward.set_index("modelo")["costo"]
    if cost[f"{name} · optimizado"] > cost[f"{name} · partida"]:
        print(f"⚠️ {name}: hacia adelante, el optimizado cuesta MÁS que la partida (solo informativo)")
    else:
        print(f"{name}: hacia adelante, el optimizado no queda peor que la partida")
monthly.pivot(index="mes", columns="modelo", values="cobertura").round(3)'''),

code('''# MAE (del percentil 75, no comparable con modelos de valor típico) y porcentaje tarde por región del cliente, modelo elegido
regions = metrics_by_group(real, chosen[winner]["plan"], chosen[winner]["plan"], reserved["customer_region"].to_numpy(dtype=object))
regions.insert(0, "modelo", winner)
regions.round(4)'''),

md('''## 7. Guardar

`reports/06_ensayos.csv` (una fila por ensayo), `reports/06_best_params.json` (decisión de cada finalista y modelo elegido),
`reports/06_comprobacion.csv` (la comprobación honesta), `reports/06_adelante.csv`, `reports/06_adelante_mensual.csv` y
`reports/06_regiones.csv` (informativos). El motivo de cada decisión se redacta en
`reports/06_optimizacion.md`.'''),

code('''def plain(value):
    """Convierte números de numpy a números de Python para escribir el JSON."""
    return value.item() if isinstance(value, np.generic) else value

summary = {
    "fecha": date.today().isoformat(),
    "semillas": {"reserva": RESERVE_SEED, "pliegues": RANDOM_STATE, "muestreador": RANDOM_STATE, "bootstrap": RANDOM_STATE},
    "presupuesto_por_finalista": TUNING_TRIALS,
    "seguimiento": {"donde": where.split(" (")[0], "experimento": TUNING_EXPERIMENT},
    "finalistas": {},
    "eleccion": {"modelo": winner, "diferencia_a_menos_b": diff_ab, "ic": [low_ab, high_ab], "motivo": winner_motive},
    "hacia_adelante": {"pedidos_entrenamiento": len(fwd_train), "pedidos_validacion": len(fwd_valid)},
    "minutos_de_ejecucion": round((time.perf_counter() - STARTED) / 60, 1),
}
for row in checks.to_dict("records"):
    name = row["modelo"]
    summary["finalistas"][name] = {
        "parametros_partida": {k: plain(v) for k, v in starting_params(name).items()},
        "parametros_optimizados": {k: plain(v) for k, v in studies[name].best_params.items()},
        "mejor_costo_busqueda_optimista": studies[name].best_value,
        "comprobacion": {k: plain(row[k]) for k in ["costo_partida", "costo_optimizado", "diferencia", "decision", "motivo",
                                                    "arboles_partida", "arboles_optimizado"]}
                        | {"ic": [row["ic_inferior"], row["ic_superior"]]},
        "parametros_elegidos": {k: plain(v) for k, v in chosen[name]["params"].items()},
    }

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
trials.round(6).to_csv(REPORTS_DIR / "06_ensayos.csv", index=False)
checks.round(6).to_csv(REPORTS_DIR / "06_comprobacion.csv", index=False)
forward[["modelo", "costo", "cobertura", "tarde", "mae", "n_ajuste", "n_validacion"]].round(6).to_csv(REPORTS_DIR / "06_adelante.csv", index=False)
monthly.round(6).to_csv(REPORTS_DIR / "06_adelante_mensual.csv", index=False)
regions.round(6).to_csv(REPORTS_DIR / "06_regiones.csv", index=False)
(REPORTS_DIR / "06_best_params.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"guardado: {len(trials)} ensayos · decisión en 06_best_params.json · modelo elegido: {winner}")'''),

md('''## 8. Las pruebas automáticas

Desde la raíz del repo:

```bash
pytest tests
```

Espacios, constructores, búsqueda, registro, intervalo por cliente y reglas de decisión en `tests/test_tuning.py`; reserva y
partición hacia adelante con clientes reservados en `tests/test_split.py`.'''),
]
nbf.write(nb, "notebooks/06_optimizacion.ipynb")
print("ok", len(nb.cells))
