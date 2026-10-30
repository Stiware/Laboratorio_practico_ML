# Databricks notebook source
# MAGIC %md
# MAGIC # 06 · Optimización de métricas técnicas con seguimiento — versión Databricks
# MAGIC
# MAGIC El cálculo ocurre **aquí**, en cómputo *serverless* de Databricks, y cada ensayo queda registrado en **MLflow
# MAGIC dentro de Databricks** con el **tablero completo de métricas**: las de negocio (costo) y las técnicas
# MAGIC (**AUC**, PR-AUC, **recall** por clase, precisión, exactitud balanceada, F1, log loss).
# MAGIC
# MAGIC **Qué se optimiza:** una métrica **técnica**, el **AUC macro** (`METRICA = "auc_ovr"`): mide qué tan bien las
# MAGIC probabilidades *ordenan* a los clientes, sin depender de dónde se ponga el umbral. Aunque se optimice una sola,
# MAGIC **todas se registran**, y eso permite ver qué le pasa al costo y al recall mientras el AUC mejora.
# MAGIC
# MAGIC > **¿Por qué no optimizar el recall?** Solo, es una trampa: se obtiene recall 1,0 marcando a *todos* como `Poor`.
# MAGIC > Se optimiza una métrica que premia el orden (AUC) y se **vigila** el recall junto a la precisión.
# MAGIC
# MAGIC **Qué mirar en la interfaz al terminar:** menú izquierdo → *Experiments* → `creditsight-06-optimizacion-auc`.
# MAGIC Las ejecuciones **madre** (`optuna-lightgbm`, `optuna-random forest`) y, desplegadas, sus ensayos; ordena por
# MAGIC `auc_ovr`, por `cost` o por `recall_poor`; marca ensayos y pulsa **Compare**.
# MAGIC
# MAGIC **Datos y código:** salen de este mismo repositorio, clonado como *Git folder* en Databricks. El notebook arma el
# MAGIC conjunto de entrenamiento desde `data/01-clasificacion/credit_score_classification.csv` con las mismas funciones del
# MAGIC notebook 03 (`load_raw`, `clean`, `split_by_customer`). Para actualizar el código: se sube el cambio al repo y aquí se
# MAGIC hace *Pull* (menú de Git de la carpeta).

# COMMAND ----------

# MAGIC %pip install -q lightgbm xgboost optuna python-dotenv

# COMMAND ----------

# Reinicia Python para que la sesión vea las librerías recién instaladas
dbutils.library.restartPython()

# COMMAND ----------

import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

METRICA = "auc_ovr"                         # métrica que optimiza la búsqueda (todas se registran igual)
EXPERIMENTO = "creditsight-06-optimizacion-auc"

from databricks.sdk import WorkspaceClient

# Carpeta de experimentos: la del usuario que ejecuta el notebook (cámbiale el nombre si usas otra)
USUARIO = WorkspaceClient().current_user.me().user_name
CARPETA = f"/Users/{USUARIO}/curso_laboratorio_casos_avanzados"
os.environ["EXPERIMENT_DIR"] = CARPETA

# Un notebook de un Git folder corre con la carpeta del notebook como directorio actual; se sube hasta
# encontrar `src/creditsight` para importar el paquete sin instalarlo (igual que en los notebooks locales)
SRC = next(p / "src" for p in [Path.cwd().resolve(), *Path.cwd().resolve().parents] if (p / "src" / "creditsight").is_dir())
sys.path.insert(0, str(SRC))
os.environ["PYTHONPATH"] = f"{SRC}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"   # también para procesos hijos
print("Código del ejemplo en:", SRC)
import creditsight  # noqa: E402,F401  (si falla aquí, la ruta impresa arriba no contiene el paquete)

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score

from creditsight.config import GROUP_COL, TARGET
from creditsight.data.cleaning import clean, load_raw
from creditsight.data.split import group_cv, split_by_customer
from creditsight.models.train import build_pipeline, candidates, encode_target, metrics_from_proba
from creditsight.models.tuning import (
    MODELS, OBJETIVOS, bootstrap_cost_difference, bootstrap_metric_difference, tune,
)
from creditsight.tracking.mlflow_utils import configure_mlflow, set_experiment

where = configure_mlflow()                       # dentro de Databricks no hay que configurar nada
experimento = set_experiment(EXPERIMENTO)
print("MLflow:", where)
print("Experimento:", experimento)
print(f"Se optimiza: {METRICA} ({OBJETIVOS[METRICA]})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reservar clientes que la búsqueda no verá
# MAGIC
# MAGIC Del conjunto de entrenamiento separamos el 25 % de los **clientes** (por cliente, siempre). La búsqueda trabaja
# MAGIC con el 75 %; la prueba final sigue cerrada.

# COMMAND ----------

# Mismo conjunto de entrenamiento que el notebook 03: limpiar y separar la prueba final POR CLIENTE
df = clean(load_raw())
train, _test = split_by_customer(df, test_size=0.2)      # la prueba final no se toca aquí
tune_df, val_df = split_by_customer(train, test_size=0.25, seed=7)
X_tune, y_tune = tune_df.drop(columns=[TARGET]), encode_target(tune_df[TARGET])
X_val, y_val = val_df.drop(columns=[TARGET]), encode_target(val_df[TARGET])
cv = group_cv(3)
print(f"búsqueda: {tune_df.Customer_ID.nunique():,} clientes · validación reservada: {val_df.Customer_ID.nunique():,} clientes")

# COMMAND ----------

# MAGIC %md
# MAGIC ## La búsqueda
# MAGIC
# MAGIC Optuna usa un muestreador bayesiano (TPE). Cada ensayo es una **ejecución anidada** dentro de una ejecución madre
# MAGIC en MLflow. **Mientras corre, abre la pestaña *Experiments* y verás cómo se llena la lista**, con todo el tablero.

# COMMAND ----------

# Parámetro "modo": "humo" hace una prueba rápida de 2 ensayos por modelo (para comprobar que todo corre);
# "completo" es la corrida real. Una corrida que falla se reintenta sola y duplica ejecuciones: probar primero en humo.
dbutils.widgets.dropdown("modo", "completo", ["completo", "humo"])
TRIALS = {"lightgbm": 2, "random forest": 2} if dbutils.widgets.get("modo") == "humo" else {"lightgbm": 20, "random forest": 12}
print("modo:", dbutils.widgets.get("modo"), TRIALS)
studies = {}


def run_search(name):
    studies[name] = tune(name, X_tune, y_tune, cv, n_trials=TRIALS[name], storage="/tmp/optuna_estudios.db", metric=METRICA)
    s = studies[name]
    print(f"{name:14s} mejor {METRICA} en la búsqueda = {s.best_value:.4f}  (ensayo {s.best_trial.number} de {TRIALS[name]})")


run_search("lightgbm")

# COMMAND ----------

run_search("random forest")

# COMMAND ----------

fig, axes = plt.subplots(1, 2, figsize=(11, 3.2), sharey=True)
acumular = np.maximum.accumulate if OBJETIVOS[METRICA] == "maximize" else np.minimum.accumulate
for ax, (name, study) in zip(axes, studies.items()):
    values = [t.value for t in study.trials]
    ax.plot(values, "o", color="#9aa0a6", label="ensayo")
    ax.plot(acumular(values), "-", color="#4c72b0", label="mejor hasta ahora")
    ax.set_title(name); ax.set_xlabel("ensayo"); ax.legend()
axes[0].set_ylabel(f"{METRICA} en validación cruzada")
plt.tight_layout(); plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## El tablero completo
# MAGIC
# MAGIC Cada ensayo registró **todas** las métricas. Los mejores por AUC, con las demás al lado: así se ve qué pasa con el
# MAGIC costo y con el recall mientras el AUC mejora.

# COMMAND ----------

runs = mlflow.search_runs(experiment_names=[experimento], filter_string="tags.etapa = '06_optimizacion' and metrics.cost > 0",
                          max_results=500)
runs = runs[runs["tags.mlflow.parentRunId"].notna()] if "tags.mlflow.parentRunId" in runs else runs
columnas = {
    "tags.modelo": "modelo", "metrics.auc_ovr": "AUC (OvR)", "metrics.auc_poor": "AUC Poor", "metrics.pr_auc_poor": "PR-AUC Poor",
    "metrics.recall_poor": "recall Poor", "metrics.recall_standard": "recall Standard", "metrics.recall_good": "recall Good",
    "metrics.precision_poor": "precisión Poor", "metrics.balanced_accuracy": "exact. balanceada", "metrics.macro_f1": "F1 macro",
    "metrics.log_loss": "log loss", "metrics.cost": "costo", "metrics.fit_seconds": "segundos",
}
tablero = runs[[c for c in columnas if c in runs.columns]].rename(columns=columnas).sort_values("AUC (OvR)", ascending=False)
display(tablero.head(12).round(4))

# COMMAND ----------

# MAGIC %md
# MAGIC ### ¿Optimizar el AUC también baja el costo?
# MAGIC
# MAGIC Cada punto es un ensayo. Si optimizar el AUC bajara el costo, los puntos formarían una línea descendente. Cuando no
# MAGIC es así, es la señal de que **una métrica técnica y una de negocio no siempre se mueven juntas**.

# COMMAND ----------

fig, ax = plt.subplots(figsize=(6.2, 3.8))
for modelo, grupo in tablero.groupby("modelo"):
    ax.scatter(grupo["AUC (OvR)"], grupo["costo"], label=modelo, alpha=0.8)
ax.set_xlabel("AUC (OvR macro) · más es mejor"); ax.set_ylabel("costo esperado · menos es mejor"); ax.legend()
plt.tight_layout(); plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## ¿Qué parámetros importaron?

# COMMAND ----------

imp = pd.DataFrame({name: pd.Series(optuna.importance.get_param_importances(study)) for name, study in studies.items()})
# sin fillna(""): mezclar números y texto en una columna hace fallar la conversión a Spark de display()
display(imp.round(3).reset_index().rename(columns={"index": "parámetro"}))

# COMMAND ----------

# MAGIC %md
# MAGIC ## La comprobación honesta
# MAGIC
# MAGIC Comparamos, sobre los **clientes reservados**, el modelo con los parámetros de partida contra el modelo con los
# MAGIC mejores parámetros encontrados. Dos veredictos, cada uno con un intervalo del 95 % que remuestrea **clientes**
# MAGIC (no filas): el del **AUC** (lo que se optimizó) y el del **costo** (lo que le importa al negocio). Si el
# MAGIC intervalo cruza el cero, la mejora **no se distingue del ruido**.

# COMMAND ----------

auc_fn = lambda yy, pp: roc_auc_score(yy, pp, multi_class="ovr", average="macro", labels=[0, 1, 2])
defaults = {n: m for n, (m, _) in candidates().items()}
rows, board = [], []
for name, study in studies.items():
    _, make_fn, scale = MODELS[name]
    tuned = build_pipeline(make_fn(study.best_params), scale=scale).fit(X_tune, y_tune)
    default = build_pipeline(defaults[name], scale=scale).fit(X_tune, y_tune)
    p_tuned, p_default = tuned.predict_proba(X_val), default.predict_proba(X_val)
    m_tuned, m_default = metrics_from_proba(y_val, p_tuned), metrics_from_proba(y_val, p_default)
    c_diff, c_low, c_high = bootstrap_cost_difference(y_val, p_tuned, p_default, val_df[GROUP_COL])
    a_diff, a_low, a_high = bootstrap_metric_difference(y_val, p_tuned, p_default, val_df[GROUP_COL].to_numpy(), auc_fn)
    rows.append({
        "modelo": name,
        # veredicto del NEGOCIO (lo usa la etapa 7 para decidir si se queda con los parámetros optimizados)
        "costo por defecto": m_default["cost"], "costo optimizado": m_tuned["cost"],
        "diferencia (opt. − defecto)": c_diff, "IC 95 % inferior": c_low, "IC 95 % superior": c_high,
        "¿mejora real?": "sí" if c_high < 0 else ("no distinguible" if c_low <= 0 <= c_high else "empeora"),
        # veredicto TÉCNICO (lo que se optimizó)
        "AUC por defecto": m_default["auc_ovr"], "AUC optimizado": m_tuned["auc_ovr"],
        "diferencia de AUC": a_diff, "IC AUC inferior": a_low, "IC AUC superior": a_high,
        "¿mejora real del AUC?": "sí" if a_low > 0 else ("no distinguible" if a_low <= 0 <= a_high else "empeora"),
    })
    for etiqueta, m in (("por defecto", m_default), ("optimizado", m_tuned)):
        board.append({"modelo": name, "parámetros": etiqueta, "AUC (OvR)": m["auc_ovr"], "AUC Poor": m["auc_poor"],
                      "PR-AUC Poor": m["pr_auc_poor"], "recall Poor": m["recall_poor"], "recall Standard": m["recall_standard"],
                      "recall Good": m["recall_good"], "precisión Poor": m["precision_poor"],
                      "exact. balanceada": m["balanced_accuracy"], "F1 macro": m["macro_f1"], "log loss": m["log_loss"],
                      "costo": m["cost"]})
check = pd.DataFrame(rows).set_index("modelo")
display(pd.DataFrame(board).set_index(["modelo", "parámetros"]).round(4))
display(check.round(4))

# COMMAND ----------

import sklearn  # noqa: E402

best = {
    name: {"params": study.best_params, "metric": METRICA, "cv_value": study.best_value,
           "mlflow_parent_run_id": study.user_attrs.get("mlflow_parent_run_id"), "tracking": "databricks (serverless)"}
    for name, study in studies.items()
}
payload = {
    "best": best,
    "check": check.reset_index().to_dict(orient="records"),
    "tablero": pd.DataFrame(board).to_dict(orient="records"),
    "entorno": {"sklearn": sklearn.__version__, "pandas": pd.__version__, "mlflow": mlflow.__version__, "optuna": optuna.__version__},
    "experimento": experimento, "metrica": METRICA,
}
dbutils.notebook.exit(json.dumps(payload, default=float))
