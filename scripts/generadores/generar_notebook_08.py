"""Escribe notebooks/08_registro_modelo.ipynb (etapa 8): solo orquesta funciones de src/olistbr/. Se ejecuta desde la raíz.

⚠️ No volver a correr este generador sobre un notebook ya ejecutado sin guardar antes sus salidas: las borra. Cada ejecución del notebook
crea una versión nueva en el registro.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

nb.cells = [
code('''import logging
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
# Sin datos personales en las salidas: MLflow imprime el enlace del workspace y sus mensajes informativos muestran la ruta del
# experimento, que incluye el correo de la cuenta (regla 2)
os.environ["MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT"] = "true"
logging.getLogger("mlflow").setLevel(logging.WARNING)
# Permite importar el paquete de src/ sin instalarlo
sys.path.insert(0, str(Path.cwd().resolve().parent / "src"))'''),

md('''# 08 · Registro del modelo

Especificación: `especificaciones/08-registro.md` · Plan: `specs/005-final-validation-registry/plan.md`.
Este notebook **solo orquesta**: la lógica está en `src/olistbr/tracking/registry.py`, con pruebas.

**Tracking frente a registro.** El tracking (etapa 6) guardó 44 ejecuciones de prueba: «¿qué probamos?». El registro guarda **versiones del
modelo**: «¿qué modelo usamos y en qué versión?». El alias **`champion`** es una etiqueta móvil: quien use el modelo pide «el champion».

**Reglas:** se registra **exactamente** `models/final_pipeline.joblib` de la etapa 7 (no se reentrena nada); el ejemplo de entrada no lleva
identificadores; la prueba se lee **solo** para comprobar que el modelo recuperado predice igual (no se decide nada con ella).'''),

code('''import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException

from olistbr.config import MODELS_DIR, PROCESSED_DIR, REPORTS_DIR, TARGET, TOURNAMENT_GROUPS
from olistbr.features.engineering import add_features
from olistbr.models.metrics import mean_delivery_cost
from olistbr.tracking.mlflow_utils import configure_mlflow, set_experiment, uses_databricks
from olistbr.tracking.registry import input_example, load_champion, register_final_model, set_champion

logging.getLogger("mlflow").setLevel(logging.WARNING)  # después de importar MLflow, que reconfigura su nivel al cargarse

MODEL_NAME = "LightGBM cuantil 0,75"
REGISTERED_NAME = "ontrack_plazo_entrega"
EXPERIMENT = "ontrack-modelo-final"'''),

md('''## 0. Conexión con el seguimiento

Databricks (Unity Catalog) si `.env` conecta; si no, MLflow local de respaldo. El `.env` no se imprime.'''),

code('''where = configure_mlflow()
experiment = set_experiment(EXPERIMENT)
PLATFORM = "databricks (Unity Catalog)" if uses_databricks() else "local (respaldo)"
print("MLflow:", PLATFORM, "| experimento:", experiment.rsplit("/", 1)[-1])'''),

md('''## 1. Cargar el modelo final y los resultados de la prueba'''),

code('''pipeline = joblib.load(MODELS_DIR / "final_pipeline.joblib")
results = json.loads((REPORTS_DIR / "07_metricas_test.json").read_text(encoding="utf-8"))
frozen = results["decisiones_congeladas"]
model_row = next(r for r in results["resultado"] if r["regla"] == MODEL_NAME)
baseline_row = next(r for r in results["resultado"] if r["regla"] == frozen["baseline_principal"])
train = add_features(pd.read_parquet(PROCESSED_DIR / "train.parquet"))
example = input_example(train, n=3)
print(f"modelo: {MODEL_NAME} | prueba abierta el {results['prueba_abierta']} | costo en la prueba: {model_row['costo']:.3f}")
print("columnas del ejemplo de entrada:", len(example.columns), "| sin identificadores:",
      not {"customer_unique_id", "order_id", "order_estimated_delivery_date"} & set(example.columns))'''),

md('''## 2. Qué se registra

Parámetros (tipo de modelo, hiperparámetros, percentil, variables), **métricas de la prueba**, firma, ejemplo de entrada y etiquetas.
El modelo se guarda con `cloudpickle` (aviso: un pickle puede ejecutar código al cargarse; solo se cargan modelos del propio grupo).'''),

code('''params = {"modelo": MODEL_NAME, "percentil": 0.75, "grupos_variables": ",".join(TOURNAMENT_GROUPS), **frozen["parametros"]}
metrics = {
    "costo_prueba": model_row["costo"], "costo_prueba_ic_inf": model_row["costo_ic_inf"], "costo_prueba_ic_sup": model_row["costo_ic_sup"],
    "tarde_prueba": model_row["tarde"], "cobertura_prueba": model_row["cobertura"], "mae_plazo_prueba": model_row["mae"],
    "dif_costo_vs_plazo_fijo": model_row["dif_costo"], "dif_ic_inf": model_row["dif_ic_inf"], "dif_ic_sup": model_row["dif_ic_sup"],
    "costo_plazo_fijo_prueba": baseline_row["costo"],
}
tags = {"etapa": "08_registro", "datos": "train.parquet", "congelado": frozen["fecha_hora"], "prueba_abierta": results["prueba_abierta"]}
fallback_reason = None
try:
    info = register_final_model(pipeline, params, metrics, example, REGISTERED_NAME, tags=tags)
except (MlflowException, OSError) as error:  # permiso, cuota o conexión: respaldo local con el motivo (regla 4); otro error se detiene
    fallback_reason = f"{type(error).__name__}: {str(error).splitlines()[0][:200]}"
    print("⚠️ el registro en Databricks falló; se usa el respaldo local. Motivo:", fallback_reason)
    configure_mlflow(force_local=True)
    set_experiment(EXPERIMENT)
    PLATFORM = "local (respaldo)"
    info = register_final_model(pipeline, params, metrics, example, REGISTERED_NAME, tags=tags)
print(f"registrado: {info['nombre']} · versión {info['version']} · plataforma: {PLATFORM}")'''),

md('''## 3. Poner el alias `champion`'''),

code('''champion_version = set_champion(info["nombre"], version=info["version"])
assert champion_version == info["version"], "el champion no apunta a la versión recién registrada"
print(f"alias champion → versión {champion_version}")'''),

md('''## 4. Comprobación de ida y vuelta

Se carga `models:/<nombre>@champion` y se compara con el modelo local: plazos de 500 pedidos de la prueba (semilla 42) y costo 3 a 1 en toda la
prueba contra el de la etapa 7. La prueba se lee aquí **solo** para esto (decisión del grupo); no cambia nada del modelo.'''),

code('''recovered = load_champion(info["nombre"])
test = add_features(pd.read_parquet(PROCESSED_DIR / "test.parquet"))
X_test = test.drop(columns=[TARGET])
sample = X_test.sample(n=500, random_state=42)
same_plans = bool(np.allclose(recovered.predict(sample), pipeline.predict(sample)))
cost_registry = mean_delivery_cost(test[TARGET], recovered.predict(X_test))
cost_stage7 = model_row["costo"]
matches = bool(np.isclose(cost_registry, cost_stage7, rtol=0, atol=1e-6))
print(f"plazos iguales en 500 pedidos: {same_plans} | costo con el modelo del registro: {cost_registry:.6f} | etapa 7: {cost_stage7:.6f} | coincide: {matches}")
assert same_plans and matches, "el modelo registrado no es el que se evaluó"'''),

md('''## 5. Guardar la constancia'''),

code('''record = {
    "plataforma": PLATFORM, "motivo_respaldo": fallback_reason, "experimento": EXPERIMENT, "nombre": info["nombre"],
    "version": info["version"], "alias": "champion", "run_id": info["run_id"], "fecha": datetime.now().astimezone().isoformat(timespec="seconds"),
    "ida_y_vuelta": {"plazos_iguales_500": same_plans, "costo_registro": cost_registry, "costo_etapa_7": cost_stage7, "coincide": matches},
}
(REPORTS_DIR / "08_registro.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
print("guardado: reports/08_registro.json")'''),

md('''## Qué sigue

Etapa 9: exportar `models/model.joblib` **desde el champion**, escribir `models/input_schema.json`, adaptar `src/olistbr/inference/predict.py`
a regresión (días y plazo) y la app de Streamlit, que **no consulta Databricks**.'''),
]
nbf.write(nb, "notebooks/08_registro_modelo.ipynb")
print("ok", len(nb.cells))
