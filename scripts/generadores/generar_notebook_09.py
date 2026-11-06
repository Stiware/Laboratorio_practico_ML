"""Escribe notebooks/09_inferencia.ipynb (etapa 9): solo orquesta funciones de src/olistbr/. Se ejecuta desde la raíz del repo.

⚠️ No volver a correr este generador sobre un notebook ya ejecutado sin guardar antes sus salidas: las borra.
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
os.environ["MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT"] = "true"  # sin enlaces del workspace en las salidas (regla 2)
# Permite importar el paquete de src/ sin instalarlo
sys.path.insert(0, str(Path.cwd().resolve().parent / "src"))'''),

md('''# 09 · Inferencia — del modelo registrado a una fecha de entrega recomendada

Especificación: `especificaciones/09-inferencia-app.md` · Plan: `specs/006-inference-app/plan.md`.
Este notebook **solo orquesta**: la lógica está en `src/olistbr/inference/`, con pruebas.

Un modelo entrenado no es un producto. Un producto recibe datos **incompletos o mal escritos de una persona real** y responde con algo
útil o con un **error claro**. Aquí: (1) se exporta el modelo `champion` a `models/model.joblib`; (2) se escribe el esquema del formulario;
(3) se prueban tres pedidos (normal, con «No sé» y con un dato imposible); (4) se comprueba que el archivo exportado es el modelo evaluado,
**sin volver a leer la prueba**; (5) se revisa la ficha del modelo. La app (`app/streamlit_app.py`) usa esos archivos y **no consulta Databricks**.'''),

code('''import json

import joblib
import numpy as np
import pandas as pd

from olistbr.config import MODELS_DIR, PROCESSED_DIR, RANDOM_STATE, REPORTS_DIR, TARGET
from olistbr.features.engineering import add_features
from olistbr.config import OTHER_LABEL
from olistbr.inference.predict import InputValidationError, order_to_features, predict_deadline
from olistbr.inference.schema import FORM_FIELDS, build_schema
from olistbr.tracking.registry import model_columns

registry = json.loads((REPORTS_DIR / "08_registro.json").read_text(encoding="utf-8"))
REGISTERED_NAME = registry["nombre"]
print("modelo registrado:", REGISTERED_NAME, "· versión", registry["version"], "· alias", registry["alias"])'''),

md('''## 1. Exportar el modelo `champion`

La fuente de verdad es el **registro**. Si Databricks no conecta, se usa la copia local de la etapa 7 (`final_pipeline.joblib`) y el notebook lo dice.'''),

code('''from mlflow.exceptions import MlflowException

from olistbr.tracking.mlflow_utils import configure_mlflow, uses_databricks
from olistbr.tracking.registry import load_champion

logging.getLogger("mlflow").setLevel(logging.WARNING)  # después de importar MLflow: sin mensajes con la ruta del experimento
try:
    configure_mlflow()
    if not uses_databricks():
        raise MlflowException("MLflow quedó en modo local: el champion está en Databricks")
    model = load_champion(REGISTERED_NAME)
    SOURCE = f"registro (champion, versión {registry['version']})"
except (MlflowException, OSError) as error:
    model = joblib.load(MODELS_DIR / "final_pipeline.joblib")
    SOURCE = f"respaldo local final_pipeline.joblib ({type(error).__name__})"
joblib.dump(model, MODELS_DIR / "model.joblib")
size_mb = (MODELS_DIR / "model.joblib").stat().st_size / 1e6
print(f"exportado models/model.joblib desde: {SOURCE} · {size_mb:.1f} MB")'''),

md('''## 2. El esquema de entrada

Qué pregunta la app (11 datos del pedido), qué valores son **posibles** (decisión del grupo, Q3) y qué conoce el modelo (estados, categorías,
rangos observados). También guarda el % de pedidos tarde de la prueba, en total y por región, que la app muestra como referencia.
Se arma **solo con el entrenamiento** y con los resultados ya guardados de la etapa 7.'''),

code('''train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
results = json.loads((REPORTS_DIR / "07_metricas_test.json").read_text(encoding="utf-8"))
segments = pd.read_csv(REPORTS_DIR / "07_segmentos.csv")
schema = build_schema(train, test_results=results, segments=segments, path=MODELS_DIR / "input_schema.json")
print(f"esquema guardado: {len(schema['fields'])} campos · {len(schema['states'])} estados · {len(schema['categories'])} categorías")
print("periodo de entrenamiento:", schema["observed_period"], "· % tarde en la prueba:", f"{schema['late_share_total']:.1%}")
pd.DataFrame({"posible": schema["valid_range"], "observado": schema["observed_range"], "mediana": schema["median"]})'''),

md('''## 3. Probar la predicción

Tres casos: un pedido **normal**, uno **incompleto** (la persona marcó «No sé» en la distancia y la categoría) y uno con un dato **imposible**.
Lo esperado: el primero responde; el segundo también, con advertencias; el tercero **se niega y explica qué corregir**.'''),

code('''normal = pd.DataFrame([{
    "order_purchase_timestamp": "2018-05-14 10:30", "seller_state": "SP", "customer_state": "BA", "distance_km": 1450.0,
    "product_category_name": "health_beauty", "n_items": 1, "main_seller_items": 1, "price_sum": 120.0, "freight_sum": 25.0,
    "weight_g_sum": 900.0, "volume_cm3_sum": 6000.0,
}])
print("— pedido normal —")
print(predict_deadline(normal, model, schema).to_string())

incomplete = normal.assign(distance_km=np.nan, product_category_name=None)
print("\\n— pedido con «No sé» en la distancia y la categoría —")
result = predict_deadline(incomplete, model, schema)
print(result.drop(columns="advertencias").to_string())
for message in result.attrs["warnings"]:
    print("  advertencia:", message)

print("\\n— pedido imposible (distancia de 9.000 km y 3 ítems del vendedor principal en un pedido de 2) —")
try:
    predict_deadline(normal.assign(distance_km=9000.0, n_items=2, main_seller_items=3), model, schema)
except InputValidationError as error:
    print("Error claro:", error)'''),

md('''## 4. Comprobar que el archivo exportado es el modelo que se evaluó

Sin volver a leer la prueba (ya se leyó para medir en la etapa 7 y para la ida y vuelta del registro en la etapa 8): `models/model.joblib` debe dar
**los mismos plazos** que `models/final_pipeline.joblib` (el modelo evaluado) en 1.000 pedidos del entrenamiento. Se guarda esa muestra en
`data/processed/` (no se versiona) para la comprobación en Python 3.12 con `scripts/verificar_modelo.py`.'''),

code('''sample = add_features(train)[model_columns()].sample(n=1000, random_state=RANDOM_STATE)  # solo las 25 columnas del modelo: sin identificadores
evaluated = joblib.load(MODELS_DIR / "final_pipeline.joblib")
exported = joblib.load(MODELS_DIR / "model.joblib")
same = bool(np.allclose(exported.predict(sample), evaluated.predict(sample), rtol=0, atol=1e-9))
print(f"plazos iguales en 1.000 pedidos del entrenamiento: {same}")
assert same, "el modelo exportado no es el que se evaluó"
sample.to_parquet(PROCESSED_DIR / "muestra_verificacion.parquet")
print("muestra guardada para la comprobación en Python 3.12: data/processed/muestra_verificacion.parquet (solo columnas del modelo)")

# El camino del formulario (`order_to_features`) da lo mismo que el del entrenamiento (`add_features`) con el modelo real,
# en pedidos del entrenamiento cuyo vendedor es «otros» (lo que la app supone para todos)
others = train[train["seller_id"] == OTHER_LABEL].sample(n=1000, random_state=RANDOM_STATE)
via_form = exported.predict(order_to_features(others[FORM_FIELDS]))
via_training = exported.predict(add_features(others))
same_path = bool(np.allclose(via_form, via_training, rtol=0, atol=1e-9))
print(f"formulario y entrenamiento dan el mismo plazo en 1.000 pedidos con vendedor «otros»: {same_path}")
assert same_path, "la conversión del formulario no reproduce lo que vio el modelo"'''),

md('''## 5. La ficha del modelo

`models/ficha_modelo.md`: qué es, con qué datos, qué rendimiento tiene en la prueba y **qué no hay que hacer con él**. Se completa con las cifras
de `reports/07_validacion.md` y `08_registro.md`.'''),

code('''card = (MODELS_DIR / "ficha_modelo.md").read_text(encoding="utf-8")
blanks = card.count("_______________")
print(f"ficha_modelo.md: {blanks} campos en blanco" if blanks else "ficha_modelo.md completa")'''),

md('''## 6. La app

```bash
streamlit run app/streamlit_app.py
```

Comprueba que el formulario aparece, que un pedido normal da una fecha, que «No sé» no rompe nada y que un dato imposible muestra un error claro
(lo mismo prueban automáticamente `tests/test_app.py` y `tests/test_inference.py`).'''),
]
nbf.write(nb, "notebooks/09_inferencia.ipynb")
print("ok", len(nb.cells))
