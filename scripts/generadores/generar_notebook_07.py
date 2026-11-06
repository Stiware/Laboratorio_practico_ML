"""Escribe notebooks/07_validacion_interpretacion.ipynb (etapa 7): solo orquesta funciones de src/olistbr/. Se ejecuta desde la raíz.

⚠️ El notebook generado ABRE LA PRUEBA (una sola vez). No volver a correr este generador sobre un notebook ya ejecutado sin guardar antes
sus salidas: las borra.
"""
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
os.environ["MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT"] = "true"  # sin enlaces del workspace en las salidas
# Permite importar el paquete de src/ sin instalarlo
sys.path.insert(0, str(Path.cwd().resolve().parent / "src"))'''),

md('''# 07 · Validación e interpretación — abrir la prueba, una sola vez

Especificación: `especificaciones/07-validacion.md` · Plan: `specs/005-final-validation-registry/plan.md`.
Este notebook **solo orquesta**: la lógica está en `src/olistbr/models/validation.py`, con pruebas.

**Qué se mide aquí.** Cuánto vale de verdad el modelo elegido en la etapa 6 (LightGBM cuantil 0,75 optimizado) en pedidos de
**clientes que nunca vio**, frente a reglas simples. Es la cifra honesta del proyecto: todo lo anterior fue validación.

**Reglas de esta etapa:** todo se decide **antes** de abrir la prueba (sección 1); la prueba se abre **una sola vez** y queda constancia;
**nada del modelo se cambia después de verla** (si algo no gusta, se anota como límite); los intervalos sortean **clientes completos**.'''),

code('''import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from olistbr.config import (
    BOOTSTRAP_ROUNDS, COST_RATIOS, GROUP_COL, MODELS_DIR, PROCESSED_DIR, RANDOM_STATE, REPORTS_DIR, TARGET, TOURNAMENT_GROUPS,
)
from olistbr.features.engineering import add_features
from olistbr.models.metrics import mae_days, mean_delivery_cost
from olistbr.models.train import build_pipeline, fit_deadline_model
from olistbr.models.tuning import n_trees_used, tuned_candidate
from olistbr.models.validation import (
    MAIN_BASELINE, baseline_plans, cost_sensitivity, open_test_once, segment_table, slack_and_delay, test_report, top_shap,
)

pd.set_option("display.width", 230)
pd.set_option("display.max_columns", 40)

# Rutas de esta etapa en un solo lugar (el ensayo sin la prueba las cambia todas aquí)
TEST_PATH = PROCESSED_DIR / "test.parquet"
EXTREME_PATH = PROCESSED_DIR / "extreme_orders.parquet"
MARK_PATH = REPORTS_DIR / "prueba_abierta.txt"
OUT_DIR = REPORTS_DIR
MODEL_PATH = MODELS_DIR / "final_pipeline.joblib"
MODEL_NAME = "LightGBM cuantil 0,75"'''),

md('''## 1. Decisiones congeladas — **antes** de abrir la prueba

Todas salen de `especificaciones/07-validacion.md` (cerrada el 5 nov 2026) y de `reports/06_best_params.json`. Se guardan con fecha y hora
en `07_decisiones_congeladas.json` **antes** de la celda que abre la prueba.'''),

code('''best = json.loads((REPORTS_DIR / "06_best_params.json").read_text(encoding="utf-8"))
assert best["eleccion"]["modelo"] == MODEL_NAME, "el modelo elegido en la etapa 6 no es el que congela esta especificación"
PARAMS = best["finalistas"][MODEL_NAME]["parametros_elegidos"]

FROZEN = {
    "fecha_hora": datetime.now().astimezone().isoformat(timespec="seconds"),
    "modelo": MODEL_NAME,
    "parametros": PARAMS,
    "decision_parametros": best["finalistas"][MODEL_NAME]["comprobacion"]["decision"],
    "recalibracion": "no: la prueba se partió por cliente, no por fecha; la recalibración es una regla de uso (vigilar la cobertura mensual)",
    "metrica_principal": "costo 3 a 1 del plazo recomendado (subestimar cuesta 3 veces más)",
    "regla": "el plazo recomendado es el percentil 75 que predice el modelo (sin margen)",
    "baseline_principal": MAIN_BASELINE,
    "baselines_de_referencia": ["Siempre la mediana", "Plazo prometido por Olist"],
    "criterio": "el modelo mejora con confianza si todo el intervalo del 95 % de costo(modelo) − costo(baseline principal) está bajo cero",
    "intervalos": f"{BOOTSTRAP_ROUNDS} sorteos de clientes completos, semilla {RANDOM_STATE}",
    "segmentos": ["región del cliente", "mes de compra"],
    "interpretacion": "SHAP sobre 2.000 pedidos de la prueba; ubicación del cliente = atributo sensible",
    "no_evaluado": ["marca de fin de año (F6)", "historial por vendedor (F8)", "probabilidad de retraso por pedido"],
}
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "07_decisiones_congeladas.json").write_text(json.dumps(FROZEN, ensure_ascii=False, indent=2), encoding="utf-8")
for key, value in FROZEN.items():
    print(f"{key}: {value}")'''),

md('''## 2. ¿Hay que calibrar?

OnTrack no da probabilidades: da un **plazo**. El equivalente de «calibrar» es que el plazo cubra ≈ 75 % de los pedidos. En la reserva de la
etapa 6 cubrió el 73,9 %; hacia adelante, la cobertura cambió mes a mes (64 % a 92 %). **Decisión (Q2):** no se recalibra el modelo, porque
la prueba se partió por **cliente** (mezcla todos los meses) y no puede evaluar una recalibración en el tiempo; se mide la cobertura mes a mes
en la prueba y la recalibración queda como **regla de uso** para la ficha y la app.'''),

md('''## 3. Entrenar el modelo final con **todo** el entrenamiento

Mismas piezas que la etapa 6 (`tuned_candidate` + `fit_deadline_model`, familia B: sin margen), con los 76.924 pedidos de entrenamiento.'''),

code('''train = add_features(pd.read_parquet(PROCESSED_DIR / "train.parquet"))
X_train, y_train = train.drop(columns=[TARGET]), train[TARGET]
candidate = tuned_candidate(MODEL_NAME, PARAMS)
final = fit_deadline_model(build_pipeline(TOURNAMENT_GROUPS, model=candidate.model, scale=candidate.scale), candidate.family,
                           X_train, y_train, X_train[GROUP_COL])
assert final.margin == 0.0  # familia B: el modelo ya predice el percentil 75
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(final.pipeline, MODEL_PATH)
print(f"modelo final ajustado con {len(train):,} pedidos | árboles: {n_trees_used(final)} | guardado en {MODEL_PATH.name}")'''),

md('''## 4. Abrir la prueba — **una sola vez**

`open_test_once` deja la fecha y hora en `reports/prueba_abierta.txt` y **se niega** a abrirla de nuevo si esa constancia ya existe.'''),

code('''test = add_features(open_test_once(TEST_PATH, MARK_PATH))
X_test, real = test.drop(columns=[TARGET]), test[TARGET].to_numpy(dtype=float)
print(f"prueba abierta: {len(test):,} pedidos de {test[GROUP_COL].nunique():,} clientes | constancia: {MARK_PATH.read_text(encoding='utf-8').strip()}")
assert set(test[GROUP_COL]).isdisjoint(set(train[GROUP_COL])), "la prueba comparte clientes con el entrenamiento"'''),

md('''## 5. Resultado en la prueba, con intervalos por cliente

Costo 3 a 1, % tarde y cobertura de cada regla con su intervalo del 95 % (sorteando clientes). El MAE y el sesgo son del **plazo**
(percentil 75, más alto que el valor típico a propósito). Diferencia contra el plazo fijo del percentil 75: negativa = el modelo cuesta menos.'''),

code('''plans = {MODEL_NAME: final.recommend(X_test), **baseline_plans(train, test)}
groups = test[GROUP_COL].to_numpy()
report = test_report(real, plans, groups, main_baseline=MAIN_BASELINE, model_name=MODEL_NAME)
report.round(4)'''),

code('''row = report.set_index("regla").loc[MODEL_NAME]
base = report.set_index("regla").loc[MAIN_BASELINE]
print(f"costo del modelo: {row['costo']:.3f} [{row['costo_ic_inf']:.3f} ; {row['costo_ic_sup']:.3f}] | "
      f"{MAIN_BASELINE}: {base['costo']:.3f} [{base['costo_ic_inf']:.3f} ; {base['costo_ic_sup']:.3f}]")
print(f"diferencia (modelo − {MAIN_BASELINE}): {row['dif_costo']:+.3f} [{row['dif_ic_inf']:+.3f} ; {row['dif_ic_sup']:+.3f}]")
print("CONCLUSIÓN DE LA REGLA:", "el modelo mejora al plazo fijo con confianza (todo el intervalo bajo cero)" if row["mejora_con_confianza"]
      else "no hay evidencia de que el modelo mejore al plazo fijo (el intervalo cruza o toca el cero)")
print(f"cobertura del modelo: {row['cobertura']:.1%} [{row['cobertura_ic_inf']:.1%} ; {row['cobertura_ic_sup']:.1%}] (esperado ≈ 75 %)")'''),

md('''## 6. Errores por segmento, holgura y atraso

En lugar de «carga de trabajo» (aquí no hay revisión humana): los **días de holgura** (cuánto sobra el plazo cuando el pedido llega a tiempo)
y los **días de atraso** (cuánto falta cuando llega tarde). Segmentos: región del cliente y mes de compra.'''),

code('''months = test["order_purchase_timestamp"].dt.to_period("M").astype(str).to_numpy()
segments = segment_table(real, plans[MODEL_NAME], {"region": test["customer_region"].to_numpy(dtype=object), "mes": months})
segments[segments.segmento == "region"].round(3)'''),

code('''segments[segments.segmento == "mes"].round(3)'''),

code('''slack = pd.DataFrame([{"regla": name, **slack_and_delay(real, plan)} for name, plan in plans.items()])
slack.round(3)'''),

code('''# Sensibilidad del costo a la proporción subestimar : sobrestimar (el plazo del modelo es siempre el percentil 75)
sensitivity = cost_sensitivity(real, plans, ratios=COST_RATIOS)
sensitivity.round(3)'''),

code('''# Escenario pesimista: la prueba más los 64 pedidos de más de 100 días (separado del resultado principal; sin intervalo)
extreme = add_features(pd.read_parquet(EXTREME_PATH))
real_all = np.r_[real, extreme[TARGET].to_numpy(dtype=float)]
plans_all = {
    MODEL_NAME: np.r_[plans[MODEL_NAME], final.recommend(extreme.drop(columns=[TARGET]))],
    MAIN_BASELINE: np.r_[plans[MAIN_BASELINE], np.full(len(extreme), plans[MAIN_BASELINE][0])],
}
with_extremes = pd.DataFrame([
    {"regla": name, "pedidos": len(real_all), "costo": mean_delivery_cost(real_all, plan), "tarde": float(np.mean(real_all > plan)),
     "mae": mae_days(real_all, plan)}
    for name, plan in plans_all.items()
])
with_extremes.round(3)'''),

md('''## 7. ¿Qué mueve el plazo? (SHAP)

Valores SHAP sobre 2.000 pedidos de la prueba (semilla 42). **Límites:** explican al modelo, no al mundo (no dicen si acertó); las variables
correlacionadas se reparten el crédito (por ejemplo, `distance_km` y `log_distance`); una importancia alta no es una causa.
**Atributo sensible:** la ubicación del cliente (`customer_state`, `customer_region`) puede hacer que el plazo trate distinto a cada región.'''),

code('''shap_table = top_shap(final.pipeline, X_test, n=10, sample=2000, seed=RANDOM_STATE)
shap_table.round(3)'''),

md('''## 8. Guardar los resultados de la etapa

`reports/07_metricas_test.json` (decisiones congeladas, tabla de la prueba, holgura y atraso, sensibilidad, escenario con extremos),
`reports/07_segmentos.csv` y `reports/07_shap.csv`. No se guardan los plazos pedido a pedido. El texto se redacta en `reports/07_validacion.md`.'''),

code('''def records(frame):
    """Filas de una tabla como diccionarios, con los vacíos como null en el JSON."""
    return json.loads(frame.to_json(orient="records", force_ascii=False))

summary = {
    "decisiones_congeladas": FROZEN,
    "prueba_abierta": MARK_PATH.read_text(encoding="utf-8").strip(),
    "pedidos_prueba": len(test), "clientes_prueba": int(test[GROUP_COL].nunique()),
    "arboles": n_trees_used(final),
    "resultado": records(report),
    "mejora_con_confianza": bool(row["mejora_con_confianza"]),
    "holgura_y_atraso": records(slack),
    "sensibilidad_costo": records(sensitivity),
    "con_extremos": records(with_extremes),
}
(OUT_DIR / "07_metricas_test.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
segments.round(6).to_csv(OUT_DIR / "07_segmentos.csv", index=False)
shap_table.round(6).to_csv(OUT_DIR / "07_shap.csv", index=False)
print("guardado: 07_metricas_test.json, 07_segmentos.csv, 07_shap.csv")'''),
]
nbf.write(nb, "notebooks/07_validacion_interpretacion.ipynb")
print("ok", len(nb.cells))
