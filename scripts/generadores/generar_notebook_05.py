import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

nb.cells = [
code('''import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
# El aviso de «pocos clientes de calibración» es una red de seguridad: no se silencia
warnings.filterwarnings("default", message=".*clientes de calibración.*")
# Permite importar el paquete de src/ sin instalarlo
sys.path.insert(0, str(Path.cwd().resolve().parent / "src"))'''),

md('''# 05 · Torneo de modelos — descartar con números, no coronar

Especificación: `especificaciones/05-torneo.md` · Plan: `specs/003-model-tournament/plan.md`.
Este notebook **solo orquesta**: llama a funciones de `src/olistbr/` y muestra resultados.

**Qué se decide aquí.** El torneo no busca un campeón: busca **1 o 2 finalistas** para optimizar en la etapa siguiente.
Todos los candidatos usan **los mismos 5 pliegues por cliente** de la etapa 4 y la misma **prueba hacia adelante**
(entrenar hasta 2018-02-28, validar de marzo a agosto de 2018).

**Qué gana.** El **costo 3 a 1** del plazo recomendado: subestimar cuesta 3 veces más que sobrestimar, lo que equivale a 4 veces la
pérdida cuantílica de 0,75; por eso el plazo óptimo es el **percentil 75**. El MAE, el RMSE y el R² se reportan siempre.

**Dos familias de modelos.**
- **A (valor típico + margen):** el modelo predice un valor y se le suma un margen = percentil 75 de los errores en **clientes de
  calibración** apartados dentro de cada entrenamiento (nunca de la validación).
- **B (percentil 75 directo):** el modelo ya predice el plazo; no lleva margen.

⚠️ El MAE, RMSE y R² de la familia B **no son comparables** con los de la A (su predicción es el percentil 75, más alto a propósito).
Entre familias se compara **solo el costo**.

**Reglas de esta etapa:** solo se usa el conjunto de entrenamiento (la prueba final queda cerrada); sin la marca de fin de año ni el
historial por vendedor; el plazo prometido por Olist nunca entra.'''),

code('''import pandas as pd

from olistbr.config import FORWARD_TRAIN_END, FORWARD_VALID_END, FORWARD_VALID_START, N_FOLDS, PROCESSED_DIR, REPORTS_DIR, TARGET, TOURNAMENT_GROUPS
from olistbr.data.split import forward_split, make_group_folds
from olistbr.features.engineering import add_features
from olistbr.models.evaluation import (
    compare_models, coverage_by_month, evaluate_candidate_folds, evaluate_candidate_holdout, per_thousand, regional_metrics, tournament_table,
)
from olistbr.models.train import candidates

pd.set_option("display.width", 230)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_colwidth", 60)

BASELINES = {"mediana": "Línea base: siempre la mediana", "plazo_fijo": "Línea base: plazo fijo (percentil 75)"}
assert not {"peak_season", "history"} & set(TOURNAMENT_GROUPS), "el torneo no usa la marca de fin de año ni el historial"
print("grupos de variables del torneo:", ", ".join(TOURNAMENT_GROUPS))'''),

md('''## 1. Datos, pliegues y prueba hacia adelante

Se leen los datos de entrenamiento, se calculan las variables derivadas **una vez** y se crean los pliegues **una vez**.'''),

code('''train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
df = add_features(train)
X, y = df.drop(columns=[TARGET]), df[TARGET]

folds = make_group_folds(df)          # verifica que ningún cliente cae en los dos lados de un pliegue
assert sorted(i for _, valid_idx in folds for i in valid_idx) == list(range(len(df))), "los pliegues no cubren cada fila una vez"
fwd_train, fwd_valid = forward_split(df)

print(f"entrenamiento: {len(df):,} pedidos · {df['customer_unique_id'].nunique():,} clientes · {N_FOLDS} pliegues de {[len(v) for _, v in folds]}")
print(f"hacia adelante: entrenamiento {len(fwd_train):,} (hasta {FORWARD_TRAIN_END}) · validación {len(fwd_valid):,} ({FORWARD_VALID_START} a {FORWARD_VALID_END})")
print("percentil 75 de los días en entrenamiento (el «mejor plazo fijo»):", round(y.quantile(0.75), 2))'''),

md('''## 2. Correr el torneo

Cada candidato se ajusta en los 5 pliegues y en la partición hacia adelante. Se guarda **cada pliegue**, no solo el promedio.'''),

code('''frames, forward_rows = [], []
for name, candidate in candidates().items():
    frames.append(evaluate_candidate_folds(name, candidate, X, y, folds))
    forward_rows.append(evaluate_candidate_holdout(name, candidate, fwd_train, fwd_valid))
    mean_fit = frames[-1]["fit_seconds"].mean()
    print(f"{name:42s} familia {candidate.family:4s} listo  (ajuste medio {mean_fit:5.1f} s)")

by_fold = pd.concat(frames, ignore_index=True)
forward = pd.DataFrame(forward_rows)
len(by_fold), len(forward)'''),

md('''## 3. Resultados: la tabla del torneo

Ordenada por **costo 3 a 1** (media entre los 5 pliegues; menor es mejor). `supera_*` dice si el modelo vence a cada línea base por más
que su ruido; `empate_con_el_mejor` marca los que no se distinguen del mejor.'''),

code('''table = tournament_table(by_fold, forward, BASELINES)
business = table[["modelo", "familia", "costo_media", "costo_desv", "adelante_costo", "tarde_media", "cobertura_media", "margen_media",
                  "posicion_costo", "supera_mediana", "supera_plazo_fijo", "mejor", "empate_con_el_mejor"]]
business.round(3)'''),

md('''## 4. Exactitud: MAE, RMSE, R² y sesgo

Se comparan **dentro de cada familia**. La familia B predice el percentil 75, así que su MAE y su R² salen peores a propósito.'''),

code('''accuracy = table[["modelo", "familia", "mae_media", "mae_desv", "rmse_media", "r2_media", "sesgo_media", "adelante_mae",
                  "posicion_mae", "posicion_costo"]]
accuracy.sort_values(["familia", "mae_media"]).round(3)'''),

md('''## 5. Comparación por pares contra el mejor modelo

Pliegue por pliegue: ¿en cuántos gana el mejor y cuánto vale la diferencia frente a su ruido? Es **consistente** si gana en al menos
4 de 5 y la diferencia supera 1 vez su ruido; si no, es un **empate práctico** y gana el más simple y rápido.'''),

code('''best = table.loc[table["mejor"], "modelo"].iloc[0]
print("mejor modelo por costo (sin contar las líneas base):", best)
table.loc[~table["mejor"], ["modelo", "gana_vs_mejor", "ratio_vs_mejor", "consistente_vs_mejor", "empate_con_el_mejor"]].round(2)'''),

md('''## 6. La regresión lineal contra el ganador

La regresión lineal es el modelo del caso de negocio y siempre compite. La diferencia se expresa en **unidades de costo** por pedido y por cada 1.000 pedidos
(1 unidad de costo = un día de sobrestimación). **No se le asigna un valor en dinero.** El MAE de la familia A y el de la B no se comparan entre sí.'''),

code('''LINEAR = "Regresión lineal"
costs = by_fold.pivot(index="pliegue", columns="modelo", values="costo")
versus = compare_models(costs[best].to_numpy(), costs[LINEAR].to_numpy())
row = table.set_index("modelo")
cost_gap = row.loc[LINEAR, "costo_media"] - row.loc[best, "costo_media"]
print(f"{LINEAR} frente a {best}:")
print(f"  costo: {cost_gap:+.3f} por pedido  ->  {per_thousand(cost_gap):+.0f} unidades de costo por cada 1.000 pedidos")
print(f"  el mejor gana en {versus['texto']} pliegues · diferencia / ruido = {versus['ratio']:.2f} · consistente: {versus['consistente']}")
'''),

md('''## 7. Revisión por región del cliente

Para **todos los modelos** (no solo los de arriba): MAE y porcentaje de pedidos que llega tarde por región, con predicciones
**fuera de muestra** (principio VIII: dónde rinde peor). El MAE solo se compara dentro de cada familia.'''),

code('''region_models = [m for m in table["modelo"] if m not in BASELINES.values()]
all_candidates = candidates()
regions = pd.concat([regional_metrics(m, all_candidates[m], X, y, folds) for m in region_models], ignore_index=True)
regions.pivot(index="grupo", columns="modelo", values="tarde").round(3)'''),

code('''regions.pivot(index="grupo", columns="modelo", values="mae").round(3)'''),

md('''## 8. ¿Se mantiene la calibración en el tiempo? Cobertura mes a mes hacia adelante

En los pliegues los plazos cubren ≈ 75 % de los pedidos. En la prueba hacia adelante (marzo a agosto de 2018), ¿cubren lo mismo cada mes?
Un plazo aprendido del pasado puede quedar corto en unos meses y largo en otros si cambia el ritmo de entrega.'''),

code('''monthly_models = [best, "HistGradientBoosting cuantil 0,75", "LightGBM", LINEAR]
monthly = pd.concat([coverage_by_month(m, all_candidates[m], fwd_train, fwd_valid) for m in monthly_models], ignore_index=True)
monthly.pivot(index="mes", columns="modelo", values="cobertura").round(3)'''),

code('''# Qué pasó con los días reales de entrega mes a mes en la validación (iguales para todos los modelos)
monthly[monthly["modelo"] == best][["mes", "n_pedidos", "mediana_dias", "p75_dias"]].round(2)'''),

md('''## 9. Guardar las tablas

`reports/05_torneo_por_pliegue.csv` (cada modelo en cada pliegue y en la partición hacia adelante), `05_torneo_resumen.csv`,
`05_torneo_regiones.csv`, `05_torneo_cobertura_mensual.csv` y `05_torneo_tiempos.csv` (el tiempo de ajuste cambia entre ejecuciones y va aparte). El texto con el
motivo de cada decisión está en `reports/05_torneo.md`.'''),

code('''REPORTS_DIR.mkdir(parents=True, exist_ok=True)
per_fold = pd.concat([by_fold.drop(columns="fit_seconds"), forward.drop(columns="fit_seconds")], ignore_index=True)
times = pd.concat([by_fold[["modelo", "pliegue", "fit_seconds"]], forward[["modelo", "pliegue", "fit_seconds"]]], ignore_index=True)

per_fold.round(6).to_csv(REPORTS_DIR / "05_torneo_por_pliegue.csv", index=False)
table.round(6).to_csv(REPORTS_DIR / "05_torneo_resumen.csv", index=False)
regions.round(6).to_csv(REPORTS_DIR / "05_torneo_regiones.csv", index=False)
monthly.round(6).to_csv(REPORTS_DIR / "05_torneo_cobertura_mensual.csv", index=False)
times.round(3).to_csv(REPORTS_DIR / "05_torneo_tiempos.csv", index=False)
print(f"guardado: {len(per_fold)} filas por pliegue (10 candidatos x 6) · {len(table)} en el resumen · {len(regions)} en regiones · {len(monthly)} en cobertura mensual")'''),

md('''## 10. Las pruebas automáticas

Desde la raíz del repo:

```bash
pytest tests
```

Candidatos y modelo de plazo en `tests/test_models.py`; métricas en `tests/test_metrics.py`; evaluación, comparación y tabla del torneo en `tests/test_evaluation.py`.'''),
]
nbf.write(nb, "notebooks/05_torneo.ipynb")
print("ok", len(nb.cells))
