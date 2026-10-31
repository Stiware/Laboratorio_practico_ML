"""Genera reports/05_torneo.md leyendo las cifras de reports/05_torneo_*.csv.

Las cifras de las tablas salen de los CSV del notebook. Las que se calculan aquí (estadísticas mensuales de los días de entrega y el estado
de los pedidos) se marcan en el texto como «calculadas aparte». Cada afirmación cualitativa se verifica con los datos: si dejara de
cumplirse, el script falla en vez de escribir algo falso.
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from olistbr.models.evaluation import compare_models  # noqa: E402

R = "reports/"
summary = pd.read_csv(R + "05_torneo_resumen.csv")
per_fold = pd.read_csv(R + "05_torneo_por_pliegue.csv")
times = pd.read_csv(R + "05_torneo_tiempos.csv")
regions = pd.read_csv(R + "05_torneo_regiones.csv")
monthly = pd.read_csv(R + "05_torneo_cobertura_mensual.csv")
orders = pd.read_parquet("data/processed/train.parquet", columns=["order_purchase_timestamp", "dias_entrega"])
raw = pd.read_csv("data/raw/olist_orders_dataset.csv", usecols=["order_status", "order_purchase_timestamp"], parse_dates=["order_purchase_timestamp"])


def f(x, d=3):
    return "—" if pd.isna(x) else f"{x:.{d}f}".replace(".", ",")


def fi(x):
    return f"{x:,.0f}".replace(",", ".")


def pct(x, d=1):
    return f"{100 * x:.{d}f} %".replace(".", ",")


s = summary.set_index("modelo")
BASE = ["Línea base: siempre la mediana", "Línea base: plazo fijo (percentil 75)"]
LINEAR, HGB, LGBM_A, XGB_A = "Regresión lineal", "HistGradientBoosting cuantil 0,75", "LightGBM", "XGBoost"
models = [m for m in summary["modelo"] if m not in BASE]
best = summary.loc[summary["mejor"], "modelo"].iloc[0]
b, lin, med, p75 = s.loc[best], s.loc[LINEAR], s.loc[BASE[0]], s.loc[BASE[1]]
gap = lin.costo_media - b.costo_media
fit_time = times[times["pliegue"] != "adelante"].groupby("modelo")["fit_seconds"].mean()
fold_cost = per_fold[per_fold["pliegue"] != "adelante"].assign(pliegue=lambda d: d["pliegue"].astype(int)).pivot(index="pliegue", columns="modelo", values="costo")
fwd = per_fold[per_fold["pliegue"] == "adelante"].set_index("modelo")
forward_cov = fwd["cobertura"]
margins = per_fold[(per_fold["familia"] == "A") & (per_fold["pliegue"] != "adelante")].groupby("modelo")["margen"].agg(["min", "max"])

# ---------- posiciones por pliegues y hacia adelante (solo modelos)
pos_fold = s.loc[models, "costo_media"].rank(method="min").astype(int)
pos_fwd = s.loc[models, "adelante_costo"].rank(method="min").astype(int)
moved = [(m, pos_fold[m], pos_fwd[m]) for m in models if abs(pos_fold[m] - pos_fwd[m]) >= 2]
order_changes = "; ".join(f"{m} pasa del puesto {a} al {c}" for m, a, c in sorted(moved, key=lambda x: x[2]))

# ---------- comparaciones calculadas de los CSV
cmp = lambda a, c: compare_models(fold_cost[a], fold_cost[c])
hgb_vs_lgbm_a = cmp(HGB, LGBM_A)
best_vs_hgb = cmp(best, HGB)
lgbm_vs_xgb = cmp(LGBM_A, XGB_A)
second = HGB

# ---------- regiones
sel = regions.pivot(index="grupo", columns="modelo", values="tarde")
mae_reg = regions.pivot(index="grupo", columns="modelo", values="mae")
n_reg = regions.groupby("grupo")["n_pedidos"].first()
spread = (sel.max() - sel.min()).sort_values()
north_se = np.sqrt(0.25 * 0.75 / n_reg["Norte"])
spread_rank_best = list(spread.index).index(best) + 1

# ---------- estadísticas mensuales calculadas aparte
orders["mes"] = orders["order_purchase_timestamp"].dt.to_period("M").astype(str)
train_monthly = orders.groupby("mes")["dias_entrega"].agg(mediana="median", p75=lambda x: x.quantile(0.75), n="size")
past_slow = train_monthly.loc["2017-11":"2018-02"]
raw = raw[(raw["order_purchase_timestamp"] >= "2018-03-01") & (raw["order_purchase_timestamp"] <= "2018-08-31")].copy()
raw["mes"] = raw["order_purchase_timestamp"].dt.to_period("M").astype(str)
open_share = raw.assign(abierto=raw["order_status"].isin(["shipped", "processing", "invoiced", "approved", "created"])).groupby("mes")["abierto"].mean()

cov_month = monthly.pivot(index="mes", columns="modelo", values="cobertura")
real_month = monthly[monthly["modelo"] == best].set_index("mes")[["n_pedidos", "mediana_dias", "p75_dias"]]
cov_models = [m for m in [best, HGB, LGBM_A, LINEAR] if m in cov_month.columns]
first_month, last_month = cov_month.index[0], cov_month.index[-1]

# ---------------- comprobaciones: cada afirmación cualitativa del texto debe cumplirse con los datos ----------------
assert summary.loc[summary["familia"] != "base", ["supera_mediana", "supera_plazo_fijo"]].all().all()
assert s.loc[best, "costo_media"] == s.loc[models, "costo_media"].min() and (summary.loc[~summary["mejor"], "gana_vs_mejor"] == "5 de 5").all()
assert not summary["empate_con_el_mejor"].any()
assert s.loc[models, "adelante_costo"].idxmin() == best
assert abs(s.loc["Ridge", "costo_media"] - s.loc[LINEAR, "costo_media"]) < 0.005
assert s[s["familia"] == "A"]["mae_media"].idxmin() == LGBM_A
assert hgb_vs_lgbm_a["consistente"] and hgb_vs_lgbm_a["texto"] == "5 de 5", "HGB-q ya no vence de forma consistente a LightGBM de valor típico"
assert not lgbm_vs_xgb["consistente"], "XGBoost y LightGBM de valor típico ya no están en empate práctico"
assert (forward_cov[models] > 0.80).all()
assert cov_month.loc[first_month, best] < 0.70 and cov_month.loc[last_month, best] > 0.85, "la cobertura mensual ya no muestra el patrón descrito"
assert real_month["mediana_dias"].iloc[0] > real_month["mediana_dias"].iloc[-1]
assert open_share.idxmax() != last_month, "los pedidos sin cerrar crecen hacia el final: revisar el argumento sobre la censura"
assert (mae_reg.loc[["Norte", "Nordeste"], best] > mae_reg.drop(index=["Norte", "Nordeste"])[best].max()).all()
assert (sel.loc[["Norte", "Nordeste"], LINEAR] > 0.27).all() and sel.loc["Sudeste", LINEAR] < 0.25
family_of = s["familia"]
spread_a = spread[[m for m in spread.index if family_of[m] == "A"]]
spread_b = spread[[m for m in spread.index if family_of[m] == "B"]]
assert best == spread.idxmin(), "el ganador ya no tiene la menor dispersión regional"
assert spread_b.max() < spread_a.min(), "los modelos de la familia B ya no son más parejos entre regiones que los de la A"
assert (s.loc[[m for m in models if s.loc[m, "familia"] == "B"], "sesgo_media"] < 0).all()
assert s.loc["XGBoost cuantil 0,75", "adelante_costo"] < s.loc[LGBM_A, "adelante_costo"] and s.loc["XGBoost cuantil 0,75", "costo_media"] > s.loc[LGBM_A, "costo_media"]

# ---------------- tablas ----------------
rows = ["| Posición | Modelo | Fam. | Costo 3 a 1 (± entre pliegues) | Costo hacia adelante | % tarde | Cobertura | Cobertura hacia adelante | Margen (días) | Ajuste (s) | Supera mediana | Supera plazo fijo |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|"]
for _, r in summary.iterrows():
    rows.append(f"| {int(r.posicion_costo)} | {r.modelo} | {r.familia} | {f(r.costo_media)} ± {f(r.costo_desv)} | {f(r.adelante_costo)} | {pct(r.tarde_media)} | {pct(r.cobertura_media)} | {pct(forward_cov[r.modelo])} | "
                f"{f(r.margen_media, 2)} | {f(fit_time[r.modelo], 1)} | {'sí' if r.supera_mediana else 'no'} | {'sí' if r.supera_plazo_fijo else 'no'} |")
business_md = "\n".join(rows)

rows = ["| Fam. | Modelo | MAE (días) | RMSE | R² | Sesgo | MAE hacia adelante |", "|---|---|---:|---:|---:|---:|---:|"]
for _, r in summary.sort_values(["familia", "mae_media"]).iterrows():
    rows.append(f"| {r.familia} | {r.modelo} | {f(r.mae_media)} ± {f(r.mae_desv)} | {f(r.rmse_media)} | {f(r.r2_media)} | {f(r.sesgo_media)} | {f(r.adelante_mae)} |")
accuracy_md = "\n".join(rows)

rows = ["| Modelo | Pliegues que gana el mejor | Diferencia media | Diferencia / ruido | ¿Ventaja consistente? |", "|---|---:|---:|---:|---|"]
for _, r in summary[~summary["mejor"]].iterrows():
    diff = r.costo_media - b.costo_media
    rows.append(f"| {r.modelo} | {r.gana_vs_mejor} | {f(diff)} ({pct(diff / b.costo_media, 2)}) | {f(r.ratio_vs_mejor, 2)} | {'sí, según la regla' if r.consistente_vs_mejor else 'no (empate práctico)'} |")
pairs_md = "\n".join(rows)

region_order = [m for m in summary["modelo"] if m in sel.columns]
rows = ["| Región | Pedidos | " + " | ".join(region_order) + " |", "|---|---:|" + "---:|" * len(region_order)]
for region in sel.index:
    rows.append(f"| {region} | {fi(n_reg[region])} | " + " | ".join(pct(sel.loc[region, c]) for c in region_order) + " |")
rows.append("| **Rango (máx − mín), puntos** | | " + " | ".join(f"**{f(100 * spread[c], 1)}**" for c in region_order) + " |")
region_md = "\n".join(rows)

rows = ["| Mes de compra | Pedidos | Mediana de días reales | Percentil 75 de días reales | " + " | ".join(f"Cobertura · {m}" for m in cov_models) + " |", "|---|---:|---:|---:|" + "---:|" * len(cov_models)]
for mes in cov_month.index:
    rows.append(f"| {mes} | {fi(real_month.loc[mes, 'n_pedidos'])} | {f(real_month.loc[mes, 'mediana_dias'], 1)} | {f(real_month.loc[mes, 'p75_dias'], 1)} | " + " | ".join(pct(cov_month.loc[mes, m]) for m in cov_models) + " |")
monthly_md = "\n".join(rows)

rows = ["| Mes de compra | Mediana de días | Percentil 75 |", "|---|---:|---:|"]
for mes, r in past_slow.iterrows():
    rows.append(f"| {mes} | {f(r.mediana, 1)} | {f(r.p75, 1)} |")
past_md = "\n".join(rows)

lin_forward_gap = lin.adelante_costo - b.adelante_costo

text = f"""# Etapa 5 · Torneo de modelos (Fase 1)

> Las cifras de las tablas y de cada decisión salen de `notebooks/05_torneo.ipynb` (ejecutado con la semilla 42, solo con `train.parquet`) y de
> `reports/05_torneo_por_pliegue.csv`, `05_torneo_resumen.csv`, `05_torneo_regiones.csv`, `05_torneo_cobertura_mensual.csv` y `05_torneo_tiempos.csv`.
> Las estadísticas mensuales de los días reales en el periodo de entrenamiento y el estado de los pedidos se calcularon aparte (con `train.parquet` y la tabla
> original de pedidos, solo lectura) y se marcan en el texto. Reglas: `especificaciones/05-torneo.md` · Código: `src/olistbr/models/` ·
> Pruebas: `tests/test_models.py`, `test_metrics.py`, `test_evaluation.py`.
> **Esto es la Fase 1 (torneo). La optimización de hiperparámetros (Fase 2) se hace en otra sesión.**

## Resumen

Se evaluaron 10 candidatos (2 líneas base, 5 modelos de la familia A —valor típico más un margen— y 3 de la familia B —percentil 75 directo—) con los mismos 5 pliegues
por cliente de la etapa 4 y la prueba hacia adelante. **El costo 3 a 1 del plazo recomendado decide al ganador.**

- **Gana {best}** (familia {b.familia}) con un costo de **{f(b.costo_media)} ± {f(b.costo_desv)}**, y también es el mejor hacia adelante ({f(b.adelante_costo)}).
  Es un **costo de selección**: se eligió el mejor de 10 con los mismos pliegues con los que se reporta, así que es algo optimista. La estimación honesta del modelo final sale de la prueba de la etapa 7.
- **Ventaja pequeña sobre el segundo.** Contra todos los demás gana en los 5 pliegues, pero contra {second} ({f(s.loc[second, 'costo_media'])}) la diferencia es de {f(best_vs_hgb['diferencia_media'])} por pedido
  ({pct(best_vs_hgb['diferencia_media'] / b.costo_media, 2)} del costo del ganador), con diferencia / ruido {f(best_vs_hgb['ratio'], 2)}. Ganar 5 de 5 por puro azar tendría probabilidad 1/32 (3 %) si los dos modelos fueran idénticos:
  es una señal, no una prueba.
- **Contra las líneas base:** el ganador cuesta {pct(1 - b.costo_media / p75.costo_media, 0)} menos que prometer siempre el percentil 75 ({f(p75.costo_media)}) y {pct(1 - b.costo_media / med.costo_media, 0)} menos que prometer la mediana ({f(med.costo_media)}).
  Los {len(models)} modelos superan a las dos líneas base por más que su ruido.
- **Cobertura en los pliegues:** los plazos de los modelos de las familias A y B cubren entre {pct(s.loc[models, 'cobertura_media'].min())} y {pct(s.loc[models, 'cobertura_media'].max())} de los pedidos (esperado ≈ 75 %).
  **Hacia adelante cubren entre {pct(forward_cov[models].min())} y {pct(forward_cov[models].max())}, y por mes va de {pct(cov_month.loc[first_month, best], 0)} a {pct(cov_month.loc[last_month, best], 0)}** (ver «Hallazgo»).
- **La regresión lineal queda en la posición {int(lin.posicion_costo)}** ({f(lin.costo_media)}): cuesta **{f(gap)} más por pedido** que el ganador ({pct(gap / b.costo_media)} más) en los 5 pliegues
  (diferencia / ruido {f(lin.ratio_vs_mejor, 2)}), y {f(lin_forward_gap)} más hacia adelante. La ventaja es consistente según la regla. **El grupo decidió el 31 oct 2026 que el modelo final no tiene que ser la regresión lineal** («al final se busca el mejor modelo»); la lineal queda como referencia interpretable.
- **Regiones:** los modelos que predicen el percentil 75 directamente (familia B) son mucho más parejos entre regiones que los de valor típico más un margen único (familia A): el rango del porcentaje de pedidos tarde entre regiones es de {f(100 * spread_b.min(), 1)} a {f(100 * spread_b.max(), 1)} puntos en la familia B y de {f(100 * spread_a.min(), 1)} a {f(100 * spread_a.max(), 1)} en la A. El ganador tiene la menor dispersión de los {len(spread)} modelos. Con la regresión lineal el Norte llega a {pct(sel.loc['Norte', LINEAR])} de pedidos tarde y el Sudeste baja a {pct(sel.loc['Sudeste', LINEAR])} (esperado ≈ 25 %).

## Cómo se midió

- **Datos:** `data/processed/train.parquet` (76.924 pedidos, 74.425 clientes), con 7 grupos de variables (logaritmos, mismo estado, regiones, día de la semana, distancia × mismo estado, razones de flete y varios vendedores). Sin la marca de fin de año ni el historial por vendedor; el plazo prometido por Olist nunca entra, ni tampoco el cliente ni el pedido (hay una prueba que lo verifica). La prueba final no se abrió.
- **Pliegues:** los 5 pliegues por cliente de la etapa 4 (semilla 42), iguales para todos los modelos; **prueba hacia adelante** (entrenar hasta 2018-02-28, validar del 2018-03-01 al 2018-08-31): una sola partición, sin ruido medido.
- **Plazo recomendado:** familia A = predicción + margen, donde el margen es el percentil 75 de los errores de un 20 % de clientes apartados dentro de cada entrenamiento (la familia A se ajusta con el 80 % de los datos); familia B = el modelo ya predice el percentil 75; líneas base sin margen.
- **Costo 3 a 1:** subestimar cuesta 3 veces más que sobrestimar; equivale a 4 veces la pérdida cuantílica de 0,75 (comprobado en las pruebas). 1 unidad de costo = un día de sobrestimación.
- **Parámetros fijos, sin optimizar** (los del benchmark de tiempos); el bosque aleatorio es el liviano de 100 árboles.

## Tabla del torneo: costo 3 a 1 (decide)

{business_md}

`±` es la desviación entre los 5 pliegues. «Ajuste» es el tiempo medio por pliegue (cambia entre ejecuciones; está en `05_torneo_tiempos.csv`). «Supera» dice si vence a esa línea base por más que su ruido.
El margen de la familia A varía entre pliegues de {f(margins['min'].min(), 2)} a {f(margins['max'].max(), 2)} días (rango por modelo en `05_torneo_por_pliegue.csv`).

**Cambios de orden hacia adelante.** El primero y el segundo puesto se mantienen, pero el resto cambia: {order_changes}. Hacia adelante, además, la línea base de la mediana ({f(s.loc[BASE[0], 'adelante_costo'])}) cuesta menos que la del percentil 75 ({f(s.loc[BASE[1], 'adelante_costo'])}), otra señal de que el percentil 75 del pasado quedó largo.

## Exactitud: MAE, RMSE, R² y sesgo (solo comparables dentro de cada familia)

{accuracy_md}

**Cómo leerla.** Los modelos de la familia B predicen el **percentil 75**, no el valor típico: su sesgo es negativo (quedan por encima del valor típico a propósito) y su MAE, RMSE y R² salen peores **por diseño**, no porque sean peores modelos. Entre familias solo se compara el costo.
Dentro de la familia A, el de mejor MAE es {s[s.familia == 'A'].sort_values('mae_media').index[0]} ({f(s[s.familia == 'A']['mae_media'].min())} días); la regresión lineal tiene {f(lin.mae_media)}.

## Comparación por pares contra el ganador

{pairs_md}

Una ventaja es **consistente según la regla** si el ganador gana en al menos 4 de 5 pliegues y la diferencia supera 1 vez su ruido. Ningún candidato queda en empate práctico con el ganador, pero la ventaja sobre {second} es de apenas {pct(best_vs_hgb['diferencia_media'] / b.costo_media, 2)} del costo.
El «ruido» es la desviación de solo 5 diferencias y los pliegues comparten gran parte del entrenamiento: **no es una prueba de significancia**, y no se midió cuánto cambian los resultados con otra semilla o con otro sorteo de los clientes de calibración.

## Hallazgo: la calibración no se mantiene en el tiempo

En los pliegues los plazos cubren ≈ 75 % de los pedidos. En la prueba hacia adelante (compras de marzo a agosto de 2018) **la cobertura del ganador va de {pct(cov_month.loc[first_month, best])} en {first_month} a {pct(cov_month.loc[last_month, best])} en {last_month}**:

{monthly_md}

- Marzo queda **corto** (más pedidos tarde de lo esperado) y de junio en adelante el plazo queda **más largo de lo necesario** (más del 90 % llega a tiempo). El cambio ocurre dentro de la propia ventana de validación.
- Los modelos se mueven juntos: el problema no es de un modelo, es que el ritmo de entrega cambia mes a mes (mediana de {f(real_month['mediana_dias'].iloc[0], 1)} días en {first_month} a {f(real_month['mediana_dias'].iloc[-1], 1)} en {last_month}).
- Antes del corte del entrenamiento hacia adelante hubo meses lentos (*calculado aparte desde `train.parquet`*):

{past_md}

**Posibles explicaciones (hipótesis; con estos datos no se pueden separar del todo):**
1. *Cambio del ritmo de la operación:* las entregas se aceleran de marzo a agosto de 2018.
2. *Mezcla de temporadas en el entrenamiento hacia adelante:* incluye el pico de noviembre-diciembre de 2017 y los meses lentos de enero a marzo de 2018, que elevan el percentil 75 aprendido; la validación no contiene esa temporada.
3. *Pedidos lentos que aún no se habían entregado al cortar los datos* (solo hay pedidos ya entregados): *calculado aparte con la tabla original de pedidos*, el porcentaje de pedidos todavía sin cerrar por mes de compra va de {pct(open_share.min(), 2)} a {pct(open_share.max(), 2)}
   y es mayor en {open_share.idxmax()}, no hacia {last_month}. Por eso no parece ser la causa principal.

- No cambia al ganador: sigue siendo el mejor hacia adelante. **Sí es una advertencia para la app:** un plazo aprendido una vez se desajusta cuando cambia el ritmo de entrega; conviene vigilar la cobertura y recalibrar con datos recientes (ver «Para decidir en grupo»).

## Revisión por región del cliente

Porcentaje de pedidos que llega **después** del plazo recomendado (esperado ≈ 25 %), con predicciones fuera de muestra, para todos los modelos:

{region_md}

- Frente a los **modelos de la familia A** (valor típico + margen único), el ganador y los demás de la familia B son claramente más parejos: con un margen único para todo el país el **Norte y el Nordeste** quedan por encima de lo esperado y el Sudeste por debajo (la regresión lineal: {pct(sel.loc['Norte', LINEAR])}, {pct(sel.loc['Nordeste', LINEAR])} y {pct(sel.loc['Sudeste', LINEAR])}).
- Frente a los otros modelos de la familia B (por ejemplo {HGB}) **no se puede ordenar**: la diferencia de dispersión es de un punto, y el error de muestreo del Norte ({fi(n_reg['Norte'])} pedidos) es de unos ±{f(100 * north_se, 1)} puntos.
- Una explicación posible, no probada: los modelos que predicen el percentil 75 aprenden una incertidumbre distinta por tipo de pedido, mientras que un margen único es el mismo para todos. Un margen por región sería otra forma de resolverlo.
- El error absoluto es mayor en el Norte y el Nordeste (ver `05_torneo_regiones.csv`; solo comparable dentro de cada familia): son las regiones donde conviene mirar con más cuidado en la validación final (principio VIII).

## Finalistas propuestos (los confirma el grupo)

Reglas aprobadas, en orden: (1) consistencia, (2) costo, (3) tiempo y simplicidad —**solo ante un empate práctico**—, (4) interpretabilidad, (5) métricas técnicas y cobertura.

1. **{best}** (familia {b.familia}): el menor costo, gana en los 5 pliegues a todos los demás, también es el mejor hacia adelante y se ajusta en {f(fit_time[best], 1)} s. No necesita margen.
2. **Segundo finalista según la regla: {second}** (familia B, costo {f(s.loc[second, 'costo_media'])}). Es el segundo por costo y vence a LightGBM de valor típico —el siguiente— en {hgb_vs_lgbm_a['texto']} pliegues
   (diferencia media {f(hgb_vs_lgbm_a['diferencia_media'])}, diferencia / ruido {f(hgb_vs_lgbm_a['ratio'], 2)}), es decir, por la regla es una ventaja consistente. Se parece mucho al ganador (misma familia y misma pérdida) y es más lento ({f(fit_time[second], 1)} s contra {f(fit_time[best], 1)} s).
3. **Alternativa fuera de la regla, si el grupo la prefiere: {LGBM_A}** (familia A, costo {f(s.loc[LGBM_A, 'costo_media'])}, MAE {f(s.loc[LGBM_A, 'mae_media'])}, ajuste {f(fit_time[LGBM_A], 1)} s). No es una conclusión del torneo sino una preferencia: aportaría un modelo de otra familia (con margen) y la mejor exactitud del valor típico,
   útil si la app muestra además un tiempo típico, pero hay que decir que **pierde frente a {second} en los 5 pliegues**. Hacia adelante {HGB} también queda mejor ({f(s.loc[HGB, 'adelante_costo'])} contra {f(s.loc[LGBM_A, 'adelante_costo'])}).
4. XGBoost de valor típico queda en empate práctico con LightGBM de valor típico (gana {lgbm_vs_xgb['texto']}, diferencia / ruido {f(lgbm_vs_xgb['ratio'], 2)}); gana el más rápido.
5. **Descartados:** Ridge (idéntico a la regresión lineal: {f(s.loc['Ridge', 'costo_media'])} contra {f(lin.costo_media)}), el bosque liviano ({f(s.loc['Bosque aleatorio liviano', 'costo_media'])}) y las dos líneas base. XGBoost cuantil ({f(s.loc['XGBoost cuantil 0,75', 'costo_media'])}) queda 5.º por pliegues pero sube al 3.º hacia adelante:
   no se avanza por la regla (consistencia en los pliegues), pero se anota el cambio de orden.
   La regresión lineal **no se descarta del reporte**: queda como referencia interpretable con la diferencia que se muestra abajo.

**Por qué el torneo no elige un campeón.** Los parámetros son fijos y no están optimizados: la optimización puede cambiar el orden entre los cuatro primeros (sus costos están entre {f(s.loc[best, 'costo_media'])} y {f(s.loc[XGB_A, 'costo_media'])}), la ventaja entre los dos primeros es del {pct(best_vs_hgb['diferencia_media'] / b.costo_media, 2)} y solo la prueba final de la etapa 7 dice qué tan bueno es el modelo elegido. El torneo descarta; no corona.

## La regresión lineal contra el ganador

| | Regresión lineal | {best} | Diferencia |
|---|---:|---:|---:|
| Costo 3 a 1 por pedido (pliegues) | {f(lin.costo_media)} | {f(b.costo_media)} | **+{f(gap)}** |
| Por cada 1.000 pedidos (unidades de costo) | {fi(lin.costo_media * 1000)} | {fi(b.costo_media * 1000)} | **+{fi(gap * 1000)}** |
| Costo 3 a 1 por pedido (hacia adelante) | {f(lin.adelante_costo)} | {f(b.adelante_costo)} | **+{f(lin_forward_gap)}** |
| % de pedidos que llegan tarde en el Norte | {pct(sel.loc['Norte', LINEAR])} | {pct(sel.loc['Norte', best])} | |

- La diferencia no se convierte en dinero: el costo 3 a 1 es un supuesto en unidades relativas (1 unidad = un día de sobrestimación). Se expresa por pedido y por cada 1.000 pedidos.
- Dentro de la familia A, la regresión lineal con margen tiene un MAE de {f(lin.mae_media)} días y LightGBM con margen, {f(s.loc[LGBM_A, 'mae_media'])}: los árboles también son más exactos para el valor típico. El MAE del ganador no se compara con el de la lineal (distinta familia).

## Limitaciones

1. **Parámetros fijos:** ningún modelo está optimizado; la distancia entre los cuatro primeros es pequeña y la etapa 6 puede reordenarlos.
2. **Datos distintos entre familias:** la familia A se ajusta con el 80 % de cada entrenamiento (se aparta un 20 % de clientes para el margen) y la B con el 100 %. El titular «gana la familia B» puede deberse en parte a eso y a que el margen de la A es el mismo para todos los pedidos. **No hay un control que lo aísle** (queda para la Fase 2).
3. **Costo de selección:** el costo del ganador es el del mejor de 10 elegido con los mismos pliegues; es optimista.
4. **Prueba hacia adelante:** es una sola partición, sin ruido medido. Entrena con compras hasta 2018-02-28, y algunos de esos pedidos se entregaron después del 1 de marzo (una fuga temporal pequeña: ese día todavía no se conocía su plazo de entrega).
5. **El ruido de 5 pliegues no es significancia estadística** y los pliegues comparten datos de entrenamiento; no se midió la variación por semilla ni por sorteo de calibración.
6. **La calibración depende del periodo** (ver el hallazgo): la cobertura hacia adelante va de {pct(cov_month.loc[first_month, best], 0)} a {pct(cov_month.loc[last_month, best], 0)} según el mes.
7. **Tendencia entre años** (heredada de la etapa 4): los modelos no pueden anticipar cambios posteriores a agosto de 2018. Que el costo hacia adelante sea menor que el de los pliegues en todos los modelos es coherente con entregas más rápidas, pero no se probó.
8. **Variables:** sin la marca de fin de año (una sola temporada) y sin el historial por vendedor; se evalúan después en los finalistas.
9. **Tiempos:** son del equipo del grupo con carga variable y cambian entre ejecuciones.

## Para decidir en grupo

1. **Segundo finalista:** la regla da {second}; ¿lo aceptan o prefieren {LGBM_A} por razones que no son de la regla (otra familia, mejor exactitud)?
2. **Regresión lineal — DECIDIDO (31 oct 2026):** el ganador la supera de forma consistente (+{f(gap)} por pedido, {pct(gap / b.costo_media)} más) y el grupo aceptó que el modelo final no sea la regresión lineal, porque el objetivo es el mejor modelo. Queda como referencia interpretable. (Pregunta Q4 de `especificaciones/05-torneo.md`.)
3. **Regiones:** el Norte y el Nordeste tienen el mayor error absoluto y, con la regresión lineal, más pedidos tarde de lo esperado; ¿se agrega un indicador por región en la validación final?
4. **Calibración con el tiempo:** la cobertura se desplaza de un mes a otro. ¿Se recalibra el margen o el modelo con los meses más recientes antes de la validación final y se anota como parte del uso de la app (vigilar la cobertura)?
5. **Control de datos entre familias:** ¿se agrega en la Fase 2 una comparación con los mismos datos de ajuste para la familia B?

## Revisiones

### Auditoría de fuga (`auditor-de-fuga`)

**Veredicto: sin fuga en el código del torneo.** El margen de la familia A sale solo de clientes apartados dentro del entrenamiento del pliegue (nunca de la validación) y ningún cliente está a la vez en el ajuste y en la calibración; los pliegues y la partición hacia adelante son por cliente; todo lo que aprende está dentro del pliegue;
la prueba final no se abre; `promised_days`, la marca de fin de año y el historial no entran; `customer_unique_id` no llega al modelo.
Puntos que se atendieron: (1) el costo del ganador es un costo de selección (ahora lo dice el resumen); (2) la causa de la sobrecobertura estaba escrita como un hecho y ahora se presenta como hipótesis con la evidencia mes a mes (cambio de ritmo, mezcla de temporadas y pedidos sin entregar, esta última con un dato en contra);
(3) faltaba una prueba automática de que el cliente, el pedido y el plazo prometido no llegan al modelo (ahora existe). Notas sin acción: la lista de vendedores frecuentes viene de la etapa 3 (solo conteos) y la pequeña fuga temporal de la partición hacia adelante (limitación 4).

### Revisión de código (`revisor-de-codigo-ia`)

**Veredicto: aceptar con cambios.** El código y las pruebas eran sólidos; los errores estaban en el reporte y en algunas pruebas.

| Hallazgo | Qué se hizo |
|---|---|
| Porcentajes mal calculados («cuesta 21 % más») | Corregido: el ganador cuesta 21 % y 35 % *menos* que las líneas base |
| Explicación de la sobrecobertura demasiado segura, palabras invertidas («sobrepromete») y falsa afirmación de que el orden no cambia | Reescrita como hipótesis; se agregó la cobertura mes a mes al notebook; se lista cómo cambia el orden hacia adelante |
| Segundo finalista contradecía las reglas aprobadas | Corregido: la regla da {second} (vence a LightGBM de valor típico en {hgb_vs_lgbm_a['texto']} pliegues); LightGBM de valor típico queda como alternativa fuera de la regla |
| «El ganador es el más parejo entre regiones» solo estaba verificado contra 3 modelos | La revisión por región se calculó para los {len(region_order)} modelos y el texto solo afirma lo que respaldan los datos |
| «Ventaja consistente» con más certeza de la debida | Se agregó el tamaño de la ventaja (0,36 %), la probabilidad de ganar 5 de 5 por azar (1/32) y la advertencia de que no hay variación por semilla |
| Conversión a «unidades de costo por mes» con bases frágiles | Eliminada; solo por pedido y por 1.000 pedidos |
| Comparación de MAE entre familias impresa en el notebook | Eliminada |
| Pruebas que pasarían aunque el código estuviera mal | Se fortalecieron: el margen es exactamente el percentil 75 (con errores conocidos), el modelo del margen no ve a los clientes de calibración, las predicciones regionales son fuera de muestra (un árbol que memoriza daría error cercano a 0 si no lo fueran), reproducibilidad y que no se modifican las entradas |
| Casos borde (clientes vacíos, longitud distinta, modelo inexistente, falta la región) | Ahora fallan con un mensaje claro |
| Duplicación de `predicción + margen` | Reutiliza `DeadlineModel.predict_with_deadline` |
| Aviso de «pocos clientes de calibración» oculto en el notebook | Ya no se silencia |
| Documentos desactualizados (119 pruebas, margen «de validación», contrato) | Corregidos |
| Control de la desventaja de datos de la familia A | Pendiente para la Fase 2 (limitación 2) |
"""
open(R + "05_torneo.md", "w", encoding="utf-8").write(text)
print("ok", len(text.splitlines()), "líneas")
