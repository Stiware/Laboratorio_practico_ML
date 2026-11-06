# Data Model: Optimización de hiperparámetros de OnTrack (etapa 6, Fase 2)

No hay base de datos. Las «entidades» son tablas en memoria y archivos de salida.

## Datos de búsqueda y reserva

| Entidad | Origen | Filas aprox. | Reglas |
|---|---|---:|---|
| `busqueda` | `split_by_group(train, test_size=0.25, seed=7)` → primer elemento | ≈ 57.700 pedidos (75 % de los clientes) | Solo con ella se crean pliegues y se ajustan los modelos que se comparan |
| `reserva` | mismo llamado → segundo elemento | ≈ 19.200 pedidos (25 % de los clientes) | La búsqueda nunca la ve; solo comprobación honesta, elección entre finalistas, hacia adelante y regiones |
| `folds` | `make_group_folds(busqueda, seed=42)` | 5 pares de índices | Creados una vez; iguales para todos los ensayos y los dos finalistas |
| `forward_train`, `forward_valid` | `reserved_forward_split(busqueda, reserva)` | depende de fechas | `forward_train` ⊂ `busqueda` (compras ≤ 2018-02-28); `forward_valid` ⊂ `reserva` (compras 2018-03-01 a 2018-08-31) |

Invariantes: `busqueda` ∩ `reserva` = ∅ en `customer_unique_id`; en cada pliegue, entrenamiento ∩ validación = ∅; `test.parquet` no se lee.
Las filas tienen las columnas de `add_features` más `customer_unique_id` (solo agrupa; no entra al modelo) y `dias_entrega` (objetivo).

## Finalista

| Campo | Tipo | Descripción |
|---|---|---|
| `name` | str | `"LightGBM cuantil 0,75"` o `"HistGradientBoosting cuantil 0,75"` (mismos nombres de `candidates()`) |
| `space` | función `(trial) → dict` | `lgbm_quantile_space` o `hgb_quantile_space` |
| `make` | función `(params) → modelo` | `make_lgbm_quantile` o `make_hgb_quantile`; percentil 75 fijo |
| partida | `Candidate` | `candidates()[name]`, sin cambios |

## Ensayo (una fila de `reports/06_ensayos.csv`)

| Columna | Descripción |
|---|---|
| `modelo` | nombre del finalista |
| `ensayo` | número del ensayo en el estudio (0 a 19) |
| `estado` | `COMPLETE` o `FAIL` |
| `param_<nombre>` | una columna por hiperparámetro del ensayo |
| `costo_media`, `costo_desv` | costo 3 a 1 medio y desviación (`ddof=1`) de los 5 pliegues |
| `mae_media`, `rmse_media`, `r2_media`, `sesgo_media`, `tarde_media`, `cobertura_media` | medias entre pliegues |
| `fit_seconds_media` | tiempo medio de ajuste (cambia entre ejecuciones) |
| `mlflow_run_id` | identificador de la ejecución hija (vacío si `log=False`) |

## Comprobación (por finalista)

| Campo | Descripción |
|---|---|
| `costo_partida`, `costo_optimizado` | costo 3 a 1 medio en la reserva, ajustando con toda la `busqueda` |
| `diferencia` | `costo_optimizado − costo_partida` (negativo = el optimizado cuesta menos) |
| `ic_inferior`, `ic_superior` | percentiles 2,5 y 97,5 de 1.000 sorteos de clientes (semilla 42) |
| `decision` | `"optimizados"` solo si `ic_superior < 0`; si no, `"partida"` |
| `motivo` | texto de `decide_params` |
| `arboles_partida` | árboles que usó realmente el modelo de partida (relevante en HistGradientBoosting) |

## Decisión final (`reports/06_best_params.json`)

```json
{
  "fecha": "AAAA-MM-DD",
  "semillas": {"reserva": 7, "pliegues": 42, "muestreador": 42, "bootstrap": 42},
  "presupuesto_por_finalista": 20,
  "finalistas": {
    "<nombre>": {
      "parametros_partida": {},
      "parametros_optimizados": {},
      "mejor_costo_busqueda_optimista": 0.0,
      "comprobacion": {"diferencia": 0.0, "ic": [0.0, 0.0], "decision": "partida", "motivo": "..."},
      "parametros_elegidos": {}
    }
  },
  "eleccion": {"modelo": "<nombre>", "diferencia_a_menos_b": 0.0, "ic": [0.0, 0.0], "motivo": "..."}
}
```

Los valores `0.0` y `"..."` son solo la forma: **todas las cifras salen del notebook ejecutado** (principio VII).

## Transiciones

`train.parquet` → (`split_by_group`) → `busqueda` / `reserva` → (`tune` × 2 finalistas, 20 ensayos) → parámetros optimizados →
(ajuste con `busqueda`, evaluación en `reserva`, `bootstrap_cost_difference`, `decide_params`) → parámetros elegidos por finalista →
(`bootstrap_cost_difference` A − B, `choose_finalist`) → modelo elegido → (informativo) hacia adelante y regiones → etapa 7.
