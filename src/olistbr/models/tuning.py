"""Optimización de hiperparámetros de los dos finalistas del torneo (etapa 6, `especificaciones/06-optimizacion.md`).

Sigue la forma de `tuning.py` del ejemplo CreditSight, adaptada de clasificación a **regresión**:

- Los finalistas (LightGBM cuantil 0,75 e HistGradientBoosting cuantil 0,75) predicen directamente el **percentil 75**, que es el plazo
  que minimiza el costo 3 a 1. Por eso el percentil queda **fijo**: si un ensayo lo cambiara, el costo dejaría de medir el plazo correcto.
- La búsqueda no ve a un 25 % de los clientes (la **reserva**). Con ellos se comprueba si los parámetros optimizados son mejores que los
  de partida (los del torneo), porque la mejor puntuación de una búsqueda es optimista.
- El intervalo del 95 % de esa comprobación **sortea clientes completos**, no pedidos: los pedidos de un mismo cliente no son independientes.
- Las reglas de decisión aprobadas por el grupo están escritas aquí como funciones con pruebas, para no aplicarlas «a ojo».
"""
from pathlib import Path
from typing import Callable, NamedTuple

import mlflow
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor

from olistbr.config import (
    BOOTSTRAP_ROUNDS, GROUP_COL, RANDOM_STATE, TARGET, TARGET_QUANTILE, TIE_WINNER, TOURNAMENT_GROUPS, TUNING_TRIALS,
)
from olistbr.models.evaluation import evaluate_candidate_folds
from olistbr.models.metrics import delivery_cost
from olistbr.models.train import Candidate, DeadlineModel, build_pipeline, candidates, fit_deadline_model
from olistbr.tracking.mlflow_utils import log_run

# Los dos finalistas que dio la regla del torneo (reports/05_torneo.md), con los mismos nombres de `candidates()`.
FINALIST_NAMES = ("LightGBM cuantil 0,75", "HistGradientBoosting cuantil 0,75")


def _check_finalist(name: str) -> None:
    """Falla con un mensaje claro si `name` no es uno de los dos finalistas."""
    if name not in FINALIST_NAMES:
        raise KeyError(f"{name!r} no es finalista; los finalistas son: {', '.join(FINALIST_NAMES)}")


def starting_candidate(name: str) -> Candidate:
    """Parámetros de partida de un finalista: exactamente el candidato del torneo, sin ningún cambio.

    Se toma de `candidates()` en vez de copiar los números, para que «partida contra optimizado» compare contra lo que de verdad
    se midió en el torneo (incluida la parada temprana por defecto de HistGradientBoosting, ver `research.md` D3).
    """
    _check_finalist(name)
    return candidates()[name]


# --- Comprobación honesta (US1) ----------------------------------------------------------------------

_BOOTSTRAP_CHUNK = 100  # sorteos por bloque: con ≈ 19.000 clientes, 1.000 sorteos de una vez ocuparían ≈ 150 MB


def bootstrap_cost_difference(real, plan_a, plan_b, groups, n_boot: int = BOOTSTRAP_ROUNDS, seed: int = RANDOM_STATE):
    """Diferencia de costo 3 a 1 por pedido (A menos B) con su intervalo del 95 %, **sorteando clientes completos**.

    1. Costo de cada pedido con cada plazo (`delivery_cost`, subestimar cuesta 3 veces más) y su diferencia A − B.
    2. Suma de diferencias y número de pedidos **por cliente**.
    3. `n_boot` sorteos con reemplazo de clientes; en cada uno, la diferencia media por pedido = suma de sumas / suma de pedidos.
    Devuelve `(diferencia total, percentil 2,5, percentil 97,5)`. Negativo = A cuesta menos que B.

    Sortear pedidos sueltos sería otra fuga: los pedidos de un mismo cliente no son independientes y el intervalo saldría
    más angosto de lo que es (guía §3.6).
    """
    real = np.asarray(real, dtype=float)
    plan_a = np.asarray(plan_a, dtype=float)
    plan_b = np.asarray(plan_b, dtype=float)
    groups = np.asarray(groups, dtype=object)
    if not (len(real) == len(plan_a) == len(plan_b) == len(groups)):
        raise ValueError(
            f"los datos tienen distinto largo: real={len(real)}, plan_a={len(plan_a)}, plan_b={len(plan_b)}, clientes={len(groups)}"
        )
    if len(real) == 0:
        raise ValueError("no hay pedidos: las entradas están vacías")
    if not (np.isfinite(real).all() and np.isfinite(plan_a).all() and np.isfinite(plan_b).all()):
        raise ValueError("hay días reales o plazos vacíos o infinitos (valores no finitos): no se puede calcular el costo")
    if pd.isna(groups).any():
        raise ValueError("hay pedidos sin cliente: no se pueden sortear clientes completos")
    if n_boot < 1:
        raise ValueError(f"n_boot debe ser al menos 1 (se recibió {n_boot})")

    difference = delivery_cost(real, plan_a) - delivery_cost(real, plan_b)
    per_customer = pd.DataFrame({"g": groups, "d": difference}).groupby("g", sort=True)["d"].agg(["sum", "count"])
    sums = per_customer["sum"].to_numpy()
    counts = per_customer["count"].to_numpy()
    rng = np.random.default_rng(seed)
    draws = []
    for start in range(0, n_boot, _BOOTSTRAP_CHUNK):
        size = min(_BOOTSTRAP_CHUNK, n_boot - start)
        idx = rng.integers(0, len(sums), size=(size, len(sums)))
        draws.append(sums[idx].sum(axis=1) / counts[idx].sum(axis=1))
    draws = np.concatenate(draws)
    total = float(sums.sum() / counts.sum())
    return total, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def _check_interval(low: float, high: float) -> None:
    """Falla si el intervalo no tiene sentido: límites vacíos o infinitos (no finitos), o al revés."""
    if not (np.isfinite(low) and np.isfinite(high)):
        raise ValueError(f"intervalo con límites no finitos ({low}, {high}): no se puede aplicar la regla")
    if low > high:
        raise ValueError(f"intervalo al revés: límite inferior {low} mayor que el superior {high}")


def decide_params(low: float, high: float) -> tuple:
    """Regla escrita antes de mirar (regla 6): parámetros optimizados **solo si todo el intervalo está por debajo de cero**.

    `low` y `high` son los límites del intervalo de `costo(optimizado) − costo(partida)`. Devuelve `(decisión, motivo)`, donde la
    decisión es `"optimizados"` o `"partida"`. Si el límite superior toca el cero, el intervalo no está por debajo de cero.
    """
    _check_interval(low, high)
    if high < 0:
        return "optimizados", "todo el intervalo está por debajo de cero: los parámetros optimizados cuestan menos que los de partida"
    if low > 0:
        return "partida", "todo el intervalo está por encima de cero: el optimizado es peor que la partida"
    return "partida", "el intervalo cruza o toca el cero: la mejora no se distingue del ruido y se usan los parámetros de partida"


def choose_finalist(low: float, high: float, name_a: str, name_b: str, tie_winner: str = TIE_WINNER) -> tuple:
    """Elige entre los dos finalistas con el intervalo de `costo(A) − costo(B)` en la reserva (Q7).

    Todo el intervalo bajo cero → gana A; todo sobre cero → gana B; si cruza o toca el cero es un **empate práctico** y gana
    `tie_winner` (LightGBM cuantil 0,75: el más rápido y el ganador del torneo). Devuelve `(modelo elegido, motivo)`.
    """
    _check_interval(low, high)
    if tie_winner not in (name_a, name_b):
        raise ValueError(f"el ganador en caso de empate ({tie_winner!r}) debe ser uno de los dos finalistas")
    if high < 0:
        return name_a, f"todo el intervalo está por debajo de cero: {name_a} cuesta menos que {name_b}"
    if low > 0:
        return name_b, f"todo el intervalo está por encima de cero: {name_b} cuesta menos que {name_a}"
    return tie_winner, f"el intervalo cruza el cero: empate práctico; gana {tie_winner} por la regla de desempate (Q7)"


def fit_and_recommend(candidate: Candidate, train: pd.DataFrame, valid: pd.DataFrame, groups_col: str = GROUP_COL, target: str = TARGET):
    """Ajusta un candidato **solo con `train`** y devuelve `(días reales, plazo recomendado, modelo)` en `valid`.

    Usa el mismo pipeline y el mismo modelo de plazo del torneo (`build_pipeline` con `TOURNAMENT_GROUPS` y `fit_deadline_model`),
    para que la comprobación mida exactamente igual que la etapa 5. El candidato que se pasa no se modifica.
    """
    pipeline = build_pipeline(TOURNAMENT_GROUPS, model=clone(candidate.model), scale=candidate.scale)
    X_train = train.drop(columns=[target])
    model = fit_deadline_model(pipeline, candidate.family, X_train, train[target], X_train[groups_col])
    _, recommended = model.predict_with_deadline(valid.drop(columns=[target]))
    return valid[target].to_numpy(dtype=float), np.asarray(recommended, dtype=float), model


def n_trees_used(model: DeadlineModel) -> int:
    """Árboles que usó realmente un finalista ya ajustado.

    En HistGradientBoosting puede ser menos que `max_iter` si la parada temprana estaba activa (la partida del torneo, Q5).
    """
    estimator = model.pipeline.named_steps["model"]
    if isinstance(estimator, HistGradientBoostingRegressor):
        return int(estimator.n_iter_)
    if isinstance(estimator, LGBMRegressor):
        return int(estimator.booster_.current_iteration())
    raise TypeError(f"n_trees_used solo sirve para los finalistas; se recibió {type(estimator).__name__}")


# --- Búsqueda (US2) ------------------------------------------------------------------------------------
# Los rangos salen del conocimiento del algoritmo (guía §3.2), no del azar; el motivo de cada uno está en
# especificaciones/06-optimizacion.md §4.


def lgbm_quantile_space(trial) -> dict:
    """Parámetros de un ensayo de LightGBM cuantil 0,75 (la partida del torneo: 300 árboles, tasa 0,05, 31 hojas)."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 150, 800),  # con tasas bajas hacen falta más árboles
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),  # se compensa con el número de árboles
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),  # complejidad de cada árbol; 31 queda en el medio
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 300),  # un percentil necesita hojas con más pedidos
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),  # muestrear filas reduce el sobreajuste
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),  # ≈ 100 columnas, muchas casi constantes
        "reg_lambda": trial.suggest_float("reg_lambda", 0.001, 10.0, log=True),  # regularización L2 de las hojas
    }


def hgb_quantile_space(trial) -> dict:
    """Parámetros de un ensayo de HistGradientBoosting cuantil 0,75 (mismos conceptos que LightGBM, con sus nombres)."""
    return {
        "max_iter": trial.suggest_int("max_iter", 150, 800),  # número de árboles, explícito: la parada temprana se apaga (Q5)
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 127),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 20, 300),
        "l2_regularization": trial.suggest_float("l2_regularization", 0.001, 10.0, log=True),
        "max_features": trial.suggest_float("max_features", 0.5, 1.0),  # proporción de columnas por división
    }


def _reject_fixed(params: dict, fixed: tuple, model: str) -> None:
    """Falla si un ensayo intenta cambiar algo que debe quedar fijo (el percentil, la pérdida o la parada temprana)."""
    clash = sorted(set(params) & set(fixed))
    if clash:
        raise ValueError(f"{model}: {', '.join(clash)} queda fijo en la optimización y no se puede cambiar desde un ensayo")


def make_lgbm_quantile(params: dict, random_state: int = RANDOM_STATE) -> LGBMRegressor:
    """LightGBM que predice el percentil 75, con los parámetros del ensayo. Lo fijo es igual al torneo (más `subsample_freq=1`,
    necesario para que `subsample` tenga efecto)."""
    _reject_fixed(params, ("objective", "alpha", "subsample_freq", "random_state", "n_jobs", "deterministic", "force_row_wise", "verbose"),
                  "LightGBM")
    return LGBMRegressor(
        objective="quantile", alpha=TARGET_QUANTILE, subsample_freq=1, random_state=random_state, n_jobs=2,
        deterministic=True, force_row_wise=True, verbose=-1, **params,
    )


def make_hgb_quantile(params: dict, random_state: int = RANDOM_STATE) -> HistGradientBoostingRegressor:
    """HistGradientBoosting que predice el percentil 75, **sin parada temprana** (Q5): el número de árboles es `max_iter`."""
    _reject_fixed(params, ("loss", "quantile", "early_stopping", "random_state"), "HistGradientBoosting")
    return HistGradientBoostingRegressor(
        loss="quantile", quantile=TARGET_QUANTILE, early_stopping=False, random_state=random_state, **params,
    )


class Finalist(NamedTuple):
    """Un finalista: cómo se proponen sus parámetros y cómo se construye su modelo."""

    space: Callable
    make: Callable


FINALISTS = {
    "LightGBM cuantil 0,75": Finalist(lgbm_quantile_space, make_lgbm_quantile),
    "HistGradientBoosting cuantil 0,75": Finalist(hgb_quantile_space, make_hgb_quantile),
}

class NonFiniteCost(ValueError):
    """El costo de un ensayo salió vacío o infinito: el ensayo se marca como fallido y la búsqueda sigue.

    Es la única excepción que la búsqueda deja pasar; cualquier otro error (por ejemplo, una columna que falta) la detiene.
    """


# Métricas que se guardan de cada ensayo (media y desviación entre pliegues); solo el costo se optimiza (Q2).
TRIAL_METRICS = ("costo", "mae", "rmse", "r2", "sesgo", "tarde", "cobertura")


def tuned_candidate(name: str, params: dict) -> Candidate:
    """El finalista `name` con los parámetros de un ensayo, como candidato de la familia B (sin escalar, sin margen)."""
    _check_finalist(name)
    return Candidate(FINALISTS[name].make(params), "B", False)


def _slug(name: str) -> str:
    """Nombre corto para MLflow y Optuna: «LightGBM cuantil 0,75» -> «lightgbm-cuantil»."""
    return name.split(" 0,")[0].lower().replace(" ", "-")


def _trial_board(by_fold: pd.DataFrame) -> dict:
    """Medias y desviaciones (ddof=1) entre pliegues de las métricas del ensayo, más el tiempo medio de ajuste."""
    board = {}
    for metric in TRIAL_METRICS:
        board[f"{metric}_media"] = float(by_fold[metric].mean())
        board[f"{metric}_desv"] = float(by_fold[metric].std(ddof=1))
    board["fit_seconds_media"] = float(by_fold["fit_seconds"].mean())
    return board


def tune(name: str, X: pd.DataFrame, y: pd.Series, folds: list, n_trials: int = TUNING_TRIALS, seed: int = RANDOM_STATE,
         log: bool = True, storage=None) -> optuna.Study:
    """Estudio de Optuna que **minimiza el costo 3 a 1** medio de los pliegues para un finalista (`research.md` D6).

    - Cada ensayo se mide con `evaluate_candidate_folds`, la misma función del torneo (mismas variables, mismo plazo recomendado).
    - En cada ensayo se guardan las medias y desviaciones de todas las métricas (`user_attrs`) y, con `log=True`, una ejecución hija
      en MLflow dentro de una ejecución principal por finalista.
    - Un ensayo con costo no finito queda **fallido** (también en MLflow), cuenta dentro del presupuesto y no puede ser el mejor.
      Cualquier otro error detiene la búsqueda en vez de esconderse como ensayo fallido.
    - Con `storage` (ruta de un archivo .db) el estudio se guarda y **se reanuda**: solo se hacen los ensayos que faltan hasta `n_trials`.
    `X` debe traer `customer_unique_id`: agrupa la evaluación y nunca entra al modelo.
    """
    _check_finalist(name)
    if n_trials < 1:
        raise ValueError(f"n_trials debe ser al menos 1 (se recibió {n_trials})")
    if GROUP_COL not in X.columns:
        raise ValueError(f"`X` debe traer la columna {GROUP_COL!r} para agrupar por cliente")
    slug = _slug(name)
    tags = {"etapa": "06_optimizacion", "modelo": name, "metrica_optimizada": "costo"}

    def objective(trial):
        params = FINALISTS[name].space(trial)
        board = _trial_board(evaluate_candidate_folds(name, tuned_candidate(name, params), X, y, folds))
        for key, value in board.items():
            trial.set_user_attr(key, value)
        cost = board["costo_media"]
        finite = bool(np.isfinite(cost))
        if log:  # también los fallidos quedan en MLflow, marcados con la etiqueta `estado`
            metrics = {k: v for k, v in board.items() if np.isfinite(v)}
            run_id = log_run(f"{slug}-ensayo-{trial.number:02d}", params, metrics,
                             tags={**tags, "estado": "COMPLETE" if finite else "FAIL"}, nested=True)
            trial.set_user_attr("mlflow_run_id", run_id)
        if not finite:
            raise NonFiniteCost(f"ensayo {trial.number}: el costo no es finito ({cost})")
        return cost

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed), study_name=f"ontrack-{slug}-costo",
        storage=(f"sqlite:///{Path(storage).as_posix()}" if storage else None), load_if_exists=True,
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    done = sum(t.state.is_finished() for t in study.trials)
    remaining = max(n_trials - done, 0)
    if done:
        print(f"  {name}: se reanuda el estudio ({done} ensayos ya hechos, faltan {remaining})")
    if remaining == 0:
        return study
    if log:
        with mlflow.start_run(run_name=f"optuna-{slug}"):
            mlflow.set_tags(tags)
            study.optimize(objective, n_trials=remaining, catch=(NonFiniteCost,))
            if any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
                mlflow.log_metric("best_costo", study.best_value)
                mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
    else:
        study.optimize(objective, n_trials=remaining, catch=(NonFiniteCost,))
    return study


# --- Registro y exportación (US3) -----------------------------------------------------------------------


class _NameRecorder:
    """Ensayo falso que solo anota el nombre de cada parámetro, para saber en qué orden los pide un espacio."""

    def __init__(self):
        self.names = []

    def suggest_int(self, name, low, high, log=False):
        self.names.append(name)
        return low

    def suggest_float(self, name, low, high, log=False):
        self.names.append(name)
        return low


def trials_table(study: optuna.Study, name: str) -> pd.DataFrame:
    """Una fila por ensayo (también los fallidos) para `reports/06_ensayos.csv` (columnas de `data-model.md`).

    Los parámetros van en el orden del espacio de búsqueda; las métricas son medias entre pliegues (más la desviación del costo).
    """
    _check_finalist(name)
    recorder = _NameRecorder()
    FINALISTS[name].space(recorder)
    metric_columns = ["costo_media", "costo_desv"] + [f"{m}_media" for m in TRIAL_METRICS if m != "costo"] + ["fit_seconds_media"]
    rows = []
    for trial in study.trials:
        row = {"modelo": name, "ensayo": trial.number, "estado": trial.state.name}
        row.update({f"param_{p}": trial.params.get(p) for p in recorder.names})
        row.update({column: trial.user_attrs.get(column, np.nan) for column in metric_columns})
        row["mlflow_run_id"] = trial.user_attrs.get("mlflow_run_id")
        rows.append(row)
    columns = ["modelo", "ensayo", "estado"] + [f"param_{p}" for p in recorder.names] + metric_columns + ["mlflow_run_id"]
    return pd.DataFrame(rows, columns=columns)
