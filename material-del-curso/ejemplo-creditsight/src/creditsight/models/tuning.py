"""Optimización de hiperparámetros con Optuna, registrando cada ensayo en MLflow."""
from pathlib import Path

import mlflow
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

from creditsight.config import COST_MATRIX, RANDOM_STATE
from creditsight.models.train import build_pipeline, decide_min_cost, evaluate_folds
from creditsight.tracking.mlflow_utils import log_run


def lgbm_space(trial: optuna.Trial) -> dict:
    """Espacio de búsqueda de LightGBM; los rangos salen del conocimiento del algoritmo, no al azar."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 150, 400),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def rf_space(trial: optuna.Trial) -> dict:
    """Espacio de búsqueda de random forest. Los árboles se fijan en 200: lo que se afina es su forma."""
    return {
        "max_depth": trial.suggest_int("max_depth", 6, 24),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features": trial.suggest_float("max_features", 0.2, 1.0),
    }


def make_lgbm(params: dict, random_state: int = RANDOM_STATE) -> LGBMClassifier:
    return LGBMClassifier(subsample_freq=1, random_state=random_state, verbose=-1, n_jobs=2, **params)


def make_rf(params: dict, random_state: int = RANDOM_STATE) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=200, n_jobs=2, random_state=random_state, **params)


MODELS = {
    "lightgbm": (lgbm_space, make_lgbm, False),
    "random forest": (rf_space, make_rf, False),
}

# Métricas que se pueden optimizar y en qué dirección. Todas salen de `metrics_from_proba`.
OBJETIVOS = {
    "cost": "minimize",             # negocio: costo esperado con la matriz de costos
    "log_loss": "minimize",         # calidad de las probabilidades
    "auc_ovr": "maximize",          # técnica: capacidad de ordenar (AUC macro uno-contra-resto)
    "auc_poor": "maximize",         # técnica: AUC de la clase de interés
    "pr_auc_poor": "maximize",      # técnica: precisión-recall de la clase de interés
    "macro_f1": "maximize",
    "balanced_accuracy": "maximize",
    "accuracy": "maximize",
    "recall_poor": "maximize",      # cuidado: solo, empuja a marcar a todos como Poor; vigílalo junto a la precisión
}


def tune(name: str, X: pd.DataFrame, y: np.ndarray, cv, n_trials: int, seed: int = RANDOM_STATE, log: bool = True,
         storage: str | None = None, metric: str = "cost"):
    """Corre un estudio de Optuna que optimiza `metric` (ver OBJETIVOS). Cada ensayo queda como ejecución anidada en MLflow.

    En MLflow se registran TODAS las métricas de cada ensayo (negocio y técnicas, con su desviación entre pliegues),
    aunque solo una se optimice: así se puede ordenar y comparar la lista por cualquiera de ellas.

    Con `storage` (ruta de un archivo .db) el estudio se guarda en disco y **se reanuda**: si la ejecución se corta,
    al volver a llamar solo se hacen los ensayos que faltan hasta completar `n_trials`.
    """
    if metric not in OBJETIVOS:
        raise ValueError(f"Métrica no soportada: {metric}. Opciones: {sorted(OBJETIVOS)}")
    space_fn, make_fn, scale = MODELS[name]

    def objective(trial: optuna.Trial) -> float:
        params = space_fn(trial)
        folds = evaluate_folds(build_pipeline(make_fn(params), scale=scale), X, y, cv)
        trial.set_user_attr("cost_std", float(folds.cost.std()))
        trial.set_user_attr("log_loss", float(folds.log_loss.mean()))
        if log:
            board = {col: float(folds[col].mean()) for col in folds.columns}           # tablero completo (medias)
            board.update({f"{col}_std": float(folds[col].std()) for col in folds.columns if col != "fit_seconds"})
            log_run(
                f"{name}-trial-{trial.number}",
                params,
                board,
                tags={"etapa": "06_optimizacion", "modelo": name, "metrica_optimizada": metric},
                nested=True,
            )
        return float(folds[metric].mean())

    study = optuna.create_study(
        direction=OBJETIVOS[metric], sampler=optuna.samplers.TPESampler(seed=seed),
        study_name=f"creditsight-{name}-{metric}", storage=(f"sqlite:///{Path(storage).as_posix()}" if storage else None),
        load_if_exists=True,
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    done = len([t for t in study.trials if t.state.is_finished()])
    remaining = max(n_trials - done, 0)
    if done:
        print(f"  {name}: se reanuda el estudio ({done} ensayos ya hechos, faltan {remaining})")
    if log:
        with mlflow.start_run(run_name=f"optuna-{name}") as parent:
            study.optimize(objective, n_trials=remaining)
            mlflow.log_metric(f"best_{metric}", study.best_value)
            mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
            mlflow.set_tags({"etapa": "06_optimizacion", "modelo": name, "metrica_optimizada": metric})
            study.set_user_attr("mlflow_parent_run_id", parent.info.run_id)
        study.set_user_attr("metric", metric)
    else:
        study.optimize(objective, n_trials=remaining)
    return study


def bootstrap_cost_difference(y_true, proba_a, proba_b, groups, n_boot: int = 1000, seed: int = RANDOM_STATE):
    """Diferencia de costo (A menos B) con intervalo del 95 %, remuestreando CLIENTES completos.

    Remuestrear filas sería otra fuga: los 8 meses de un cliente no son independientes.
    Un valor negativo significa que A cuesta menos que B.
    """
    y_true = np.asarray(y_true)
    cost_a = COST_MATRIX[y_true, decide_min_cost(proba_a)]
    cost_b = COST_MATRIX[y_true, decide_min_cost(proba_b)]
    per_customer = pd.DataFrame({"g": np.asarray(groups), "diff": cost_a - cost_b}).groupby("g")["diff"].agg(["sum", "count"])
    sums, counts = per_customer["sum"].to_numpy(), per_customer["count"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), size=(n_boot, len(sums)))
    boot = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return float(sums.sum() / counts.sum()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def bootstrap_metric_ci(y_true, proba, groups, metric_fn, n_boot: int = 300, seed: int = RANDOM_STATE):
    """Una métrica con su intervalo del 95 %, remuestreando CLIENTES completos (no filas).

    `metric_fn(y_true, proba)` devuelve un número. Devuelve (valor, límite inferior, límite superior).
    """
    y_true = np.asarray(y_true)
    groups = np.asarray(groups)
    index_by_customer = pd.Series(np.arange(len(groups))).groupby(groups).apply(lambda s: s.to_numpy()).to_numpy()
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        picked = rng.integers(0, len(index_by_customer), len(index_by_customer))
        rows = np.concatenate([index_by_customer[i] for i in picked])
        values.append(metric_fn(y_true[rows], proba[rows]))
    return float(metric_fn(y_true, proba)), float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def bootstrap_metric_difference(y_true, proba_a, proba_b, groups, metric_fn, n_boot: int = 300, seed: int = RANDOM_STATE):
    """Diferencia de una métrica cualquiera (A menos B) con intervalo del 95 %, remuestreando CLIENTES completos.

    `metric_fn(y_true, proba)` devuelve un número, p. ej. `lambda y, p: roc_auc_score(y, p, multi_class="ovr")`.
    Un valor positivo significa que A obtiene una métrica mayor que B (para métricas donde más es mejor).
    """
    y_true = np.asarray(y_true)
    groups = np.asarray(groups)
    index_by_customer = pd.Series(np.arange(len(groups))).groupby(groups).apply(lambda s: s.to_numpy()).to_numpy()
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        picked = rng.integers(0, len(index_by_customer), len(index_by_customer))
        rows = np.concatenate([index_by_customer[i] for i in picked])
        diffs.append(metric_fn(y_true[rows], proba_a[rows]) - metric_fn(y_true[rows], proba_b[rows]))
    full = metric_fn(y_true, proba_a) - metric_fn(y_true, proba_b)
    return float(full), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
