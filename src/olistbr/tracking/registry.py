"""Registro del modelo final (etapa 8, `especificaciones/08-registro.md`).

El **tracking** (etapa 6) guarda muchas ejecuciones de prueba: «¿qué probamos?». El **registro** guarda pocas **versiones del modelo**:
«¿qué modelo usamos y en qué versión?». El alias `champion` es una etiqueta móvil: quien use el modelo pide «el champion» sin importar
el número de versión. Usa la conexión de `mlflow_utils.py` (Databricks / Unity Catalog, o local de respaldo) sin reescribirla.

Aviso de seguridad: el modelo se guarda con `cloudpickle` (MLflow rechaza por defecto un `Pipeline` con preprocesamiento). Un archivo
pickle puede ejecutar código al cargarse: solo se cargan modelos del propio grupo.
"""
import mlflow
import pandas as pd
from mlflow.models import infer_signature

from olistbr.config import TOURNAMENT_GROUPS
from olistbr.features.engineering import feature_columns
from olistbr.tracking.mlflow_utils import registered_model_name, set_experiment

CHAMPION = "champion"


def model_columns() -> list:
    """Las columnas que usa el modelo (numéricas y categóricas de los 7 grupos del torneo), en orden."""
    columns = feature_columns(TOURNAMENT_GROUPS)
    return list(columns.numeric) + list(columns.categorical)


def input_example(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """`n` pedidos con **solo** las columnas del modelo, para registrar como ejemplo de entrada.

    Al tomar solo esas columnas quedan fuera el cliente, el pedido, el plazo prometido y el objetivo (regla 2: el ejemplo se sube a
    Databricks). `seller_id` sí queda: es el código anónimo del **vendedor** (una empresa) y una variable del modelo.
    """
    columns = model_columns()
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"a la tabla le faltan columnas del modelo: {', '.join(missing)}")
    example = df[columns].head(n).reset_index(drop=True)
    # Los enteros pasan a decimales: MLflow avisa que una columna entera en la firma no acepta datos faltantes al predecir,
    # y la app permite responder «No sé».
    integers = [c for c in example.columns if pd.api.types.is_integer_dtype(example[c])]
    return example.astype({c: "float64" for c in integers})


def register_final_model(pipeline, params: dict, metrics: dict, example: pd.DataFrame, name: str, experiment: str | None = None,
                         run_name: str = "modelo-final", tags: dict | None = None) -> dict:
    """Registra el pipeline ya ajustado como una nueva versión de `name`, en una ejecución `run_name`.

    Guarda parámetros, métricas, etiquetas, la **firma** (columnas y tipos de entrada y salida; Unity Catalog la exige) y el ejemplo.
    `name` se completa con el catálogo y el esquema en Unity Catalog (`registered_model_name`). Devuelve `run_id`, nombre y versión.
    """
    if experiment:
        set_experiment(experiment)
    full_name = registered_model_name(name)
    signature = infer_signature(example, pipeline.predict(example))
    with mlflow.start_run(run_name=run_name) as run:
        if params:
            mlflow.log_params(params)
        if metrics:
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
        if tags:
            mlflow.set_tags(tags)
        info = mlflow.sklearn.log_model(
            pipeline, name="model", input_example=example, signature=signature,
            registered_model_name=full_name, serialization_format="cloudpickle",
        )
    return {"run_id": run.info.run_id, "nombre": full_name, "version": int(info.registered_model_version)}


def set_champion(full_name: str, version: int | None = None) -> int:
    """Pone el alias `champion` en `version` (o, si no se indica, en la última versión) de `full_name` y devuelve ese número.

    Conviene pasar la versión recién registrada: si alguien registró otra entre medias, «la última» sería la equivocada.
    """
    if "'" in full_name:
        raise ValueError(f"nombre de modelo inválido: {full_name!r}")
    client = mlflow.MlflowClient()
    versions = [int(v.version) for v in client.search_model_versions(f"name = '{full_name}'")]
    if not versions:
        raise ValueError(f"no hay versiones registradas de {full_name!r}")
    chosen = max(versions) if version is None else int(version)
    if chosen not in versions:
        raise ValueError(f"la versión {chosen} no existe para {full_name!r} (hay: {sorted(versions)})")
    client.set_registered_model_alias(full_name, CHAMPION, str(chosen))
    return chosen


def load_champion(full_name: str):
    """Carga el modelo que tiene el alias `champion` (lo que usará la app, exportado en la etapa 9)."""
    return mlflow.sklearn.load_model(f"models:/{full_name}@{CHAMPION}")
