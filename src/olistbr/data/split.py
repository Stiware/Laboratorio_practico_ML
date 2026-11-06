"""Partición entrenamiento/prueba por grupo (decisión 5 del EDA).

Una misma persona (`customer_unique_id`) puede tener varios pedidos (hasta 15 en la tabla final).
Si esos pedidos se reparten entre entrenamiento y prueba, el modelo "reconoce" al cliente y la métrica
sale mejor de lo que será en producción. Por eso **todas** las particiones respetan al cliente completo.
"""
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from olistbr.config import (
    FORWARD_TRAIN_END,
    FORWARD_VALID_END,
    FORWARD_VALID_START,
    GROUP_COL,
    MIN_SELLER_ITEMS,
    N_FOLDS,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)
from olistbr.features.categories import frequent_values, group_rare_categories


def assert_disjoint_groups(*frames: pd.DataFrame, group_col: str = GROUP_COL) -> None:
    """Falla si algún grupo (cliente) aparece en más de una partición."""
    seen: set = set()
    for i, frame in enumerate(frames):
        groups = set(frame[group_col])
        overlap = seen & groups
        if overlap:
            raise AssertionError(f"{len(overlap)} grupos aparecen en más de una partición (partición {i})")
        seen |= groups


def split_by_group(
    df: pd.DataFrame,
    group_col: str = GROUP_COL,
    test_size: float = TEST_SIZE,
    seed: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parte `df` en `(train, test)` sin que ningún grupo aparezca en los dos.

    Con la misma semilla la partición es siempre la misma. El `test` queda cerrado:
    no se usa para decidir nada hasta la etapa 7.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, groups=df[group_col]))
    train = df.iloc[train_idx].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)
    assert_disjoint_groups(train, test, group_col=group_col)
    return train, test


def prepare_train_test(
    orders: pd.DataFrame,
    group_col: str = GROUP_COL,
    test_size: float = TEST_SIZE,
    seed: int = RANDOM_STATE,
    seller_col: str = "seller_id",
    weight_col: str = "main_seller_items",
    min_seller_items: int = MIN_SELLER_ITEMS,
) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """Parte por cliente y agrupa los vendedores raros usando SOLO el entrenamiento.

    El orden es lo importante y por eso vive aquí y no en el notebook:
    1. se parte por cliente (`split_by_group`);
    2. la lista de vendedores frecuentes se calcula con `train` solamente;
    3. esa misma lista se aplica a `train` y a `test` (lo que no está en ella pasa a "otros").

    Devuelve `(train, test, frequent_sellers)`.
    """
    train, test = split_by_group(orders, group_col=group_col, test_size=test_size, seed=seed)
    frequent_sellers = frequent_values(train, seller_col, weight_col, min_count=min_seller_items)
    train = group_rare_categories(train, seller_col, frequent_sellers)
    test = group_rare_categories(test, seller_col, frequent_sellers)
    return train, test, frequent_sellers


def make_group_folds(
    df: pd.DataFrame,
    n_splits: int = N_FOLDS,
    group_col: str = GROUP_COL,
    seed: int = RANDOM_STATE,
) -> list:
    """Pliegues de validación cruzada por cliente: devuelve `[(índices_entrenamiento, índices_validación), ...]`.

    Se calculan UNA vez y se reutilizan en todas las comparaciones (misma validación para cada idea).
    Ningún cliente cae en el entrenamiento y la validación del mismo pliegue; se verifica antes de devolver.
    """
    splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = [(train_idx, valid_idx) for train_idx, valid_idx in splitter.split(df, groups=df[group_col])]
    for train_idx, valid_idx in folds:
        assert_disjoint_groups(df.iloc[train_idx], df.iloc[valid_idx], group_col=group_col)
    return folds


def forward_split(
    df: pd.DataFrame,
    train_end: str = FORWARD_TRAIN_END,
    valid_start: str = FORWARD_VALID_START,
    valid_end: str = FORWARD_VALID_END,
    group_col: str = GROUP_COL,
    date_col: str = "order_purchase_timestamp",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partición «hacia adelante»: entrena con el pasado y valida con compras posteriores.

    Imita el uso real (predecir lo que viene) y sirve de guardia contra la tendencia entre años. Los días
    `train_end` y `valid_end` se incluyen completos. **El entrenamiento no incluye a los clientes que
    aparecen en la validación**, para que ningún cliente esté en los dos lados. Solo usa la tabla que recibe
    (el conjunto de entrenamiento); nunca la prueba final.
    """
    if pd.Timestamp(train_end) >= pd.Timestamp(valid_start):
        raise ValueError(f"el entrenamiento ({train_end}) debe terminar antes de que empiece la validación ({valid_start})")
    dates = df[date_col]
    one_day = pd.Timedelta(days=1)
    in_valid = (dates >= pd.Timestamp(valid_start)) & (dates < pd.Timestamp(valid_end) + one_day)
    in_train = dates < pd.Timestamp(train_end) + one_day

    valid = df[in_valid]
    if valid.empty:
        raise ValueError(f"no hay compras entre {valid_start} y {valid_end}: la validación hacia adelante está vacía")
    train = df[in_train & ~df[group_col].isin(set(valid[group_col]))]
    assert_disjoint_groups(train, valid, group_col=group_col)
    return train.reset_index(drop=True), valid.reset_index(drop=True)


def reserved_forward_split(
    search: pd.DataFrame,
    reserved: pd.DataFrame,
    train_end: str = FORWARD_TRAIN_END,
    valid_start: str = FORWARD_VALID_START,
    valid_end: str = FORWARD_VALID_END,
    group_col: str = GROUP_COL,
    date_col: str = "order_purchase_timestamp",
    target: str = TARGET,
    known_at_cutoff: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prueba hacia adelante de la optimización: entrena con el pasado de la **búsqueda** y valida con el futuro de la **reserva**.

    La de `forward_split` ya no sirve después de la búsqueda, porque sus meses de validación incluyen clientes que la búsqueda vio
    (Q6 de `especificaciones/06-optimizacion.md`). Aquí la validación solo tiene clientes reservados, que la búsqueda nunca vio.
    Los días `train_end` y `valid_end` se incluyen completos. Es informativa: no cambia ninguna decisión.

    Con `known_at_cutoff=True` (por defecto) el entrenamiento solo trae pedidos **ya entregados** antes de `valid_start`
    (compra + `target` días): una compra de fines de febrero entregada en marzo no se conocía al corte (auditor-de-fuga).
    """
    if pd.Timestamp(train_end) >= pd.Timestamp(valid_start):
        raise ValueError(f"el entrenamiento ({train_end}) debe terminar antes de que empiece la validación ({valid_start})")
    one_day = pd.Timedelta(days=1)
    train = search[search[date_col] < pd.Timestamp(train_end) + one_day]
    if known_at_cutoff:
        delivered = train[date_col] + pd.to_timedelta(train[target], unit="D")
        train = train[delivered < pd.Timestamp(valid_start)]
    dates = reserved[date_col]
    valid = reserved[(dates >= pd.Timestamp(valid_start)) & (dates < pd.Timestamp(valid_end) + one_day)]
    if valid.empty:
        raise ValueError(f"no hay compras de clientes reservados entre {valid_start} y {valid_end}: la validación está vacía")
    if train.empty:
        raise ValueError(f"no hay compras de la búsqueda hasta {train_end}: el entrenamiento está vacío")
    assert_disjoint_groups(train, valid, group_col=group_col)
    return train.reset_index(drop=True), valid.reset_index(drop=True)
