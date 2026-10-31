"""Partición entrenamiento/prueba por grupo (decisión 5 del EDA).

Una misma persona (`customer_unique_id`) puede tener varios pedidos (hasta 15 en la tabla final).
Si esos pedidos se reparten entre entrenamiento y prueba, el modelo "reconoce" al cliente y la métrica
sale mejor de lo que será en producción. Por eso **todas** las particiones respetan al cliente completo.
"""
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from olistbr.config import GROUP_COL, MIN_SELLER_ITEMS, RANDOM_STATE, TEST_SIZE
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
