"""Agrupación de categorías raras (decisión 8 del EDA).

En el EDA, 1.714 de 3.095 vendedores tenían menos de 10 ítems (contados sobre todos los datos): con tan
pocos datos el modelo los memorizaría. Aquí el conteo es distinto: usa solo el entrenamiento y los ítems del
vendedor principal de cada pedido (`main_seller_items`), por lo que la cifra final no coincide con la del EDA. Qué vendedores son "frecuentes" depende de los datos, así que se decide SOLO con
el entrenamiento y esa misma lista se aplica a la prueba.
"""
import pandas as pd

from olistbr.config import MIN_SELLER_ITEMS, OTHER_LABEL


def frequent_values(
    df: pd.DataFrame, column: str, weight_col: str, min_count: int = MIN_SELLER_ITEMS
) -> list:
    """Valores de `column` cuyo total de `weight_col` llega a `min_count`.

    Llamar siempre con la tabla de ENTRENAMIENTO. `weight_col` es el número de ítems
    (`main_seller_items`) que ese valor aportó en cada pedido.
    """
    totals = df.groupby(column)[weight_col].sum()
    return list(totals[totals >= min_count].index)


def group_rare_categories(
    df: pd.DataFrame, column: str, frequent, other_label: str = OTHER_LABEL
) -> pd.DataFrame:
    """Reemplaza por `other_label` los valores de `column` que no están en `frequent`.

    Sirve igual para entrenamiento y prueba; un valor nuevo (que solo aparece en prueba) pasa a
    `other_label`. Devuelve una copia.
    """
    result = df.copy()
    result[column] = result[column].where(result[column].isin(set(frequent)), other_label)
    return result
