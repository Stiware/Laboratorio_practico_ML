"""Pruebas de la partición por cliente (decisión 5): sin fuga, completa y reproducible."""
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import ShuffleSplit

import olistbr.data.split as split_module
from olistbr.data.split import assert_disjoint_groups, forward_split, make_group_folds, prepare_train_test, split_by_group
from olistbr.features.categories import frequent_values


def _orders(n_customers=100, per_customer=5):
    """Clientes con varios pedidos cada uno (el caso que causa fuga si se parte por fila)."""
    rows = [
        {"order_id": f"o{c}_{k}", "customer_unique_id": f"u{c}", "dias_entrega": float(k)}
        for c in range(n_customers)
        for k in range(per_customer)
    ]
    return pd.DataFrame(rows)


def test_split_by_group_no_overlap():
    """Ningún cliente cae en dos particiones y no se pierde ninguna fila."""
    df = _orders()
    train, test = split_by_group(df)
    assert set(train["customer_unique_id"]).isdisjoint(set(test["customer_unique_id"]))
    assert len(train) + len(test) == len(df)
    assert set(train["order_id"]) | set(test["order_id"]) == set(df["order_id"])
    # un cliente con 5 pedidos queda entero de un solo lado
    for _, group in df.groupby("customer_unique_id"):
        in_train = group["order_id"].isin(train["order_id"]).sum()
        assert in_train in (0, len(group))


def test_split_proportion_is_about_20_percent_of_customers():
    df = _orders(n_customers=200)
    _, test = split_by_group(df)
    assert test["customer_unique_id"].nunique() == 40  # 20 % de 200 clientes


def test_split_is_reproducible():
    df = _orders()
    train_a, test_a = split_by_group(df)
    train_b, test_b = split_by_group(df)
    pd.testing.assert_frame_equal(train_a, train_b)
    pd.testing.assert_frame_equal(test_a, test_b)


def test_row_split_overlaps():
    """Contraste: partir por fila SÍ deja clientes en train y test.

    Demuestra que la verificación de arriba sirve: si alguien cambiara la partición por fila,
    `test_split_by_group_no_overlap` fallaría.
    """
    df = _orders()
    train_idx, test_idx = next(ShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(df))
    shared = set(df.iloc[train_idx]["customer_unique_id"]) & set(df.iloc[test_idx]["customer_unique_id"])
    assert len(shared) > 0


def test_split_by_group_fails_loudly_if_it_ever_splits_by_row(monkeypatch):
    """Contraste sobre la función REAL: si `split_by_group` partiera por fila, fallaría con un error.

    Se cambia temporalmente el separador por uno que ignora los grupos (parte por fila). La función
    comprueba por sí misma que no queden clientes en los dos lados, así que debe lanzar AssertionError
    en vez de devolver datos con fuga. Y `test_split_by_group_no_overlap` fallaría igualmente.
    """
    class RowSplitter:
        def __init__(self, n_splits, test_size, random_state):
            self._inner = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)

        def split(self, X, groups=None):  # ignora `groups` a propósito
            return self._inner.split(X)

    monkeypatch.setattr(split_module, "GroupShuffleSplit", RowSplitter)
    with pytest.raises(AssertionError, match="grupos aparecen en más de una partición"):
        split_by_group(_orders())


def _orders_with_sellers():
    df = _orders()
    df["seller_id"] = "s_comun"
    df["main_seller_items"] = 2
    return df


def test_prepare_train_test_learns_sellers_only_from_train():
    """Un vendedor muy frecuente SOLO en la prueba no debe aparecer en la lista ni conservar su nombre."""
    df = _orders_with_sellers()
    _, test_customers = split_by_group(df)[0], set(split_by_group(df)[1]["customer_unique_id"])
    # el vendedor "s_solo_test" vende muchísimo, pero únicamente a clientes que caen en la prueba
    df.loc[df["customer_unique_id"].isin(test_customers), ["seller_id", "main_seller_items"]] = ["s_solo_test", 100]

    train, test, frequent = prepare_train_test(df)

    assert "s_solo_test" not in frequent                      # no se aprendió de la prueba
    assert set(test["seller_id"]) == {"otros"}                # en la prueba pasa a "otros"
    assert "s_solo_test" not in set(train["seller_id"])       # y no existe en entrenamiento
    assert set(train["seller_id"]) <= {"s_comun", "otros"}


def test_prepare_train_test_matches_manual_steps():
    """La función hace lo mismo que partir y luego calcular la lista con train (y solo con train)."""
    df = _orders_with_sellers()
    train_raw, _ = split_by_group(df)
    expected = frequent_values(train_raw, "seller_id", "main_seller_items", min_count=10)
    train, test, frequent = prepare_train_test(df)
    assert sorted(frequent) == sorted(expected)
    assert set(train["customer_unique_id"]).isdisjoint(set(test["customer_unique_id"]))
    assert len(train) + len(test) == len(df)


def test_assert_disjoint_groups_raises_on_overlap():
    """La verificación falla (AssertionError) si un cliente está en dos particiones."""
    df = _orders()
    with pytest.raises(AssertionError):
        assert_disjoint_groups(df.iloc[:20], df.iloc[10:30])


def test_assert_disjoint_groups_accepts_clean_partition():
    train, test = split_by_group(_orders())
    assert_disjoint_groups(train, test)  # no debe fallar


# ---------------------------------- Etapa 4: pliegues por cliente ----------------------------------


def test_group_folds_do_not_share_customers():
    """En cada pliegue ningún cliente está en el entrenamiento y en la validación a la vez."""
    df = _orders(100, 5)
    folds = make_group_folds(df, n_splits=5)
    assert len(folds) == 5
    for train_idx, valid_idx in folds:
        assert set(df.iloc[train_idx]["customer_unique_id"]).isdisjoint(set(df.iloc[valid_idx]["customer_unique_id"]))


def test_group_folds_cover_each_row_once():
    """Los 5 pliegues de validación suman todas las filas, sin repetir ninguna."""
    df = _orders(100, 5)
    folds = make_group_folds(df, n_splits=5)
    validation = np.concatenate([valid_idx for _, valid_idx in folds])
    assert sorted(validation) == list(range(len(df)))
    for train_idx, valid_idx in folds:
        assert len(train_idx) + len(valid_idx) == len(df)


def test_group_folds_are_reproducible():
    df = _orders(100, 5)
    first = make_group_folds(df, n_splits=5, seed=42)
    second = make_group_folds(df, n_splits=5, seed=42)
    for (train_a, valid_a), (train_b, valid_b) in zip(first, second):
        np.testing.assert_array_equal(valid_a, valid_b)
        np.testing.assert_array_equal(train_a, train_b)


# ------------------------------ Etapa 4: prueba hacia adelante (por fechas) ------------------------------


def _purchases():
    """Compras de clientes hechos a mano alrededor de la fecha de corte 2018-02-28 / 2018-03-01."""
    rows = [
        ("A", "2018-01-10 10:00"),  # A compró antes y después: queda fuera del entrenamiento
        ("A", "2018-04-10 10:00"),
        ("B", "2017-05-02 09:00"),  # solo entrenamiento
        ("C", "2018-05-20 09:00"),  # solo validación
        ("D", "2018-02-28 23:30"),  # último momento del entrenamiento
        ("E", "2018-03-01 00:00"),  # primer momento de la validación
        ("F", "2018-09-01 10:00"),  # fuera de los dos periodos
        ("G", "2018-08-31 22:00"),  # último día de la validación
    ]
    df = pd.DataFrame(rows, columns=["customer_unique_id", "order_purchase_timestamp"])
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    df["order_id"] = [f"o{i}" for i in range(len(df))]
    return df


def test_forward_split_respects_dates():
    train, valid = forward_split(_purchases())
    assert train["order_purchase_timestamp"].max() < pd.Timestamp("2018-03-01")
    assert valid["order_purchase_timestamp"].min() >= pd.Timestamp("2018-03-01")
    assert valid["order_purchase_timestamp"].max() < pd.Timestamp("2018-09-01")
    assert set(train["customer_unique_id"]) == {"B", "D"}
    assert set(valid["customer_unique_id"]) == {"A", "C", "E", "G"}  # F (septiembre) no está en ninguno


def test_forward_split_has_no_shared_customers():
    """A compró en los dos periodos: su compra de enero NO queda en el entrenamiento."""
    train, valid = forward_split(_purchases())
    assert set(train["customer_unique_id"]).isdisjoint(set(valid["customer_unique_id"]))
    assert "A" not in set(train["customer_unique_id"])
    assert (valid["customer_unique_id"] == "A").sum() == 1  # su compra de abril sí está en la validación


def test_forward_split_rejects_overlapping_dates():
    with pytest.raises(ValueError, match="debe terminar antes"):
        forward_split(_purchases(), train_end="2018-03-01", valid_start="2018-03-01")


def test_forward_split_rejects_empty_validation():
    with pytest.raises(ValueError, match="vacía"):
        forward_split(_purchases(), valid_start="2030-01-01", valid_end="2030-02-01")
