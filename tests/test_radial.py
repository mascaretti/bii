# tests/test_radial.py

import numpy as np
import jax
import jax.numpy as jnp

from bii.radial import (
    make_distance,
    make_lexico_dag,
    make_partition_indices,
    split_XZ_by_partition,
)


def _toy_data(n=20, p=2, seed=0):
    key = jax.random.PRNGKey(seed)
    key, k1 = jax.random.split(key)
    X = jax.random.multivariate_normal(k1, jnp.zeros(p), jnp.eye(p), shape=(n,))
    y = 0.02 * jax.random.multivariate_normal(k1, jnp.zeros(p), jnp.eye(p))
    Y_rows = y[None, :]
    return Y_rows, X, key


def test_make_distance_sorted_and_indices():
    Y_rows, Y_cols, _ = _toy_data(n=7, p=2, seed=1)
    DY_sorted, cols_sorted = make_distance(Y_rows, Y_cols)
    d_sorted = np.array(DY_sorted[0, :, 0])
    assert np.all(d_sorted[1:] >= d_sorted[:-1])
    c_sorted = np.array(cols_sorted[0])
    c_from_DY = np.array(DY_sorted[0, :, 1])
    assert np.array_equal(c_sorted, c_from_DY)


def test_make_lexico_dag_matches_pair_count():
    Y_rows, Y_cols, _ = _toy_data(n=10, p=2, seed=3)
    _, cols_sorted = make_distance(Y_rows, Y_cols)
    dag_pairs, mask, counts = make_lexico_dag(cols_sorted, k=4)
    assert dag_pairs.shape[1] == 3  # k-1 consecutive pairs
    assert mask.sum() == counts.sum()


def test_partition_disjoint_and_cover():
    n = 31
    p = 2
    X = jnp.arange(n * p, dtype=jnp.float32).reshape(n, p)
    Z = X * 2.0

    key = jax.random.PRNGKey(123)
    rows_idx, cols_idx = make_partition_indices(key, n, size_rows=12, shuffle=True)
    rset = set(map(int, np.array(rows_idx)))
    cset = set(map(int, np.array(cols_idx)))
    assert rset.isdisjoint(cset)
    full = rset | cset
    assert full == set(range(n))

    Xr, Xc, Zr, Zc = split_XZ_by_partition(X, Z, rows_idx, cols_idx)
    assert Xr.shape[0] == len(rset)
    assert Xc.shape[0] == len(cset)
    assert Zr.shape[0] == len(rset)
    assert Zc.shape[0] == len(cset)
