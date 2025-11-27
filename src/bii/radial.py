# radial_binning.py
# -*- coding: utf-8 -*-
"""
Minimal JAX utilities for:
  - pairwise distances (sorted, with indices)
  - deterministic k-nearest-based DAG construction
  - reproducible two-way partition of indices
"""

from typing import Tuple, Optional
import jax
import jax.numpy as jnp
from jax import random

Array = jnp.ndarray


@jax.jit
def make_distance(Y_rows: Array, Y_cols: Array) -> Tuple[Array, Array]:
    """Pairwise distances sorted ascending with sorted column indices."""
    diffs = Y_rows[:, None, :] - Y_cols[None, :, :]
    dists = jnp.linalg.norm(diffs, axis=-1)
    n, m = dists.shape
    col_idx = jnp.broadcast_to(jnp.arange(m)[None, :], (n, m))
    sort_idx = jnp.lexsort(keys=(col_idx, dists))
    d_sorted = jnp.take_along_axis(dists, sort_idx, axis=1)
    cols_sorted = jnp.take_along_axis(col_idx, sort_idx, axis=1).astype(jnp.int32)
    DY_sorted = jnp.stack([d_sorted, cols_sorted.astype(d_sorted.dtype)], axis=-1)
    DY_sorted = DY_sorted.at[..., 1].set(cols_sorted)
    return DY_sorted, cols_sorted


def make_lexico_dag(sorted_indices: Array, k: int) -> Tuple[Array, Array, Array]:
    """
    Build pairs from the first k neighbors per row by consecutive grouping:
    (i1, i2), (i3, i4), ...
    Returns (pairs, mask, counts).
    """
    idx = jnp.asarray(sorted_indices, dtype=jnp.int32)
    squeeze = False
    if idx.ndim == 1:
        idx = idx[None, :]
        squeeze = True
    n, m = idx.shape
    k_eff = min(k, m)
    num_pairs = k_eff // 2
    if num_pairs == 0:
        shape_pairs = (idx.shape[0], 0, 2)
        dag_pairs = -jnp.ones(shape_pairs, dtype=jnp.int32)
        mask = jnp.zeros(shape_pairs[:-1], dtype=jnp.bool_)
        counts = jnp.zeros((idx.shape[0],), dtype=jnp.int32)
    else:
        neighbors = idx[:, : num_pairs * 2]
        pairs = neighbors.reshape(n, num_pairs, 2)
        valid = jnp.all(pairs >= 0, axis=-1)
        dag_pairs = jnp.where(valid[..., None], pairs, -1)
        mask = valid
        counts = mask.sum(axis=1, dtype=jnp.int32)
    if squeeze:
        dag_pairs = dag_pairs[0]
        mask = mask[0]
        counts = counts[0]
    return dag_pairs, mask, counts


def _normalize_sizes(
    n: int,
    size_rows: Optional[int],
    size_cols: Optional[int],
    ratio_rows: Optional[float],
) -> Tuple[int, int]:
    if size_rows is not None and size_cols is not None:
        if size_rows + size_cols != n:
            raise ValueError(f"size_rows + size_cols must equal n={n}.")
        return int(size_rows), int(size_cols)
    if size_rows is not None:
        if not (0 <= size_rows <= n):
            raise ValueError("size_rows out of range.")
        return int(size_rows), int(n - size_rows)
    if size_cols is not None:
        if not (0 <= size_cols <= n):
            raise ValueError("size_cols out of range.")
        return int(n - size_cols), int(size_cols)
    if ratio_rows is not None:
        if not (0.0 < ratio_rows < 1.0):
            raise ValueError("ratio_rows must be in (0,1).")
        s_rows = int(jnp.floor(ratio_rows * n))
        s_rows = int(jnp.clip(s_rows, 0, n))
        return s_rows, n - s_rows
    s_rows = n // 2
    return s_rows, n - s_rows


def make_partition_indices(
    key: jax.Array,
    n: int,
    *,
    size_rows: Optional[int] = None,
    size_cols: Optional[int] = None,
    ratio_rows: Optional[float] = None,
    shuffle: bool = True,
) -> Tuple[Array, Array]:
    s_rows, s_cols = _normalize_sizes(n, size_rows, size_cols, ratio_rows)
    if shuffle:
        perm = random.permutation(key, jnp.arange(n, dtype=jnp.int32))
    else:
        perm = jnp.arange(n, dtype=jnp.int32)
    return perm[:s_rows], perm[s_rows:]


def split_XZ_by_partition(
    X: Array,
    Z: Array,
    rows_idx: Array,
    cols_idx: Array,
) -> Tuple[Array, Array, Array, Array]:
    if X.shape[0] != Z.shape[0]:
        raise ValueError("X and Z must have the same number of rows.")
    return X[rows_idx], X[cols_idx], Z[rows_idx], Z[cols_idx]
