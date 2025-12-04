# radial_binning.py
# -*- coding: utf-8 -*-
"""
Minimal JAX utilities for:
  - pairwise distances (sorted, with indices)
  - PPP-motivated shells and representative selection
  - deterministic k-nearest-based DAG construction
  - reproducible two-way partition of indices
"""

from typing import Tuple, Optional
import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.special import gammaln

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


def nearest_two_pairs(
    X_rows: Array,
    Z_rows: Array,
    X_cols: Array,
    Z_cols: Array,
    weights: Array,
) -> Tuple[Array, Array, Array]:
    """
    Build one pair per anchor using the two closest columns under a weighted metric.

    Returns:
        dag_pairs : (n_rows, 1, 2) int32, -1 if fewer than 2 cols.
        mask      : (n_rows, 1) bool
        counts    : (n_rows,) int32
    """
    X_rows = jnp.asarray(X_rows); Z_rows = jnp.asarray(Z_rows)
    X_cols = jnp.asarray(X_cols); Z_cols = jnp.asarray(Z_cols)
    w = jnp.asarray(weights)

    if Z_rows.ndim == 1:
        Z_rows = Z_rows[None, :]
        X_rows = X_rows[None, :]

    # Weighted distances in Z-space
    diff = Z_rows[:, None, :] - Z_cols[None, :, :]
    dists = jnp.sqrt(jnp.sum(w * diff * diff, axis=-1))
    col_idx = jnp.broadcast_to(jnp.arange(Z_cols.shape[0])[None, :], dists.shape)
    sort_idx = jnp.lexsort(keys=(col_idx, dists))
    top2 = jnp.take_along_axis(col_idx, sort_idx, axis=1)[:, :2]  # (n_rows, 2)

    # If fewer than 2 cols, mark invalid
    valid = Z_cols.shape[0] >= 2
    if not valid:
        dag_pairs = -jnp.ones((Z_rows.shape[0], 0, 2), dtype=jnp.int32)
        mask = jnp.zeros((Z_rows.shape[0], 0), dtype=jnp.bool_)
        counts = jnp.zeros((Z_rows.shape[0],), dtype=jnp.int32)
    else:
        dag_pairs = top2.reshape(Z_rows.shape[0], 1, 2).astype(jnp.int32)
        mask = jnp.ones((Z_rows.shape[0], 1), dtype=jnp.bool_)
        counts = jnp.ones((Z_rows.shape[0],), dtype=jnp.int32)
    return dag_pairs, mask, counts


# ------------------------------ PPP shells (equal expected counts) ------------------------------


def _unit_ball_volume(d: int) -> Array:
    """V_d = pi^(d/2) / Gamma(1 + d/2)."""
    d = float(d)
    logV = (d / 2.0) * jnp.log(jnp.pi) - gammaln(1.0 + d / 2.0)
    return jnp.exp(logV)


def equal_expected_count_shells_from_Rmax(Rmax: float, num_shells: int, dim: int) -> Array:
    """
    Equal expected counts under homogeneous PPP via equal *volume* shells.
    In d dimensions: r_i = Rmax * ((i+1)/S)^(1/d) for i=0..S-1.
    """
    if num_shells < 1:
        raise ValueError("num_shells must be >= 1")
    if Rmax <= 0:
        raise ValueError("Rmax must be > 0")
    dtype = jnp.asarray(Rmax).dtype
    d = jnp.asarray(float(dim), dtype=dtype)
    i = jnp.arange(1, num_shells + 1, dtype=dtype)
    denom = jnp.asarray(float(num_shells), dtype=dtype)
    return jnp.asarray(Rmax, dtype=dtype) * (i / denom) ** (1.0 / d)


def equal_expected_count_shells_via_lambda(
    lambda_hat: float,
    target_expected_per_shell: float,
    num_shells: int,
    dim: int,
    r0: float = 0.0,
) -> Array:
    """
    Equal expected counts under homogeneous PPP with intensity lambda_hat:
      Δ = target / (lambda_hat * V_d),  r_i^d = r_{i-1}^d + Δ => r_i = ( r0^d + (i+1)*Δ )^(1/d)
    """
    if num_shells < 1:
        raise ValueError("num_shells must be >= 1")
    if lambda_hat <= 0:
        raise ValueError("lambda_hat must be > 0")
    if target_expected_per_shell <= 0:
        raise ValueError("target_expected_per_shell must be > 0")

    dtype = jnp.result_type(jnp.asarray(lambda_hat), jnp.asarray(target_expected_per_shell))
    Vd = _unit_ball_volume(dim).astype(dtype)
    Delta = (
        jnp.asarray(target_expected_per_shell, dtype=dtype)
        / (jnp.asarray(lambda_hat, dtype=dtype) * Vd)
    )
    d = jnp.asarray(float(dim), dtype=dtype)
    base = (jnp.asarray(r0, dtype=dtype) ** d)
    i = jnp.arange(1, num_shells + 1, dtype=dtype)
    return (base + i * Delta) ** (1.0 / d)


def assign_to_shells_aligned(
    DY_sorted: Array,
    cols_sorted: Array,
    radii: Array,
) -> Tuple[Array, Array]:
    """
    Assign distances to shells with labels aligned both to sorted order and original columns.
    """
    d_sorted = DY_sorted[..., 0]
    S = radii.shape[0]
    radii = radii.astype(d_sorted.dtype)
    eps = jnp.finfo(d_sorted.dtype).eps
    tol_per_shell = 8.0 * eps * jnp.maximum(1.0, jnp.abs(radii))
    exceeds = d_sorted[..., None] > (radii[None, None, :] + tol_per_shell)
    shell_candidates = jnp.sum(exceeds, axis=-1).astype(jnp.int32)

    max_r = radii[-1]
    outer_tol = tol_per_shell[-1]
    within_outer = d_sorted <= (max_r + outer_tol)
    shell_sorted = jnp.where(within_outer, jnp.minimum(shell_candidates, S - 1), -1)

    def remap_row(shell_row, cols_row):
        out = -jnp.ones_like(shell_row, dtype=jnp.int32)
        out = out.at[cols_row].set(shell_row)
        return out

    shell_by_col = jax.vmap(remap_row)(shell_sorted, cols_sorted)
    return shell_sorted, shell_by_col


def sample_representatives_uniform_aligned(
    DY_sorted: Array,
    shell_sorted: Array,
    radii: Array,
    key: jax.Array,
) -> Tuple[Array, Array]:
    """
    For each row and shell, pick one representative uniformly at random.
    Returns representatives aligned to original columns.
    """
    d_sorted = DY_sorted[..., 0]
    col_sorted = DY_sorted[..., 1].astype(jnp.int32)
    n, m = d_sorted.shape
    S = radii.shape[0]
    keys = random.split(key, n * S).reshape(n, S, 2)

    def sample_row(row_shells, row_cols, row_dists, row_keys):
        def sample_one_shell(s_key, s_idx):
            mask = (row_shells == s_idx)
            any_in_shell = jnp.any(mask)
            g = random.gumbel(s_key, (m,))
            logits = jnp.where(mask, 0.0, -jnp.inf)
            pos = jnp.argmax(logits + g)
            col = jnp.where(any_in_shell, row_cols[pos], -1)
            dist = jnp.where(any_in_shell, row_dists[pos], jnp.nan)
            return col, dist

        shell_ids = jnp.arange(S, dtype=jnp.int32)
        cols_s, dists_s = jax.vmap(sample_one_shell)(row_keys, shell_ids)
        return cols_s, dists_s

    rep_cols, rep_dists = jax.vmap(sample_row)(shell_sorted, col_sorted, d_sorted, keys)
    return rep_cols, rep_dists


def select_representatives_first_in_shell(
    DY_sorted: Array,
    shell_sorted: Array,
    num_shells: int,
) -> Tuple[Array, Array]:
    """Deterministically pick the closest column in each shell."""
    d_sorted = DY_sorted[..., 0]
    col_sorted = DY_sorted[..., 1].astype(jnp.int32)
    n, m = d_sorted.shape
    S = int(num_shells)
    shell_ids = jnp.arange(S, dtype=jnp.int32)
    positions = jnp.arange(m, dtype=jnp.int32)

    def select_row(row_shells, row_cols, row_dists):
        def select_shell(s_idx):
            mask = row_shells == s_idx
            any_in_shell = jnp.any(mask)
            pos = jnp.min(jnp.where(mask, positions, m))
            col = jnp.where(any_in_shell, row_cols[pos], -1)
            dist = jnp.where(any_in_shell, row_dists[pos], jnp.nan)
            return col, dist

        cols_s, dists_s = jax.vmap(select_shell)(shell_ids)
        return cols_s, dists_s

    rep_cols, rep_dists = jax.vmap(select_row)(shell_sorted, col_sorted, d_sorted)
    return rep_cols, rep_dists


def select_representatives_by_rank(
    DY_sorted: Array,
    num_shells: int,
) -> Tuple[Array, Array]:
    """Deterministically pick the k-th nearest neighbor for the k-th annulus."""
    d_sorted = DY_sorted[..., 0]
    col_sorted = DY_sorted[..., 1].astype(jnp.int32)
    n, m = d_sorted.shape
    S = int(num_shells)
    positions = jnp.arange(S, dtype=jnp.int32)

    def select_row(row_cols, row_dists):
        valid = positions < m
        cols = jnp.where(valid, row_cols[positions], -1)
        dists = jnp.where(valid, row_dists[positions], jnp.nan)
        return cols, dists

    rep_cols, rep_dists = jax.vmap(select_row)(col_sorted, d_sorted)
    return rep_cols, rep_dists


def make_shell_dag_pairs(rep_cols_by_shell: Array) -> Tuple[Array, Array, Array]:
    """Build all admissible pairs from shell representatives (directed j>i)."""
    rep_cols = jnp.asarray(rep_cols_by_shell, dtype=jnp.int32)
    squeeze = (rep_cols.ndim == 1)
    if rep_cols.ndim not in (1, 2):
        raise ValueError("rep_cols_by_shell must be a 1-D or 2-D array.")
    if squeeze:
        rep_cols = rep_cols[None, :]

    _, S = rep_cols.shape
    if S < 2:
        shape_pairs = (rep_cols.shape[0], 0, 2)
        dag_pairs = -jnp.ones(shape_pairs, dtype=jnp.int32)
        mask = jnp.zeros(shape_pairs[:-1], dtype=jnp.bool_)
        counts = jnp.zeros((rep_cols.shape[0],), dtype=jnp.int32)
    else:
        idx_i, idx_j = jnp.triu_indices(S, k=1)
        idx_i = idx_i.astype(jnp.int32)
        idx_j = idx_j.astype(jnp.int32)
        left = rep_cols[:, idx_i]
        right = rep_cols[:, idx_j]
        valid = (left >= 0) & (right >= 0)
        lo = jnp.minimum(left, right)
        hi = jnp.maximum(left, right)
        valid = valid & (hi > lo)
        pairs = jnp.stack([lo, hi], axis=-1)
        dag_pairs = jnp.where(valid[..., None], pairs, -1)
        mask = valid
        counts = mask.sum(axis=1, dtype=jnp.int32)

    if squeeze:
        dag_pairs = dag_pairs[0]
        mask = mask[0]
        counts = counts[0]
    return dag_pairs, mask, counts


def make_lexico_dag(sorted_indices: Array, k: int) -> Tuple[Array, Array, Array]:
    """
    Build pairs from the first k neighbors per row by overlapping consecutive grouping:
    (i1, i2), (i2, i3), (i3, i4), ... (i_{k-1}, i_k).
    Returns (pairs, mask, counts).
    """
    idx = jnp.asarray(sorted_indices, dtype=jnp.int32)
    squeeze = False
    if idx.ndim == 1:
        idx = idx[None, :]
        squeeze = True
    n, m = idx.shape
    k_eff = min(k, m)
    num_pairs = max(k_eff - 1, 0)
    if num_pairs == 0:
        shape_pairs = (idx.shape[0], 0, 2)
        dag_pairs = -jnp.ones(shape_pairs, dtype=jnp.int32)
        mask = jnp.zeros(shape_pairs[:-1], dtype=jnp.bool_)
        counts = jnp.zeros((idx.shape[0],), dtype=jnp.int32)
    else:
        neighbors = idx[:, :k_eff]
        left = neighbors[:, :-1]
        right = neighbors[:, 1:]
        pairs = jnp.stack([left, right], axis=-1)  # (n, num_pairs, 2)
        valid = (left >= 0) & (right >= 0)
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
