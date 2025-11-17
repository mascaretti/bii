# radial_binning.py
# -*- coding: utf-8 -*-
"""
Minimal JAX utilities for:
  - pairwise distances (sorted, with indices)
  - PPP-motivated shells with equal expected counts
  - assigning neighbors to shells (labels aligned to original columns)
  - sampling one representative per shell uniformly at random (aligned)
  - reproducible two-way partition of indices (same split for X and Z)

This version fixes the plotting/colour bug by returning shell labels
**aligned with original Y_cols indices**.
"""

from typing import Tuple, Optional
import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.special import gammaln

Array = jnp.ndarray


# ------------------------------ Distance (sorted) ------------------------------

@jax.jit
def make_distance(Y_rows: Array, Y_cols: Array) -> Tuple[Array, Array]:
    """
    Pairwise distances from each row in Y_rows (n,q) to each row in Y_cols (m,q),
    returned in ascending-distance order alongside the original column indices.

    Returns:
      DY_sorted:  (n, m, 2)
          [..., 0] = distance (ascending)
          [..., 1] = Y_cols column index (int32) in that sorted order
      cols_sorted: (n, m) int32, the same column indices as DY_sorted[..., 1]
    """
    diffs = Y_rows[:, None, :] - Y_cols[None, :, :]     # (n, m, q)
    dists = jnp.linalg.norm(diffs, axis=-1)             # (n, m)

    n, m = dists.shape
    col_idx = jnp.broadcast_to(jnp.arange(m)[None, :], (n, m))

    # Sort by (distance, column) to be deterministic on ties
    sort_idx   = jnp.lexsort(keys=(col_idx, dists))     # (n, m)
    d_sorted   = jnp.take_along_axis(dists, sort_idx, axis=1)
    cols_sorted = jnp.take_along_axis(col_idx, sort_idx, axis=1).astype(jnp.int32)

    DY_sorted = jnp.stack([d_sorted, cols_sorted.astype(d_sorted.dtype)], axis=-1)
    DY_sorted = DY_sorted.at[..., 1].set(cols_sorted)   # keep indices as int32
    return DY_sorted, cols_sorted


# ------------------------------ PPP shells (equal expected counts) ------------------------------

def _unit_ball_volume(d: int) -> Array:
    """V_d = pi^(d/2) / Gamma(1 + d/2)."""
    d = jnp.array(d, dtype=jnp.float32)
    logV = (d / 2.0) * jnp.log(jnp.pi) - gammaln(1.0 + d / 2.0)
    return jnp.exp(logV)

def equal_expected_count_shells_from_Rmax(Rmax: float, num_shells: int, dim: int) -> Array:
    """
    Equal expected counts under homogeneous PPP via equal *volume* shells.
    In d dimensions:
      r_i = Rmax * ((i+1)/S)^(1/d)  for i=0..S-1.
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
      Δ = target / (lambda_hat * V_d),  r_i^d = r_{i-1}^d + Δ
      => r_i = ( r0^d + (i+1)*Δ )^(1/d)
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
    )  # in r^d units
    d = jnp.asarray(float(dim), dtype=dtype)
    base = (jnp.asarray(r0, dtype=dtype) ** d)
    i = jnp.arange(1, num_shells + 1, dtype=dtype)
    return (base + i * Delta) ** (1.0 / d)


# ------------------------------ Assign to shells (aligned to original columns) ------------------------------

def assign_to_shells_aligned(
    DY_sorted: Array,
    cols_sorted: Array,
    radii: Array,
) -> Tuple[Array, Array]:
    """
    Produce shell labels both:
      - in sorted order (useful internally), and
      - ALIGNED TO ORIGINAL COLUMN INDICES (for plotting/usage).

    Intervals are (0, r1], (r1, r2], ..., (r_{S-1}, r_S]; distances > r_S => -1.

    Returns:
      shell_sorted: (n, m) labels aligned with DY_sorted ascending order
      shell_by_col: (n, m) labels aligned to original Y_cols indices
    """
    d_sorted = DY_sorted[..., 0]                            # (n, m)
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

    # Remap to original column order: for each row, place sorted labels at cols_sorted positions
    def remap_row(shell_row, cols_row):
        out = -jnp.ones_like(shell_row, dtype=jnp.int32)
        out = out.at[cols_row].set(shell_row)
        return out

    shell_by_col = jax.vmap(remap_row)(shell_sorted, cols_sorted)  # (n, m)
    return shell_sorted, shell_by_col


# ------------------------------ Random representative per shell (aligned) ------------------------------

def sample_representatives_uniform_aligned(
    DY_sorted: Array,
    shell_sorted: Array,
    radii: Array,
    key: jax.Array,
) -> Tuple[Array, Array]:
    """
    For each row and each shell s, pick ONE representative uniformly at random
    among the members of that shell. Operates in the **sorted order** for efficiency,
    and returns representatives aligned to **original columns**.

    Args:
      DY_sorted   : (n, m, 2), distances ascending; DY_sorted[..., 1] are column indices
      shell_sorted: (n, m), shell labels aligned with DY_sorted
      radii       : (S,)
      key         : PRNGKey

    Returns:
      rep_cols_by_shell : (n, S) int32, sampled column index in Y_cols or -1 if empty
      rep_dists_by_shell: (n, S) float32, distance of sampled representative or NaN
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
            # Uniform sampling on True positions via Gumbel–Max (equal logits)
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
    """
    Deterministically pick, for each shell, the closest column (first occurrence in
    the sorted distance array). Useful when we need a fixed DAG per row.

    Args:
      DY_sorted   : (n, m, 2)
      shell_sorted: (n, m)
      num_shells  : number of shells (S) to extract.

    Returns:
      rep_cols_by_shell : (n, S) int32, -1 if shell empty
      rep_dists_by_shell: (n, S) float32, NaN if shell empty
    """
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
    """
    Deterministically pick the k-th nearest neighbour for the k-th annulus.

    Args:
      DY_sorted  : (n, m, 2) sorted distances/indices (output of make_distance).
      num_shells : number of annuli/shells; annulus k selects the k-th nearest neighbour.

    Returns:
      rep_cols_by_shell : (n, S) int32, -1 if not enough neighbours.
      rep_dists_by_shell: (n, S) float32, NaN if not enough neighbours.
    """
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


# ------------------------------ Shell DAG pairs ------------------------------

def make_shell_dag_pairs(rep_cols_by_shell: Array) -> Tuple[Array, Array, Array]:
    """
    Build the largest possible DAG (all admissible pairs) from shell representatives.

    Args:
      rep_cols_by_shell: (n, S) or (S,) int32 with one column index per shell.
                         Empty shells must be encoded as -1.

    Returns:
      dag_pairs : (n, K, 2) int32 with directed edges (i -> j) where j > i.
                  K = S * (S - 1) / 2 and invalid rows are filled with -1.
      pair_mask : (n, K) bool array marking which pairs in dag_pairs are valid.
      pair_count: (n,) int32 number of valid edges per row.
                  When the input was 1-D, the outputs are squeezed accordingly.
    """
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


# ------------------------------ Partition (same split for X and Z) ------------------------------

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
    # default ~50/50
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
    """
    Disjoint partition of {0..n-1} into rows_idx and cols_idx (same split for X and Z).
    """
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
    """
    Apply the same disjoint partition to X and Z.
    """
    if X.shape[0] != Z.shape[0]:
        raise ValueError("X and Z must have the same number of rows.")
    return X[rows_idx], X[cols_idx], Z[rows_idx], Z[cols_idx]
