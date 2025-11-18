# -*- coding: utf-8 -*-
"""
Inference utilities for shell-based pair comparisons.

This module complements `bii.radial` by:
  - turning shell-induced DAG edges into binary TARGET variables T_{ij}
  - computing Probit-link probabilities P_{ij} in an auxiliary space Z
  - packaging all ingredients into a JAX-friendly Bernoulli log-likelihood
"""

from dataclasses import dataclass
from typing import Tuple, Callable

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm
from jax.scipy import special as jsp_special

from .radial import (
    make_distance,
    equal_expected_count_shells_from_Rmax,
    assign_to_shells_aligned,
    select_representatives_first_in_shell,
    make_shell_dag_pairs,
)

Array = jnp.ndarray


def remap_distances_to_columns(DY_sorted: Array, cols_sorted: Array) -> Array:
    """
    Convert sorted distances (output of `make_distance`) into matrices aligned with
    the original column indices.

    Args:
      DY_sorted  : (n_rows, n_cols, 2)
      cols_sorted: (n_rows, n_cols)

    Returns:
      distances_by_col: (n_rows, n_cols) with entry [r, c] equal to the distance
                        between row r and column c.
    """
    d_sorted = DY_sorted[..., 0]

    def remap_row(dist_row, col_row):
        out = jnp.empty_like(dist_row)
        return out.at[col_row].set(dist_row)

    return jax.vmap(remap_row)(d_sorted, cols_sorted)


def compute_T_from_dag(
    distances_by_col: Array,
    dag_pairs: Array,
    pair_mask: Array,
) -> Array:
    """
    Build TARGET indicators T_{ij} for each row/edge: T=1 iff point i is closer to the
    row anchor than point j.

    Args:
      distances_by_col: (n_rows, n_cols) matrix of row-to-column distances aligned with
                        the original column ordering (see `remap_distances_to_columns`).
      dag_pairs       : (n_rows, K, 2) int32 array with column indices per edge.
      pair_mask       : (n_rows, K) bool mask selecting valid edges.

    Returns:
      T: (n_rows, K) int32 array of {0,1} indicators.
    """
    dist = jnp.asarray(distances_by_col)
    pairs = jnp.asarray(dag_pairs, dtype=jnp.int32)
    mask = jnp.asarray(pair_mask, dtype=jnp.bool_)

    squeeze = False
    if pairs.ndim == 2:
        pairs = pairs[None, ...]
        mask = mask[None, ...]
        squeeze = True

    n_rows, n_cols = dist.shape[0], dist.shape[1]

    def row_fn(dist_row, pair_row, mask_row):
        idx_i = jnp.clip(pair_row[:, 0], 0, n_cols - 1)
        idx_j = jnp.clip(pair_row[:, 1], 0, n_cols - 1)
        di = dist_row[idx_i]
        dj = dist_row[idx_j]
        indicator = di <= dj
        return jnp.where(mask_row, indicator, False)

    indicators = jax.vmap(row_fn)(dist, pairs, mask).astype(jnp.int32)
    if squeeze:
        indicators = indicators[0]
    return indicators


def _noise_variance(weights: Array, sigma: Array) -> Array:
    """Variance of Σ w_i (ε_i^2 - ε_i'^2) with ε_i ~ N(0, σ_i^2)."""
    w2 = jnp.square(weights)
    sigma4 = jnp.square(jnp.square(sigma))
    return jnp.asarray(4.0) * jnp.sum(w2 * sigma4)


def probit_pairwise_probabilities(
    Z_rows: Array,
    Z_cols: Array,
    dag_pairs: Array,
    pair_mask: Array,
    weights: Array,
    sigma: Array,
) -> Array:
    """
    Compute P_{ij} = Φ( -Δ / σ_noise ) using the weighted squared-distance margin:
      Δ = d(z, z0; w) - d(z', z0; w) ≈ ⟨w, V⟩,  V_i = (z_i - z0_i)^2 - (z'_i - z0_i)^2

    Args:
      Z_rows  : (n_rows, d) anchor points.
      Z_cols  : (n_cols, d) candidate points aligned with column indices.
      dag_pairs: (n_rows, K, 2) column indices per edge (shared with TARGET edges).
      pair_mask: (n_rows, K) bool mask selecting valid edges.
      weights : (d,) non-negative weights w_i.
      sigma   : (d,) standard deviations for the diagonal noise Σ.

    Returns:
      probs: (n_rows, K) array with Φ(-margin / σ_noise) on valid edges, zero elsewhere.
    """
    Z_rows = jnp.asarray(Z_rows)
    Z_cols = jnp.asarray(Z_cols)
    pairs = jnp.asarray(dag_pairs, dtype=jnp.int32)
    mask = jnp.asarray(pair_mask, dtype=jnp.bool_)
    weights = jnp.asarray(weights)
    sigma = jnp.asarray(sigma)

    squeeze = False
    if pairs.ndim == 2:
        pairs = pairs[None, ...]
        mask = mask[None, ...]
        squeeze = True
    if Z_rows.ndim == 1:
        Z_rows = Z_rows[None, ...]

    n_rows = Z_rows.shape[0]
    n_cols = Z_cols.shape[0]

    noise_var = _noise_variance(weights, sigma)
    noise_std = jnp.sqrt(noise_var)
    inv_std = jnp.where(noise_std > 0, 1.0 / noise_std, 0.0)

    def row_fn(z0, pair_row, mask_row):
        idx_i = jnp.clip(pair_row[:, 0], 0, n_cols - 1)
        idx_j = jnp.clip(pair_row[:, 1], 0, n_cols - 1)
        zi = Z_cols[idx_i]
        zj = Z_cols[idx_j]
        diff_i = zi - z0
        diff_j = zj - z0
        V = jnp.square(diff_i) - jnp.square(diff_j)
        margin = jnp.sum(V * weights, axis=-1)
        prob_noise = norm.cdf(-margin * inv_std)
        prob_det = (margin <= 0).astype(prob_noise.dtype)
        prob = jnp.where(noise_std > 0, prob_noise, prob_det)
        return jnp.where(mask_row, prob, 0.0)

    probs = jax.vmap(row_fn)(Z_rows, pairs, mask)
    if squeeze:
        probs = probs[0]
    return probs


@dataclass
class PairwiseComparisonData:
    dag_pairs: Array
    pair_mask: Array
    targets: Array
    margin_features: Array


def _compute_margin_features(
    Z_rows: Array,
    Z_cols: Array,
    dag_pairs: Array,
) -> Array:
    """
    Precompute V_{k,ij} = (z_i - z0)^2 - (z_j - z0)^2 for every row/pair.
    """
    Z_rows = jnp.asarray(Z_rows)
    Z_cols = jnp.asarray(Z_cols)
    pairs = jnp.asarray(dag_pairs, dtype=jnp.int32)

    squeeze = False
    if pairs.ndim == 2:
        pairs = pairs[None, ...]
        squeeze = True
    if Z_rows.ndim == 1:
        Z_rows = Z_rows[None, ...]

    n_cols = Z_cols.shape[0]

    def row_fn(z0, pair_row):
        idx_i = jnp.clip(pair_row[:, 0], 0, n_cols - 1)
        idx_j = jnp.clip(pair_row[:, 1], 0, n_cols - 1)
        diff_i = Z_cols[idx_i] - z0
        diff_j = Z_cols[idx_j] - z0
        return jnp.square(diff_i) - jnp.square(diff_j)

    features = jax.vmap(row_fn)(Z_rows, pairs)
    if squeeze:
        features = features[0]
    return features


def build_pairwise_data(
    X_rows: Array,
    X_cols: Array,
    Z_rows: Array,
    Z_cols: Array,
    *,
    num_shells: int,
    quantile_outer: float = 0.2,
) -> PairwiseComparisonData:
    """
    Construct DAG edges from shells in Z-space and align them with TARGET distances.

    Returns all tensors needed for the Bernoulli log-likelihood.
    """
    # Shells in Z
    DZ_sorted, Z_cols_sorted = make_distance(Z_rows, Z_cols)
    d_sorted = np.array(DZ_sorted[..., 0])
    Rmax_rows = np.quantile(d_sorted, quantile_outer, axis=1)
    radii_rows = jnp.stack(
        [
            equal_expected_count_shells_from_Rmax(
                float(max(r, 1e-6)),
                num_shells,
                dim=Z_rows.shape[-1],
            )
            for r in Rmax_rows
        ]
    )

    shell_rows = []
    for i in range(Z_rows.shape[0]):
        shell_sorted_row, _ = assign_to_shells_aligned(
            DZ_sorted[i : i + 1],
            Z_cols_sorted[i : i + 1],
            radii_rows[i],
        )
        shell_rows.append(shell_sorted_row[0])
    shell_sorted = jnp.stack(shell_rows)

    rep_cols, _ = select_representatives_first_in_shell(
        DZ_sorted,
        shell_sorted,
        num_shells,
    )
    dag_pairs, dag_mask, _ = make_shell_dag_pairs(rep_cols)

    # TARGET distances in X
    DX_sorted, X_cols_sorted = make_distance(X_rows, X_cols)
    dist_by_col = remap_distances_to_columns(DX_sorted, X_cols_sorted)
    targets = compute_T_from_dag(dist_by_col, dag_pairs, dag_mask)

    # Margin features in Z
    margin_features = _compute_margin_features(Z_rows, Z_cols, dag_pairs)

    return PairwiseComparisonData(
        dag_pairs=dag_pairs,
        pair_mask=dag_mask,
        targets=targets,
        margin_features=margin_features,
    )


def make_loglikelihood(pairwise_data: PairwiseComparisonData, sigma: Array) -> Callable[[Array], Array]:
    """
    Create a closure w -> loglikelihood(w | data) suitable for NumPyro.
    """
    sigma = jnp.asarray(sigma)

    def loglik(weights: Array) -> Array:
        weights = jnp.asarray(weights)
        V = pairwise_data.margin_features
        mask = pairwise_data.pair_mask
        targets = pairwise_data.targets.astype(weights.dtype)
        margins = jnp.einsum("rkd,d->rk", V, weights)

        noise_var = _noise_variance(weights, sigma)
        noise_std = jnp.sqrt(noise_var)
        inv_std = jnp.where(noise_std > 0, 1.0 / noise_std, 0.0)
        scaled = -margins * inv_std

        log_cdf = jsp_special.log_ndtr(scaled)
        log_ccdf = jsp_special.log_ndtr(-scaled)
        log_probs = jnp.where(targets == 1, log_cdf, log_ccdf)
        log_probs = jnp.where(mask, log_probs, 0.0)
        return jnp.sum(log_probs)

    return loglik


__all__ = [
    "compute_T_from_dag",
    "probit_pairwise_probabilities",
    "remap_distances_to_columns",
    "PairwiseComparisonData",
    "build_pairwise_data",
    "make_loglikelihood",
]
