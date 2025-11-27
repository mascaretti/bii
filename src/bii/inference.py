# -*- coding: utf-8 -*-
"""
Inference utilities for lexicographic DAG pair comparisons (k-NN based).
"""
from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm
from jax.scipy import special as jsp_special
from numpyro.diagnostics import effective_sample_size
from .radial import make_distance, make_lexico_dag

Array = jnp.ndarray


def remap_distances_to_columns(DY_sorted: Array, cols_sorted: Array) -> Array:
    d_sorted = DY_sorted[..., 0]
    def remap_row(dist_row, col_row):
        out = jnp.empty_like(dist_row)
        return out.at[col_row].set(dist_row)
    return jax.vmap(remap_row)(d_sorted, cols_sorted)


def compute_T_from_dag(distances_by_col: Array, dag_pairs: Array, pair_mask: Array) -> Array:
    dist = jnp.asarray(distances_by_col)
    pairs = jnp.asarray(dag_pairs, dtype=jnp.int32)
    mask = jnp.asarray(pair_mask, dtype=jnp.bool_)
    squeeze = False
    if pairs.ndim == 2:
        pairs = pairs[None, ...]; mask = mask[None, ...]; squeeze = True
    n_rows, n_cols = dist.shape[0], dist.shape[1]
    def row_fn(dist_row, pair_row, mask_row):
        idx_i = jnp.clip(pair_row[:, 0], 0, n_cols - 1)
        idx_j = jnp.clip(pair_row[:, 1], 0, n_cols - 1)
        di = dist_row[idx_i]; dj = dist_row[idx_j]
        indicator = di <= dj
        return jnp.where(mask_row, indicator, False)
    indicators = jax.vmap(row_fn)(dist, pairs, mask).astype(jnp.int32)
    return indicators[0] if squeeze else indicators


def _noise_variance(weights: Array, sigma: Array) -> Array:
    w2 = jnp.square(weights)
    sigma4 = jnp.square(jnp.square(sigma))
    return jnp.asarray(4.0) * jnp.sum(w2 * sigma4)


def probit_pairwise_probabilities(Z_rows, Z_cols, dag_pairs, pair_mask, weights, sigma):
    Z_rows = jnp.asarray(Z_rows); Z_cols = jnp.asarray(Z_cols)
    pairs = jnp.asarray(dag_pairs, dtype=jnp.int32)
    mask = jnp.asarray(pair_mask, dtype=jnp.bool_)
    weights = jnp.asarray(weights); sigma = jnp.asarray(sigma)
    squeeze = False
    if pairs.ndim == 2:
        pairs = pairs[None, ...]; mask = mask[None, ...]; squeeze = True
    if Z_rows.ndim == 1:
        Z_rows = Z_rows[None, ...]
    n_cols = Z_cols.shape[0]
    noise_var = _noise_variance(weights, sigma)
    noise_std = jnp.sqrt(noise_var)
    inv_std = jnp.where(noise_std > 0, 1.0 / noise_std, 0.0)
    def row_fn(z0, pair_row, mask_row):
        idx_i = jnp.clip(pair_row[:, 0], 0, n_cols - 1)
        idx_j = jnp.clip(pair_row[:, 1], 0, n_cols - 1)
        V = jnp.square(Z_cols[idx_i] - z0) - jnp.square(Z_cols[idx_j] - z0)
        margin = jnp.sum(V * weights, axis=-1)
        prob_noise = norm.cdf(-margin * inv_std)
        prob_det = (margin <= 0).astype(prob_noise.dtype)
        prob = jnp.where(noise_std > 0, prob_noise, prob_det)
        return jnp.where(mask_row, prob, 0.0)
    probs = jax.vmap(row_fn)(Z_rows, pairs, mask)
    return probs[0] if squeeze else probs


@dataclass
class PairwiseComparisonData:
    dag_pairs: Array
    pair_mask: Array
    targets: Array
    margin_features: Array


def _compute_margin_features(Z_rows: Array, Z_cols: Array, dag_pairs: Array) -> Array:
    Z_rows = jnp.asarray(Z_rows); Z_cols = jnp.asarray(Z_cols)
    pairs = jnp.asarray(dag_pairs, dtype=jnp.int32)
    squeeze = False
    if pairs.ndim == 2:
        pairs = pairs[None, ...]; squeeze = True
    if Z_rows.ndim == 1:
        Z_rows = Z_rows[None, ...]
    n_cols = Z_cols.shape[0]
    def row_fn(z0, pair_row):
        idx_i = jnp.clip(pair_row[:, 0], 0, n_cols - 1)
        idx_j = jnp.clip(pair_row[:, 1], 0, n_cols - 1)
        return jnp.square(Z_cols[idx_i] - z0) - jnp.square(Z_cols[idx_j] - z0)
    feats = jax.vmap(row_fn)(Z_rows, pairs)
    return feats[0] if squeeze else feats


def build_pairwise_data(
    X_rows: Array,
    X_cols: Array,
    Z_rows: Array,
    Z_cols: Array,
    *,
    k_neighbors: int,
) -> PairwiseComparisonData:
    DX_sorted, X_idx = make_distance(X_rows, X_cols)
    DZ_sorted, Z_idx = make_distance(Z_rows, Z_cols)
    dag_pairs, dag_mask, _ = make_lexico_dag(Z_idx, k_neighbors)
    dist_by_col = remap_distances_to_columns(DX_sorted, X_idx)
    targets = compute_T_from_dag(dist_by_col, dag_pairs, dag_mask)
    margin_features = _compute_margin_features(Z_rows, Z_cols, dag_pairs)
    return PairwiseComparisonData(dag_pairs=dag_pairs, pair_mask=dag_mask, targets=targets, margin_features=margin_features)


def make_loglikelihood(pairwise_data: PairwiseComparisonData, sigma: Array) -> Callable[[Array], Array]:
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


def summarize_posterior_metrics(
    samples: Array,
    w_true: Array,
    *,
    alpha: float = 0.05,
    num_warmup: Optional[int] = None,
    num_samples: Optional[int] = None,
    total_time: Optional[float] = None,
) -> dict:
    """Compute RMSE, MCIW, ECP, ESS fraction, and time to 1000 effective samples."""
    samples = jnp.asarray(samples)
    w_true = jnp.asarray(w_true)
    mean_w = samples.mean(axis=0)
    rmse = float(jnp.sqrt(jnp.mean((mean_w - w_true) ** 2)))

    low, high = np.percentile(np.array(samples), [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0)
    mciw = float(jnp.mean(high - low))
    ecp = float(((w_true >= low) & (w_true <= high)).mean())

    arr = np.array(samples)
    if arr.ndim == 1:  # (num_samples,) for scalar param
        arr = arr[None, :, None]
    elif arr.ndim == 2:  # (num_samples, dim) -> (1, num_samples, dim)
        arr = arr[None, ...]
    num_chains = arr.shape[0]
    num_draws = arr.shape[1]
    total_draws = max(num_chains * num_draws, 1)

    ess = effective_sample_size(arr)
    ess_safe = jnp.maximum(jnp.asarray(ess, dtype=jnp.float32), 1e-6)
    ess_mean = jnp.nan_to_num(jnp.mean(ess_safe), nan=0.0)
    ess_frac = float(ess_mean / total_draws)

    time_1000_ess = None
    if total_time is not None and num_warmup is not None and num_samples is not None and num_samples > 0:
        tburn = total_time * (num_warmup / float(num_warmup + num_samples))
        tsave = total_time * (num_samples / float(num_warmup + num_samples))
        per_dim = tburn + tsave * (1000.0 / ess_safe)
        time_1000_ess = float(jnp.nan_to_num(jnp.mean(per_dim), nan=0.0))

    return {
        "rmse": rmse,
        "mciw": mciw,
        "ecp": ecp,
        "ess_frac": ess_frac,
        "time_1000_ess": time_1000_ess,
    }


__all__ = [
    "compute_T_from_dag",
    "probit_pairwise_probabilities",
    "remap_distances_to_columns",
    "PairwiseComparisonData",
    "build_pairwise_data",
    "make_loglikelihood",
    "summarize_posterior_metrics",
]
