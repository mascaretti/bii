"""Thin composition layer: triplets -> log posterior -> NUTS or MAP -> diagnostics."""

import time

import jax
from jax import numpy as jnp
from jax import random

from bii.data import make_triplets
from bii.diagnostics import aligned_triplets, compute_ess, compute_rhat, compute_waic
from bii.inference import slope_posterior, slope_scale
from bii.priors import make_dirichlet_logposterior
from bii.sampling import run_map, run_nuts


def fit_bii(
    key,
    X_pool,
    Z_pool,
    sig,
    n_triplets=15,
    anchor_fraction=0.5,
    triplet_sampler=None,
    alpha=None,
    kappa=1.0,
    slope_grid=None,
    inference_method="nuts",
    num_samples=1000,
    num_warmup=500,
    num_chains=4,
    step_size=1e-3,
    target_acceptance_rate=0.8,
    map_steps=3000,
    map_lr=1e-2,
    map_restarts=1,
    compute_waic_flag=True,
    progress_bar=False,
):
    """Posterior of the metric weights from paired pools of points.

    Args:
        key: JAX random key.
        X_pool: (N, p_x) target points.
        Z_pool: (N, p) source points.
        sig: noise standard deviation of the source: a scalar, per feature (p,), per
            point (N,), or per point and feature (N, p). Per-point values are resolved
            to the triplets through their indices. Full covariance matrices are not
            supported.
        n_triplets: candidate pairs per anchor, ``m``.
        anchor_fraction: fraction of the pool used as anchors.
        triplet_sampler: optional callable
            ``(key, X_pool, Z_pool, sig, n_triplets, anchor_fraction) -> (T, X, Z, indices)``;
            default :func:`bii.data.make_triplets`.
        alpha: (p,) Dirichlet concentration; default ``ones(p)``.
        kappa: temperature of the likelihood; ``1 / n_triplets`` in the paper.
        slope_grid: optional (R,) calibrated slopes, e.g. :func:`bii.make_slope_grid`;
            the calibration slope is then integrated out (see
            :func:`bii.priors.make_dirichlet_logposterior`). The result gains
            ``slope_posterior``, the posterior over the grid averaged over draws, and
            ``slope_mean``, the posterior mean of the slope.
        inference_method: ``"nuts"`` (the posterior) or ``"map"`` (its mode only;
            ``w_samples`` then has shape ``(1, 1, p)``).
        num_samples, num_warmup, num_chains, step_size, target_acceptance_rate: NUTS.
        map_steps, map_lr, map_restarts: Adam steps, learning rate and random restarts
            for the MAP.
        compute_waic_flag: compute WAIC, at the posterior-mean slope when there is a
            grid.
        progress_bar: NUTS only; print the progress about every 10% of the draws.

    Returns:
        dict with ``w_samples`` (num_samples, num_chains, p), ``raw_samples`` (the
        same in ``theta``), ``T``, ``Z``, ``triplet_indices``, ``n_triplets``,
        ``kappa``, ``slope_grid``, ``slope_posterior``, ``slope_mean``, ``waic``,
        ``n_aligned`` (aligned triplets per draw), ``elapsed_seconds`` and
        ``diagnostics`` (acceptance, R-hat and ESS for NUTS; the log-posterior
        trace for MAP).
    """
    t0 = time.perf_counter()
    p = Z_pool.shape[1]

    if alpha is None:
        alpha = jnp.ones(p)

    # Step 1 — form triplets
    key, key_trip = random.split(key)
    if triplet_sampler is None:
        T, X, Z, indices = make_triplets(key_trip, X_pool, Z_pool, n_triplets, anchor_fraction)
    else:
        T, X, Z, indices = triplet_sampler(key_trip, X_pool, Z_pool, sig, n_triplets,
                                           anchor_fraction)

    # Step 1b — resolve per-point sigmas to per-triplet if needed
    sig_arr = jnp.asarray(sig)
    N = X_pool.shape[0]
    if sig_arr.ndim == 1 and sig_arr.shape[0] == N:
        # Pool-level per-point sigmas (N,) -> triplet-level (n_triplets, 3)
        sig_resolved = sig_arr[indices]
    elif sig_arr.ndim == 2 and sig_arr.shape == (N, p):
        # Pool-level per-point diagonal sigmas (N, p) -> (n_triplets, 3, p)
        sig_resolved = sig_arr[indices]
    elif sig_arr.ndim == 0 or (sig_arr.ndim == 1 and sig_arr.shape[0] == p):
        sig_resolved = sig
    else:
        raise ValueError(
            f"sig of shape {sig_arr.shape} is not interpretable: expected a "
            f"scalar, per-feature (p,) = ({p},), per-point (N,) = ({N},), or "
            f"per-point diagonal (N, p) = ({N}, {p}). Full (non-diagonal) "
            f"covariance matrices are not supported."
        )

    # Step 2 — the log posterior; a slope grid is integrated out, so the
    # position is theta alone
    logprob_fn = make_dirichlet_logposterior(T, Z, sig_resolved, alpha, kappa,
                                             slope_grid=slope_grid)
    init_position = jnp.zeros(p)

    # Step 3 — run inference
    if inference_method == "nuts":
        key, key_nuts = random.split(key)
        raw_samples, acceptance_rates = run_nuts(
            key_nuts, logprob_fn, init_position,
            num_samples, num_warmup, num_chains, step_size, target_acceptance_rate,
            progress_bar=progress_bar,
        )
        w_samples = jax.vmap(jax.vmap(jax.nn.softmax))(raw_samples)

        diagnostics = {
            "acceptance_rate": jnp.mean(acceptance_rates),
            "acceptance_rate_per_chain": jnp.mean(acceptance_rates, axis=0),
        }
        if num_chains > 1:
            diagnostics["rhat"] = compute_rhat(w_samples)
            diagnostics["ess"] = compute_ess(w_samples)

    elif inference_method == "map":
        # The posterior mode, as a single "draw" so that the layout is unchanged
        key, key_map = random.split(key)
        map_position, logprob_history = run_map(
            key_map, logprob_fn, p,
            num_steps=map_steps, lr=map_lr, n_restarts=map_restarts,
        )
        raw_samples = map_position[None, None, :]              # (1, 1, p)
        w_samples = jax.nn.softmax(map_position[None, :])[:, None, :]  # (1, 1, p)

        diagnostics = {
            "logprob_history": logprob_history,
            "final_logprob": logprob_history[-1],
            "map_position": map_position,
        }

    else:
        raise ValueError(f"Unknown inference_method: {inference_method!r}")

    w_flat = w_samples.reshape(-1, p)

    # Posterior of the calibrated slope: per draw over the grid, averaged.
    # lax.map keeps memory at one (R, n) block per draw.
    slope_post = None
    slope_mean = None
    if slope_grid is not None:
        grid = jnp.asarray(slope_grid)
        sub = w_flat[:: max(1, w_flat.shape[0] // 512)]

        def _slope_one(w_):
            post_ = slope_posterior(w_, T, Z, sig_resolved, grid, kappa=kappa)
            scale_ = slope_scale(w_, Z, sig_resolved)
            return post_, jnp.sum(post_ * grid) / scale_

        posts, lams = jax.lax.map(_slope_one, sub)
        slope_post = jnp.mean(posts, axis=0)
        slope_mean = float(jnp.mean(lams))
    lam_eval = slope_mean if slope_mean is not None else 1.0

    # WAIC (optional: it can run out of memory with many triplets and draws)
    waic = (compute_waic(w_flat, T, Z, sig_resolved, lam=lam_eval)
            if compute_waic_flag else None)

    n_aligned = aligned_triplets(w_flat, T, Z, sig_resolved)

    return {
        "w_samples": w_samples,
        "raw_samples": raw_samples,
        "T": T,
        "Z": Z,
        "triplet_indices": indices,
        "n_triplets": int(T.shape[0]),
        "kappa": kappa,
        "slope_grid": slope_grid,
        "slope_posterior": slope_post,
        "slope_mean": slope_mean,
        "waic": waic,
        "n_aligned": n_aligned,
        "elapsed_seconds": time.perf_counter() - t0,
        "diagnostics": diagnostics,
    }
