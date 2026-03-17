"""Thin composition layer: triplets -> logposterior -> sampling -> diagnostics."""

import time

import jax
from jax import numpy as jnp
from jax import random

from bii.data import make_triplets
from bii.diagnostics import compute_ess, compute_rhat, compute_waic
from bii.priors import (
    make_dirichlet_logposterior,
    make_sparse_dirichlet_logposterior,
    sparse_dirichlet_dim,
    sparse_dirichlet_to_simplex,
)
from bii.sampling import run_nuts, run_vi, sample_vi


def fit_bii(
    key,
    X_pool,
    Z_pool,
    sig,
    prior="dirichlet",
    n_triplets=500,
    anchor_fraction=0.5,
    # Prior hyperparams
    alpha=None,
    kappa=1.0,
    # Inference method
    inference_method="nuts",
    # NUTS params
    num_samples=1000,
    num_warmup=500,
    num_chains=4,
    step_size=1e-3,
    target_acceptance_rate=0.8,
    # VI params
    vi_steps=5000,
    vi_lr=1e-2,
    vi_elbo_samples=8,
    vi_num_samples=2000,
):
    """Unified Bayesian inference pipeline for metric weights.

    Args:
        key: JAX random key.
        X_pool: (N, p_x) clean embeddings.
        Z_pool: (N, p_z) noisy/normalised embeddings.
        sig: noise std — scalar, (p,), or (p, p) covariance.
        prior: ``"dirichlet"`` or ``"sparse_dirichlet"``.
        n_triplets: destination pairs per anchor.
        anchor_fraction: fraction of pool used as anchors.
        alpha: Dirichlet concentration; default ``ones(p)``.
        kappa: power-likelihood correction.
        inference_method: ``"nuts"`` or ``"vi"``.
        num_samples: posterior draws per chain (NUTS).
        num_warmup: NUTS warmup steps.
        num_chains: number of MCMC chains.
        step_size: initial NUTS step size.
        target_acceptance_rate: NUTS target acceptance.
        vi_steps: VI optimization steps.
        vi_lr: Adam learning rate for VI.
        vi_elbo_samples: MC samples per ELBO estimate.
        vi_num_samples: samples drawn from fitted variational posterior.

    Returns:
        dict with keys ``w_samples``, ``raw_samples``, ``T``, ``Z``,
        ``triplet_indices``, ``prior``, ``kappa``, ``waic``,
        ``elapsed_seconds``, ``diagnostics``.
    """
    t0 = time.perf_counter()
    p = Z_pool.shape[1]

    if alpha is None:
        alpha = jnp.ones(p)

    # Step 1 — form triplets
    key, key_trip = random.split(key)
    T, X, Z, indices = make_triplets(key_trip, X_pool, Z_pool, n_triplets, anchor_fraction)

    # Step 2 — build log-posterior
    if prior == "dirichlet":
        logprob_fn = make_dirichlet_logposterior(T, Z, sig, alpha, kappa)
        init_position = jnp.zeros(p)
    elif prior == "sparse_dirichlet":
        logprob_fn = make_sparse_dirichlet_logposterior(T, Z, sig, kappa)
        dim = sparse_dirichlet_dim(p)
        init_position = jnp.zeros(dim)
    else:
        raise ValueError(f"Unknown prior: {prior!r}")

    # Step 3 — run inference
    if inference_method == "nuts":
        key, key_nuts = random.split(key)
        raw_samples, acceptance_rates = run_nuts(
            key_nuts, logprob_fn, init_position,
            num_samples, num_warmup, num_chains, step_size, target_acceptance_rate,
        )

        # Extract w on simplex
        if prior == "dirichlet":
            w_samples = jax.vmap(jax.vmap(jax.nn.softmax))(raw_samples)
        else:
            w_samples = jax.vmap(jax.vmap(sparse_dirichlet_to_simplex))(raw_samples)

        diagnostics = {
            "acceptance_rate": jnp.mean(acceptance_rates),
            "acceptance_rate_per_chain": jnp.mean(acceptance_rates, axis=0),
        }
        if num_chains > 1:
            diagnostics["rhat"] = compute_rhat(w_samples)
            diagnostics["ess"] = compute_ess(w_samples)

    elif inference_method == "vi":
        key, key_vi, key_sample = random.split(key, 3)
        vi_dim = sparse_dirichlet_dim(p) if prior == "sparse_dirichlet" else p

        mu, log_sigma, elbo_history = run_vi(
            key_vi, logprob_fn, vi_dim,
            num_steps=vi_steps, lr=vi_lr, num_elbo_samples=vi_elbo_samples,
        )
        theta_samples, w_flat_vi = sample_vi(key_sample, mu, log_sigma, vi_num_samples)

        if prior == "sparse_dirichlet":
            w_flat_vi = jax.vmap(sparse_dirichlet_to_simplex)(theta_samples)

        raw_samples = theta_samples[:, None, :]
        w_samples = w_flat_vi[:, None, :]

        diagnostics = {
            "elbo_history": elbo_history,
            "final_elbo": elbo_history[-1],
            "mu": mu,
            "log_sigma": log_sigma,
        }
    else:
        raise ValueError(f"Unknown inference_method: {inference_method!r}")

    # WAIC
    w_flat = w_samples.reshape(-1, p)
    waic = compute_waic(w_flat, T, Z, sig)

    elapsed = time.perf_counter() - t0

    return {
        "w_samples": w_samples,
        "raw_samples": raw_samples,
        "T": T,
        "Z": Z,
        "triplet_indices": indices,
        "prior": prior,
        "kappa": kappa,
        "waic": waic,
        "elapsed_seconds": elapsed,
        "diagnostics": diagnostics,
    }
