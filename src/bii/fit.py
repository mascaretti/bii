"""
Unified fit_bii pipeline: triplet formation → NUTS sampling → posterior summary.
"""

import time

import blackjax
import jax
from jax import numpy as jnp
from jax import random

from bii.data import T_from_X
from bii.horseshoe import horseshoe_dim, horseshoe_to_simplex, log_horseshoe_posterior
from bii.inference import loglik_w, loglik_w_per_triplet
from bii.sparse_dirichlet import (
    log_sparse_dirichlet_posterior,
    sparse_dirichlet_dim,
    sparse_dirichlet_to_simplex,
)
from bii.vi import run_vi, sample_vi

# ---------------------------------------------------------------------------
# Triplet formation
# ---------------------------------------------------------------------------


def _random_triplets(key, X_pool, Z_pool, n_triplets, anchor_fraction=0.1):
    """Partition pool into anchors/destinations; draw n_triplets pairs per anchor.

    Steps:
      1. Split N points into anchors (anchor_fraction) and destinations.
      2. For each anchor, randomly sample n_triplets pairs of destinations.
      3. Labels T are computed from clean distances in X_pool.

    Total triplets returned = n_anchors * n_triplets.

    Convention: [:, 0] = anchor, [:, 1] = closer dest, [:, 2] = farther dest.
    """
    N = X_pool.shape[0]
    n_anchors = max(1, int(N * anchor_fraction))

    key, k_split, k_pairs = random.split(key, 3)

    # Partition: first n_anchors of a random permutation are anchors
    perm = random.permutation(k_split, N)
    anchor_idx = perm[:n_anchors]
    dest_idx = perm[n_anchors:]
    n_dest = dest_idx.shape[0]

    # For each anchor, draw n_triplets random pairs from destinations
    # Total keys needed: n_anchors * n_triplets
    total = n_anchors * n_triplets
    pair_keys = random.split(k_pairs, total)

    def _sample_pair(key):
        return random.choice(key, n_dest, shape=(2,), replace=False)

    pair_positions = jax.vmap(_sample_pair)(pair_keys)  # (total, 2)

    # Build index array: (total, 3) = [anchor, dest1, dest2]
    anchor_repeated = jnp.repeat(anchor_idx, n_triplets)  # (total,)
    indices = jnp.stack(
        [
            anchor_repeated,
            dest_idx[pair_positions[:, 0]],
            dest_idx[pair_positions[:, 1]],
        ],
        axis=1,
    )  # (total, 3)

    X = X_pool[indices]  # (total, 3, p)
    Z = Z_pool[indices]  # (total, 3, p)
    T = T_from_X(X)
    return T, Z, indices


# ---------------------------------------------------------------------------
# WAIC
# ---------------------------------------------------------------------------


def _compute_waic(w_samples_flat, T, Z, sig):
    """Compute WAIC from posterior w samples.

    Args:
        w_samples_flat: (S, p) posterior draws on the simplex.
        T: (n,) binary labels.
        Z: (n, 3, p) embeddings.
        sig: noise std.

    Returns:
        WAIC scalar.
    """
    # Per-triplet log-lik for each posterior draw: (S, n)
    per_triplet_ll = jax.vmap(lambda w: loglik_w_per_triplet(w, T, Z, sig))(w_samples_flat)

    # lppd = sum_t log mean_s p(T_t | w^(s))
    # Use log-sum-exp for numerical stability
    S = per_triplet_ll.shape[0]
    lppd = jnp.sum(jax.scipy.special.logsumexp(per_triplet_ll, axis=0) - jnp.log(S))

    # p_waic = sum_t var_s(log p(T_t | w^(s)))
    p_waic = jnp.sum(jnp.var(per_triplet_ll, axis=0))

    return -2.0 * (lppd - p_waic)


# ---------------------------------------------------------------------------
# NUTS runner (shared by both priors)
# ---------------------------------------------------------------------------


def _run_nuts(
    key,
    logprob_fn,
    init_position,
    num_samples,
    num_warmup,
    num_chains,
    step_size,
    target_acceptance_rate,
):
    """Run multi-chain NUTS via BlackJAX.

    Returns:
        raw_samples: (num_samples, num_chains, dim)
        acceptance_rates: (num_samples, num_chains)
    """
    dim = init_position.shape[0]

    # Per-chain init with small perturbations
    key, *init_keys = random.split(key, num_chains + 1)
    init_positions = jnp.stack([
        init_position + 0.1 * random.normal(k, shape=(dim,)) for k in init_keys
    ])

    warmup = blackjax.window_adaptation(
        blackjax.nuts,
        logprob_fn,
        num_warmup,
        target_acceptance_rate=target_acceptance_rate,
        initial_step_size=step_size,
    )

    keys_warmup = random.split(key, num_chains)

    all_samples = []
    all_acceptance = []

    def _make_step_fn(kernel):
        def one_step(state, key):
            new_state, info = kernel.step(key, state)
            return new_state, (new_state.position, info.acceptance_rate)

        return one_step

    for chain_idx in range(num_chains):
        (state, params), _ = warmup.run(keys_warmup[chain_idx], init_positions[chain_idx])
        kernel = blackjax.nuts(logprob_fn, **params)
        one_step = _make_step_fn(kernel)

        key, subkey = random.split(key)
        sample_keys = random.split(subkey, num_samples)
        _, (samples_chain, acc_chain) = jax.lax.scan(one_step, state, sample_keys)

        all_samples.append(samples_chain)
        all_acceptance.append(acc_chain)

    raw_samples = jnp.stack(all_samples, axis=1)
    acceptance_rates = jnp.stack(all_acceptance, axis=1)
    return raw_samples, acceptance_rates


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def _rhat(samples):
    """Gelman-Rubin R-hat.  samples: (num_samples, num_chains, p)."""
    n, m, _ = samples.shape
    chain_means = jnp.mean(samples, axis=0)
    global_mean = jnp.mean(chain_means, axis=0)
    B = n / (m - 1) * jnp.sum((chain_means - global_mean[None, :]) ** 2, axis=0)
    W = jnp.mean(jnp.var(samples, axis=0, ddof=1), axis=0)
    var_plus = ((n - 1) / n) * W + (1 / n) * B
    return jnp.sqrt(var_plus / (W + 1e-10))


def _ess(samples):
    """Simple ESS via lag-1 autocorrelation.  samples: (num_samples, num_chains, p)."""
    n, m, p = samples.shape
    total = n * m
    flat = samples.reshape(-1, p)
    acf1 = jnp.array([jnp.corrcoef(flat[:-1, i], flat[1:, i])[0, 1] for i in range(p)])
    return total / (1 + 2 * jnp.maximum(acf1, 0))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def fit_bii(
    key,
    X_pool,
    Z_pool,
    sig,
    prior="dirichlet",
    n_triplets=500,
    anchor_fraction=0.5,
    triplet_strategy="random",
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
        X_pool: (N, p) clean embeddings.
        Z_pool: (N, p) noisy/normalised embeddings.
        sig: noise std — scalar or (p,).
        prior: ``"dirichlet"``, ``"horseshoe"``, or ``"sparse_dirichlet"``.
        n_triplets: destination pairs per anchor. Total triplets =
            ``int(N * anchor_fraction) * n_triplets``.
        anchor_fraction: fraction of pool used as anchors (default 0.1).
        triplet_strategy: ``"random"`` (only option for now).
        alpha: Dirichlet concentration; default ``ones(p)``.
        kappa: power-likelihood correction.
        inference_method: ``"nuts"`` or ``"vi"`` (mean-field variational).
        num_samples: posterior draws per chain (NUTS).
        num_warmup: NUTS warmup steps.
        num_chains: number of MCMC chains.
        step_size: initial NUTS step size.
        target_acceptance_rate: NUTS target acceptance.
        vi_steps: number of VI optimization steps.
        vi_lr: Adam learning rate for VI.
        vi_elbo_samples: MC samples per ELBO estimate.
        vi_num_samples: samples drawn from fitted variational posterior.

    Returns:
        dict with keys ``w_samples``, ``raw_samples``, ``T``,
        ``triplet_indices``, ``prior``, ``kappa``, ``waic``,
        ``elapsed_seconds``, ``diagnostics``.
    """
    t0 = time.perf_counter()
    p = Z_pool.shape[1]

    if alpha is None:
        alpha = jnp.ones(p)

    # ------------------------------------------------------------------
    # Step 1 — form triplets
    # ------------------------------------------------------------------
    key, key_trip = random.split(key)
    T, Z, indices = _random_triplets(key_trip, X_pool, Z_pool, n_triplets, anchor_fraction)

    # ------------------------------------------------------------------
    # Step 2 — build log-posterior & init
    # ------------------------------------------------------------------
    if prior == "dirichlet":
        _alpha = alpha
        _kappa = kappa

        def logprob_fn(theta):
            w = jax.nn.softmax(theta)
            return _kappa * loglik_w(w, T, Z, sig) + jnp.sum((_alpha - 1.0) * jnp.log(w + 1e-12))

        init_position = jnp.zeros(p)

    elif prior == "sparse_dirichlet":

        def logprob_fn(position):
            return log_sparse_dirichlet_posterior(position, T, Z, sig, kappa)

        dim = sparse_dirichlet_dim(p)
        init_position = jnp.zeros(dim)

    elif prior == "horseshoe":

        def logprob_fn(position):
            return log_horseshoe_posterior(position, T, Z, sig, kappa)

        dim = horseshoe_dim(p)
        init_position = jnp.zeros(dim)

    else:
        raise ValueError(f"Unknown prior: {prior!r}")

    # ------------------------------------------------------------------
    # Step 3 — run inference
    # ------------------------------------------------------------------
    if inference_method == "nuts":
        key, key_nuts = random.split(key)
        raw_samples, acceptance_rates = _run_nuts(
            key_nuts,
            logprob_fn,
            init_position,
            num_samples,
            num_warmup,
            num_chains,
            step_size,
            target_acceptance_rate,
        )

        # ------------------------------------------------------------------
        # Step 4 — extract w, diagnostics, WAIC
        # ------------------------------------------------------------------
        if prior == "dirichlet":
            w_samples = jax.vmap(jax.vmap(jax.nn.softmax))(raw_samples)
        elif prior == "sparse_dirichlet":
            w_samples = jax.vmap(jax.vmap(sparse_dirichlet_to_simplex))(raw_samples)
        else:
            w_samples = jax.vmap(jax.vmap(horseshoe_to_simplex))(raw_samples)

        # Diagnostics
        mean_acc = jnp.mean(acceptance_rates)
        acc_per_chain = jnp.mean(acceptance_rates, axis=0)

        diagnostics = {
            "acceptance_rate": mean_acc,
            "acceptance_rate_per_chain": acc_per_chain,
        }

        if num_chains > 1:
            diagnostics["rhat"] = _rhat(w_samples)
            diagnostics["ess"] = _ess(w_samples)

    elif inference_method == "vi":
        if prior not in ("dirichlet", "sparse_dirichlet"):
            raise ValueError("VI is only supported with 'dirichlet' or 'sparse_dirichlet' priors.")

        key, key_vi, key_sample = random.split(key, 3)

        # sparse_dirichlet has position dim = 2p+2, not p
        vi_dim = sparse_dirichlet_dim(p) if prior == "sparse_dirichlet" else p

        mu, log_sigma, elbo_history = run_vi(
            key_vi,
            logprob_fn,
            vi_dim,
            num_steps=vi_steps,
            lr=vi_lr,
            num_elbo_samples=vi_elbo_samples,
        )
        theta_samples, w_flat_vi = sample_vi(key_sample, mu, log_sigma, vi_num_samples)

        # For sparse_dirichlet, extract w via sparse_dirichlet_to_simplex
        if prior == "sparse_dirichlet":
            w_flat_vi = jax.vmap(sparse_dirichlet_to_simplex)(theta_samples)

        # Reshape to (S, 1, p) to match NUTS convention
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
    waic = _compute_waic(w_flat, T, Z, sig)

    elapsed = time.perf_counter() - t0

    return {
        "w_samples": w_samples,
        "raw_samples": raw_samples,
        "T": T,
        "triplet_indices": indices,
        "prior": prior,
        "kappa": kappa,
        "waic": waic,
        "elapsed_seconds": elapsed,
        "diagnostics": diagnostics,
    }
