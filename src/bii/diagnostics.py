"""Posterior diagnostics: WAIC, R-hat, ESS — all pure functions."""

import jax
from jax import numpy as jnp

from bii.inference import loglik_w_per_triplet


def compute_waic(w_samples_flat, T, Z, sig):
    """Compute WAIC from posterior w samples.

    Args:
        w_samples_flat: (S, p) posterior draws on the simplex.
        T: (n,) binary labels.
        Z: (n, 3, p) embeddings.
        sig: noise std.

    Returns:
        WAIC scalar (deviance scale: ``-2 * elpd``).
    """
    per_triplet_ll = jax.vmap(lambda w: loglik_w_per_triplet(w, T, Z, sig))(w_samples_flat)

    S = per_triplet_ll.shape[0]
    lppd = jnp.sum(jax.scipy.special.logsumexp(per_triplet_ll, axis=0) - jnp.log(S))
    p_waic = jnp.sum(jnp.var(per_triplet_ll, axis=0))

    return -2.0 * (lppd - p_waic)


def compute_rhat(samples):
    """Gelman-Rubin R-hat convergence diagnostic.

    Args:
        samples: (num_samples, num_chains, p).

    Returns:
        R-hat values for each parameter (p,).
    """
    n, m, _ = samples.shape
    chain_means = jnp.mean(samples, axis=0)
    global_mean = jnp.mean(chain_means, axis=0)
    B = n / (m - 1) * jnp.sum((chain_means - global_mean[None, :]) ** 2, axis=0)
    W = jnp.mean(jnp.var(samples, axis=0, ddof=1), axis=0)
    var_plus = ((n - 1) / n) * W + (1 / n) * B
    return jnp.sqrt(var_plus / (W + 1e-10))


def compute_ess(samples):
    """Effective sample size via lag-1 autocorrelation.

    Args:
        samples: (num_samples, num_chains, p).

    Returns:
        ESS for each parameter (p,).
    """
    n, m, p = samples.shape
    total = n * m
    flat = samples.reshape(-1, p)
    acf1 = jnp.array([jnp.corrcoef(flat[:-1, i], flat[1:, i])[0, 1] for i in range(p)])
    return total / (1 + 2 * jnp.maximum(acf1, 0))
