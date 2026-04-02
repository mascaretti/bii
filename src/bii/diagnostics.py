"""Posterior diagnostics: WAIC, R-hat, ESS, alignment — all pure functions."""

import jax
from jax import numpy as jnp

from bii.inference import delta_V_one_triplet, loglik_w_per_triplet

def compute_waic(w_samples_flat, T, Z, sig):
    """Compute WAIC using lax.map to prevent OOM."""
    # Sequential map over samples instead of vmap
    per_triplet_ll = jax.lax.map(
        lambda w: loglik_w_per_triplet(w, T, Z, sig),
        w_samples_flat
    )

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


def weight_entropy(w_samples):
    """Normalized entropy of weight vectors on the simplex.

    Measures concentration of the weight vector:
    - 0 = uniform (no alignment / all nutrients equally important)
    - 1 = point mass on one nutrient (perfect alignment)

    Args:
        w_samples: (S, p) posterior draws on the simplex.

    Returns:
        (S,) normalized alignment scores in [0, 1].
    """
    p = w_samples.shape[1]
    # Clip to avoid log(0)
    w_safe = jnp.clip(w_samples, 1e-30, None)
    H = -jnp.sum(w_safe * jnp.log(w_safe), axis=1)
    return 1.0 - H / jnp.log(p)


def _sig_to_sig2(sig):
    """Convert sig to sig2: square if vector/scalar, pass through if matrix."""
    sig = jnp.asarray(sig)
    if sig.ndim <= 1:
        return jnp.square(sig)
    return sig


def triplet_accuracy(w_samples, T, Z, sig):
    """Triplet prediction accuracy for each posterior sample.

    For each weight vector w, computes the fraction of triplets where
    the model's predicted label matches the observed label T.
    This is the BII analogue of the Information Imbalance scalar:
    - ~0.5 = random (no alignment between X and weighted Z)
    - ~1.0 = perfect alignment

    Args:
        w_samples: (S, p) posterior draws on the simplex.
        T: (n,) binary triplet labels.
        Z: (n, 3, p) triplet embeddings.
        sig: noise std — scalar, (p,), or (p, p) covariance.

    Returns:
        (S,) accuracy values in [0, 1].
    """
    zi, zj, zk = Z[:, 1], Z[:, 2], Z[:, 0]
    beta = jnp.exp(sig**2) * (jnp.exp(sig**2) - 1)

    def accuracy_one(w):
        def dv(zi, zj, zk):
            return delta_V_one_triplet(zi, zj, zk, w,
                                       beta * zi**2,
                                       beta * zj**2,
                                       beta * zk**2)
        delta, _V = jax.vmap(dv)(zi, zj, zk)
        pred = (delta <= 0.0).astype(jnp.float32)
        return jnp.mean(pred == T)

    return jax.lax.map(accuracy_one, w_samples)


def alignment_index(w_samples, T, Z, sig):
    """Normalised cross-entropy alignment index.

    Maps the mean per-triplet log-likelihood to [0, 1]:
      Δ(w) = 1 + ℓ̄(w) / log(2)
    where ℓ̄ = (1/n) Σ_t log p(T_t | w).

    - Random guessing (P=0.5): ℓ̄ = -log(2)  →  Δ = 0
    - Perfect prediction (P→1): ℓ̄ → 0       →  Δ = 1

    Args:
        w_samples: (S, p) posterior draws on the simplex.
        T: (n,) binary triplet labels.
        Z: (n, 3, p) triplet embeddings.
        sig: noise std — scalar, (p,), or (p, p) covariance.

    Returns:
        (S,) alignment index values in [0, 1].
    """
    def delta_one(w):
        ll = loglik_w_per_triplet(w, T, Z, sig)  # (n,)
        mean_ll = jnp.mean(ll)
        return 1.0 + mean_ll / jnp.log(2.0)

    return jax.lax.map(delta_one, w_samples)
