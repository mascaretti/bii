"""Likelihood functions for triplet comparisons — all pure functions."""

import jax
from jax import numpy as jnp
from jax.scipy.special import log_ndtr


@jax.jit
def delta_V_one_triplet(zi, zj, zk, w, sig2_i, sig2_j, sig2_k):
    a = zi - zk          # candidate 1 - anchor
    b = zj - zk          # candidate 2 - anchor

    # Mean: now includes bias correction
    delta = jnp.sum(w * (a*a - b*b + sig2_i - sig2_j))

    # Variance: observation-specific
    w2 = w * w
    V = jnp.sum(w2 * (
        4 * sig2_i * a*a
      + 4 * sig2_j * b*b
      + 4 * (sig2_i + sig2_j) * sig2_k
      + 4 * (a - b)**2 * sig2_k
      + 2 * (sig2_i**2 + sig2_j**2)
    ))

    return delta, V

@jax.jit
def logP_log1mP_from_deltaV(delta, V):  # noqa: N802
    """Log probabilities from delta and V via normal CDF."""
    s = delta / jnp.sqrt(V + 1e-12)
    logP = log_ndtr(-s)  # noqa: N806
    log1mP = log_ndtr(s)  # noqa: N806
    return logP, log1mP


def _sig_to_sig2(sig):
    """Convert sig to sig2: square if vector/scalar, pass through if matrix."""
    sig = jnp.asarray(sig)
    if sig.ndim <= 1:
        return jnp.square(sig)
    return sig


@jax.jit
def loglik_w(w, T, Z, sig):
    """Log-likelihood given weights w directly on the simplex."""
    zi, zj, zk = Z[:, 1], Z[:, 2], Z[:, 0]

    beta = jnp.exp(sig**2) * (jnp.exp(sig**2) - 1)

    def dv(zi, zj, zk):
        return delta_V_one_triplet(zi, zj, zk, w,
                                   beta * zi**2,
                                   beta * zj**2,
                                   beta * zk**2)
    delta, V = jax.vmap(dv)(zi, zj, zk)
    logP, log1mP = logP_log1mP_from_deltaV(delta, V)
    return jnp.sum(T * logP + (1.0 - T) * log1mP)


@jax.jit
def loglik_w_per_triplet(w, T, Z, sig):
    """Per-triplet log-likelihood given weights w on the simplex."""
    zi, zj, zk = Z[:, 1], Z[:, 2], Z[:, 0]

    beta = jnp.exp(sig**2) * (jnp.exp(sig**2) - 1)

    def dv(zi, zj, zk):
        return delta_V_one_triplet(zi, zj, zk, w,
                                   beta * zi**2,
                                   beta * zj**2,
                                   beta * zk**2)

    delta, V = jax.vmap(dv)(zi, zj, zk)
    logP, log1mP = logP_log1mP_from_deltaV(delta, V)
    return T * logP + (1.0 - T) * log1mP


@jax.jit
def loglik_theta(theta, T, Z, sig):
    """Log-likelihood in unconstrained theta-space via softmax."""
    return loglik_w(jax.nn.softmax(theta), T, Z, sig)
