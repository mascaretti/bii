"""Likelihood functions for triplet comparisons — all pure functions."""

import jax
from jax import numpy as jnp
from jax.scipy.special import log_ndtr


@jax.jit
def delta_V_one_triplet(zi, zj, zk, w, sig2):  # noqa: N802
    """Compute mean (delta) and variance (V) for the Gaussian approximation.

    Args:
        zi, zj, zk: (p,) embeddings of the three triplet points.
        w: (p,) simplex weights.
        sig2: (p,) diagonal variances, or (p, p) full covariance matrix.

    Returns:
        ``(delta, V)`` — both scalars.
    """
    a = zi - zk
    b = zj - zk
    delta = jnp.sum(w * (a * a - b * b))

    if sig2.ndim == 1:
        w2_sig2 = (w * w) * sig2
        aa = jnp.sum(w2_sig2 * (a * a))
        bb = jnp.sum(w2_sig2 * (b * b))
        ab = jnp.sum(w2_sig2 * (a * b))
        tr = jnp.sum((w * w) * (sig2 * sig2))
    else:
        M = w[:, None] * sig2 * w[None, :]  # noqa: N806
        aa = a @ M @ a
        bb = b @ M @ b
        ab = a @ M @ b
        WS = w[:, None] * sig2  # noqa: N806
        tr = jnp.sum(M * sig2)
        ##tr = jnp.sum(WS * WS)

    V = 8.0 * (aa + bb - ab) + 12.0 * tr
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
    sig2 = _sig_to_sig2(sig)
    zi, zj, zk = Z[:, 1], Z[:, 2], Z[:, 0]

    def dv(zi, zj, zk):
        return delta_V_one_triplet(zi, zj, zk, w, sig2)

    delta, V = jax.vmap(dv)(zi, zj, zk)
    logP, log1mP = logP_log1mP_from_deltaV(delta, V)
    return jnp.sum(T * logP + (1.0 - T) * log1mP)


@jax.jit
def loglik_w_per_triplet(w, T, Z, sig):
    """Per-triplet log-likelihood given weights w on the simplex."""
    sig2 = _sig_to_sig2(sig)
    zi, zj, zk = Z[:, 1], Z[:, 2], Z[:, 0]

    def dv(zi, zj, zk):
        return delta_V_one_triplet(zi, zj, zk, w, sig2)

    delta, V = jax.vmap(dv)(zi, zj, zk)
    logP, log1mP = logP_log1mP_from_deltaV(delta, V)
    return T * logP + (1.0 - T) * log1mP


@jax.jit
def loglik_theta(theta, T, Z, sig):
    """Log-likelihood in unconstrained theta-space via softmax."""
    return loglik_w(jax.nn.softmax(theta), T, Z, sig)
