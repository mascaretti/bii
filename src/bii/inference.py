"""Likelihood functions for triplet comparisons — all pure functions."""

import jax
from jax import numpy as jnp
from jax.scipy.special import log_ndtr


@jax.jit
def delta_V_one_triplet(zi, zj, zk, w, sig2_i, sig2_j, sig2_k):  # noqa: N802
    """Compute mean (mu) and variance (V) for the Gaussian probit approximation.

    Supports both per-point isotropic noise (sig2_* scalars) and
    general diagonal noise (sig2_* as (p,) vectors).

    From Theorem 1 (sample.tex): y_l | z_l ~ N(z_l, Σ_l), Σ_l diagonal.

    Convention: i = candidate 1, j = candidate 2, k = anchor.

    Args:
        zi, zj, zk: (p,) embeddings of the three triplet points.
        w: (p,) simplex weights.
        sig2_i, sig2_j, sig2_k: per-point noise variances.
            Scalar: per-point isotropic (σ_l² I).
            (p,): general diagonal (Σ_l = diag(σ²_{l,d})).

    Returns:
        ``(mu, V)`` — both scalars.
    """
    a = zi - zk
    b = zj - zk
    w2 = w * w

    # S_u = Σ_i + Σ_k, S_v = Σ_j + Σ_k, S_k = Σ_k  (diagonal entries)
    su = sig2_i + sig2_k
    sv = sig2_j + sig2_k

    # Mean: δ(w) + tr(W (S_u - S_v))  =  δ(w) + Σ w_d (σ²_{i,d} - σ²_{j,d})
    mu = jnp.sum(w * (a * a - b * b)) + jnp.sum(w * (sig2_i - sig2_j))

    # Variance of L: 4(a' W S_u W a + b' W S_v W b - 2 a' W S_k W b)
    var_L = (
        4.0 * jnp.sum(w2 * su * a * a)
        + 4.0 * jnp.sum(w2 * sv * b * b)
        - 8.0 * jnp.sum(w2 * sig2_k * a * b)
    )

    # Variance of Q: 2 tr((W S_u)²) + 2 tr((W S_v)²) - 4 tr((W S_k)²)
    var_Q = (
        2.0 * jnp.sum(w2 * su * su)
        + 2.0 * jnp.sum(w2 * sv * sv)
        - 4.0 * jnp.sum(w2 * sig2_k * sig2_k)
    )

    V = var_L + var_Q
    return mu, V


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


def _make_sig2_fn(sig, noise_model):
    """Return a function z_l -> sig2_l for the given noise model."""
    sig2 = _sig_to_sig2(sig)
    if noise_model == "additive":
        def sig2_fn(z_l):
            return sig2
        return sig2_fn
    elif noise_model == "multiplicative":
        beta = jnp.exp(sig2) * (jnp.exp(sig2) - 1.0)
        def sig2_fn(z_l):
            return beta * z_l**2
        return sig2_fn
    else:
        raise ValueError(f"Unknown noise_model: {noise_model!r}")


def loglik_w(w, T, Z, sig, noise_model="additive"):
    """Log-likelihood given weights w directly on the simplex.

    Args:
        w: (p,) simplex weights.
        T: (n,) binary triplet labels.
        Z: (n, 3, p) triplet embeddings.
        sig: noise std — scalar or (p,).
        noise_model: ``"additive"`` (shared σ) or ``"multiplicative"`` (β·z²).
    """
    sig2_fn = _make_sig2_fn(sig, noise_model)
    zi, zj, zk = Z[:, 1], Z[:, 2], Z[:, 0]

    def dv(zi, zj, zk):
        return delta_V_one_triplet(zi, zj, zk, w,
                                   sig2_fn(zi), sig2_fn(zj), sig2_fn(zk))

    delta, V = jax.vmap(dv)(zi, zj, zk)
    logP, log1mP = logP_log1mP_from_deltaV(delta, V)
    return jnp.sum(T * logP + (1.0 - T) * log1mP)


def loglik_w_per_triplet(w, T, Z, sig, noise_model="additive"):
    """Per-triplet log-likelihood given weights w on the simplex."""
    sig2_fn = _make_sig2_fn(sig, noise_model)
    zi, zj, zk = Z[:, 1], Z[:, 2], Z[:, 0]

    def dv(zi, zj, zk):
        return delta_V_one_triplet(zi, zj, zk, w,
                                   sig2_fn(zi), sig2_fn(zj), sig2_fn(zk))

    delta, V = jax.vmap(dv)(zi, zj, zk)
    logP, log1mP = logP_log1mP_from_deltaV(delta, V)
    return T * logP + (1.0 - T) * log1mP


def loglik_theta(theta, T, Z, sig, noise_model="additive"):
    """Log-likelihood in unconstrained theta-space via softmax."""
    return loglik_w(jax.nn.softmax(theta), T, Z, sig, noise_model)
