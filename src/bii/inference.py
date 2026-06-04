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


def logP_log1mP_from_deltaV(delta, V, clip_s=None):  # noqa: N802
    """Log probabilities from delta and V via normal CDF.

    When ``clip_s`` is given, ``s = delta / sqrt(V)`` is truncated to
    ``[-clip_s, clip_s]`` before the normal CDF. The clipped saturating
    regime contributes a bounded loss and zero gradient in ``w``, which
    defuses the magnesium-collapse failure mode without changing the
    sampler. Equivalent to a censored-probit likelihood with the censoring
    threshold at ``clip_s``.
    """
    s = delta / jnp.sqrt(V + 1e-12)
    if clip_s is not None:
        s = jnp.clip(s, -clip_s, clip_s)
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


def _resolve_sig2(sig, noise_model, zi, zj, zk):
    """Resolve per-triplet noise variances.

    Supports:
        - scalar or (p,): global noise, dispatched via noise_model.
        - (n_triplets, 3): per-point isotropic, pre-resolved via indices.
        - (n_triplets, 3, p): per-point diagonal, pre-resolved via indices.

    Returns:
        ``(sig2_i, sig2_j, sig2_k)`` — each scalar, (n_triplets,), or (n_triplets, p).
    """
    sig = jnp.asarray(sig)
    if sig.ndim >= 2:
        # Pre-resolved per-triplet sigmas.
        # Column order matches Z: 0=anchor(k), 1=dest1(i), 2=dest2(j)
        sig2_i = sig[:, 1] ** 2
        sig2_j = sig[:, 2] ** 2
        sig2_k = sig[:, 0] ** 2
        return sig2_i, sig2_j, sig2_k
    else:
        sig2_fn = _make_sig2_fn(sig, noise_model)
        return sig2_fn(zi), sig2_fn(zj), sig2_fn(zk)


def loglik_w(w, T, Z, sig, noise_model="additive", triplet_weights=None, clip_s=None):
    """Log-likelihood given weights w directly on the simplex.

    Args:
        w: (p,) simplex weights.
        T: (n,) binary triplet labels.
        Z: (n, 3, p) triplet embeddings.
        sig: noise std — scalar, (p,), or (n_triplets, 3) pre-resolved.
        noise_model: ``"additive"`` (shared σ) or ``"multiplicative"`` (β·z²).
            Ignored when sig is (n_triplets, 3).
        triplet_weights: optional (n,) per-triplet importance weights. When
            provided, the loglik becomes ``sum(weights * per_triplet_logP)``
            instead of the plain sum. Designed for importance-weighted
            samplers (see :func:`bii.data.make_triplets_rank_weighted`).
        clip_s: optional float. When set, the per-triplet probit statistic
            ``s = delta / sqrt(V)`` is clipped to ``[-clip_s, clip_s]``
            before the normal CDF. Bounds the saturating contribution of
            any single triplet and zeroes the gradient in ``w`` outside
            the trust region — a censored-probit robustifier.
    """
    zi, zj, zk = Z[:, 1], Z[:, 2], Z[:, 0]
    sig = jnp.asarray(sig)

    if sig.ndim >= 2:
        # Pre-resolved per-triplet sigmas — pass sig2 directly, no sig2_fn
        sig2_i, sig2_j, sig2_k = _resolve_sig2(sig, noise_model, zi, zj, zk)

        def dv(zi, zj, zk, s2i, s2j, s2k):
            return delta_V_one_triplet(zi, zj, zk, w, s2i, s2j, s2k)

        delta, V = jax.vmap(dv)(zi, zj, zk, sig2_i, sig2_j, sig2_k)
    else:
        sig2_fn = _make_sig2_fn(sig, noise_model)

        def dv(zi, zj, zk):
            return delta_V_one_triplet(zi, zj, zk, w,
                                       sig2_fn(zi), sig2_fn(zj), sig2_fn(zk))

        delta, V = jax.vmap(dv)(zi, zj, zk)

    logP, log1mP = logP_log1mP_from_deltaV(delta, V, clip_s=clip_s)
    per_triplet = T * logP + (1.0 - T) * log1mP
    if triplet_weights is None:
        return jnp.sum(per_triplet)
    return jnp.sum(triplet_weights * per_triplet)


def loglik_w_per_triplet(w, T, Z, sig, noise_model="additive"):
    """Per-triplet log-likelihood given weights w on the simplex."""
    zi, zj, zk = Z[:, 1], Z[:, 2], Z[:, 0]
    sig = jnp.asarray(sig)

    if sig.ndim >= 2:
        sig2_i, sig2_j, sig2_k = _resolve_sig2(sig, noise_model, zi, zj, zk)

        def dv(zi, zj, zk, s2i, s2j, s2k):
            return delta_V_one_triplet(zi, zj, zk, w, s2i, s2j, s2k)

        delta, V = jax.vmap(dv)(zi, zj, zk, sig2_i, sig2_j, sig2_k)
    else:
        sig2_fn = _make_sig2_fn(sig, noise_model)

        def dv(zi, zj, zk):
            return delta_V_one_triplet(zi, zj, zk, w,
                                       sig2_fn(zi), sig2_fn(zj), sig2_fn(zk))

        delta, V = jax.vmap(dv)(zi, zj, zk)

    logP, log1mP = logP_log1mP_from_deltaV(delta, V)
    return T * logP + (1.0 - T) * log1mP


def loglik_theta(theta, T, Z, sig, noise_model="additive"):
    """Log-likelihood in unconstrained theta-space via softmax."""
    return loglik_w(jax.nn.softmax(theta), T, Z, sig, noise_model)
