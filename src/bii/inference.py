"""Triplet likelihood: the moments of the gap (Lemma 1) and the probit (Theorem 1)."""

import jax
from jax import numpy as jnp
from jax.scipy.special import log_ndtr


@jax.jit
def delta_V_one_triplet(zi, zj, zk, w, sig2_i, sig2_j, sig2_k):  # noqa: N802
    """Mean ``mu`` and variance ``V`` of the weighted gap of one triplet (Lemma 1).

    The gap is ``||y_i - y_k||_W^2 - ||y_j - y_k||_W^2`` with ``y_l ~ N(z_l, Sigma_l)``,
    ``Sigma_l`` diagonal and ``W = diag(w)``. Convention: ``i`` and ``j`` are the two
    candidates, ``k`` the anchor.

    Args:
        zi, zj, zk: (p,) source points.
        w: (p,) weights on the simplex.
        sig2_i, sig2_j, sig2_k: noise variances, scalars (isotropic) or (p,).

    Returns:
        ``(mu, V)``, two scalars.
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


def _delta_V_all(w, Z, sig):  # noqa: N802
    """Per-triplet ``(mu, V)`` under ``w``, both of shape ``(n,)``.

    ``Z`` is ``(n, 3, p)`` with the anchor in column 0. ``sig`` is the noise standard
    deviation: a scalar, per feature ``(p,)``, or per triplet and point, ``(n, 3)`` or
    ``(n, 3, p)``, in the column order of ``Z``. Full covariance matrices are not
    supported.
    """
    zi, zj, zk = Z[:, 1], Z[:, 2], Z[:, 0]
    sig = jnp.asarray(sig)
    if sig.ndim >= 2:
        n, p = zi.shape[0], zi.shape[1]
        if sig.shape not in {(n, 3), (n, 3, p)}:
            raise ValueError(
                f"sig of shape {sig.shape} is not a per-triplet sigma array; expected "
                f"(n_triplets, 3) = ({n}, 3) or (n_triplets, 3, p) = ({n}, 3, {p})."
            )
        sig2_i = sig[:, 1] ** 2
        sig2_j = sig[:, 2] ** 2
        sig2_k = sig[:, 0] ** 2

        def dv(zi, zj, zk, s2i, s2j, s2k):
            return delta_V_one_triplet(zi, zj, zk, w, s2i, s2j, s2k)

        return jax.vmap(dv)(zi, zj, zk, sig2_i, sig2_j, sig2_k)

    sig2 = jnp.square(jnp.asarray(sig))

    def dv(zi, zj, zk):
        return delta_V_one_triplet(zi, zj, zk, w, sig2, sig2, sig2)

    return jax.vmap(dv)(zi, zj, zk)


def _log_probs(delta, V, lam=1.0):
    """``log P(T = 1)`` and ``log P(T = 0)`` under the probit, at slope ``lam``.

    ``s = lam * mu / sqrt(V)`` and ``P(T = 1) = Phi(-s)``: label 1 means that the
    first candidate is the closer one.
    """
    s = lam * delta / jnp.sqrt(V + 1e-12)
    return log_ndtr(-s), log_ndtr(s)


def loglik_w_per_triplet(w, T, Z, sig, lam=1.0):
    """Log-likelihood of each triplet, shape ``(n,)``; see :func:`loglik_w`."""
    delta, V = _delta_V_all(w, Z, sig)
    logP, log1mP = _log_probs(delta, V, lam)  # noqa: N806
    return T * logP + (1.0 - T) * log1mP


def loglik_w(w, T, Z, sig, lam=1.0):
    """Log-likelihood of the triplet labels given weights ``w`` on the simplex.

    Args:
        w: (p,) weights.
        T: (n,) labels, 1 when the first candidate is closer to the anchor.
        Z: (n, 3, p) source triplets, anchor in column 0.
        sig: noise standard deviation; see :func:`_delta_V_all`.
        lam: calibration slope, ``s -> lam * s``; 1 is the plain probit.
    """
    delta, V = _delta_V_all(w, Z, sig)
    logP, log1mP = _log_probs(delta, V, lam)  # noqa: N806
    return jnp.sum(T * logP + (1.0 - T) * log1mP)


def slope_scale(w, Z, sig):
    """Confidence ``s_bar(w)``: the root-mean-square standardised gap; uses no labels.

    Converts a calibrated slope into a slope on ``s``: ``lam = slope / slope_scale(w)``.
    """
    delta, V = _delta_V_all(w, Z, sig)
    s2 = delta**2 / (V + 1e-12)
    return jnp.sqrt(jnp.mean(s2) + 1e-12)


def loglik_w_slope_grid(w, T, Z, sig, slope_grid):
    """Log-likelihood at each calibrated slope of a grid, shape ``(R,)``.

    Entry ``r`` is :func:`loglik_w` at ``lam = slope_grid[r] / slope_scale(w)``. On the
    calibrated scale every ``w`` faces the same grid of confidences, so a less
    confident metric gains no prior mass (the collapse of the paper's Section 5).

    Args:
        w, T, Z, sig: as in :func:`loglik_w`.
        slope_grid: (R,) positive calibrated slopes, e.g.
            :func:`bii.priors.make_slope_grid`.
    """
    scale = slope_scale(w, Z, sig)

    def one(slope):
        return loglik_w(w, T, Z, sig, lam=slope / scale)

    return jax.vmap(one)(jnp.asarray(slope_grid))


def slope_posterior(w, T, Z, sig, slope_grid, kappa=1.0):
    """Posterior over the calibrated-slope grid given ``w``, shape ``(R,)``.

    Equal prior weights on the grid and the ``kappa``-tempered likelihood, as in
    :func:`bii.priors.make_dirichlet_logposterior`.
    """
    ll = loglik_w_slope_grid(w, T, Z, sig, slope_grid)
    return jax.nn.softmax(kappa * ll)
