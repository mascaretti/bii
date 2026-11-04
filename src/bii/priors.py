"""The log posterior of the weights under a Dirichlet prior.

``make_dirichlet_logposterior`` closes over the data and returns a pure
``logprob_fn(theta) -> scalar`` with ``w = softmax(theta)``, for NUTS or MAP.
"""

import jax
from jax import numpy as jnp
from jax.scipy.special import logsumexp

from bii.inference import loglik_w, loglik_w_slope_grid


def make_slope_grid(lo=1e-3, hi=10.0, num=49):
    """Log-spaced grid of calibrated slopes, for ``slope_grid``.

    Equal weights on this grid are a log-uniform prior on the calibrated slope over
    ``[lo, hi]``. Values near ``lo`` make every triplet close to a coin flip; the value
    :func:`bii.inference.slope_scale` recovers the plain likelihood.
    """
    return jnp.exp(jnp.linspace(jnp.log(lo), jnp.log(hi), num))


def _sum_logdensity(theta):
    """Standard normal log-density of the component of ``theta`` along ``(1, ..., 1)``.

    ``softmax(theta)`` ignores that direction, so without this term the posterior is
    flat along it and NUTS drifts. The density factorises into a function of ``w`` and
    one of ``sum(theta)``, so the term makes the posterior proper and leaves the law of
    ``w`` unchanged.
    """
    return -0.5 * jnp.sum(theta) ** 2 / theta.shape[0]


def make_dirichlet_logposterior(T, Z, sig, alpha, kappa=1.0, slope_grid=None):
    """Return ``logprob_fn(theta) -> scalar`` for a Dirichlet(``alpha``) prior on ``w``.

    Args:
        T: (n,) labels.
        Z: (n, 3, p) source triplets.
        sig: noise standard deviation; see :func:`bii.inference.loglik_w`.
        alpha: (p,) Dirichlet concentration.
        kappa: temperature of the likelihood (Theorem 2 of the paper).
        slope_grid: optional (R,) positive calibrated slopes, e.g.
            :func:`make_slope_grid`. The calibration slope is then integrated out with
            equal weights on the grid, inside the tempering:
            ``logsumexp_r(kappa * ll_r) - log R``. The position stays ``theta``.

    The density on ``theta`` includes the Jacobian of the softmax, so the prior term
    is ``sum(alpha * log w)``; every posterior also carries :func:`_sum_logdensity`.
    """
    if slope_grid is not None:
        slope_grid = jnp.asarray(slope_grid)
        if (slope_grid.ndim != 1 or slope_grid.size == 0
                or not bool(jnp.all(slope_grid > 0))):
            raise ValueError("slope_grid must be a non-empty 1-D array of positive slopes.")
        log_r = jnp.log(slope_grid.shape[0])

        def logprob_fn_with_slope(theta):
            w = jax.nn.softmax(theta)
            ll = loglik_w_slope_grid(w, T, Z, sig, slope_grid)
            return (logsumexp(kappa * ll) - log_r
                    + jnp.sum(alpha * jnp.log(w + 1e-12)) + _sum_logdensity(theta))

        return logprob_fn_with_slope

    def logprob_fn(theta):
        w = jax.nn.softmax(theta)
        ll = loglik_w(w, T, Z, sig)
        return (kappa * ll + jnp.sum(alpha * jnp.log(w + 1e-12))
                + _sum_logdensity(theta))

    return logprob_fn
