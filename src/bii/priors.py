"""Prior log-densities as composable factory functions.

Each ``make_*_logposterior`` closes over data and returns a pure
``logprob_fn(position) -> scalar`` suitable for NUTS or VI.
"""

import jax
from jax import numpy as jnp
from jax.scipy.special import gammaln

from bii.inference import loglik_w

# ---------------------------------------------------------------------------
# Dirichlet
# ---------------------------------------------------------------------------


def make_dirichlet_logposterior(T, Z, sig, alpha, kappa=1.0):
    """Return ``logprob_fn(theta) -> scalar`` for a Dirichlet prior.

    Args:
        T: (n,) binary triplet labels.
        Z: (n, 3, p) triplet embeddings.
        sig: noise std (scalar, (p,), or (p, p) covariance).
        alpha: (p,) Dirichlet concentration.
        kappa: power-likelihood exponent.
    """

    def logprob_fn(theta):
        w = jax.nn.softmax(theta)
        return kappa * loglik_w(w, T, Z, sig) + jnp.sum(alpha * jnp.log(w + 1e-12))

    return logprob_fn


# ---------------------------------------------------------------------------
# Sparse Dirichlet  (Z-distributed per-feature concentrations)
# ---------------------------------------------------------------------------


def sparse_dirichlet_dim(p):
    """Total parameter dimension: ``2p + 2``."""
    return 2 * p + 2


def sparse_dirichlet_to_simplex(position):
    """Extract simplex weights from a sparse Dirichlet position vector."""
    p = (position.shape[0] - 2) // 2
    theta = position[:p]
    return jax.nn.softmax(theta)


def _log_z_density(z, a, b, mu, sigma):
    """Log-density of the Z-distribution ``Z(a, b, mu, sigma)``.

    ``f(z) = (1/sigma) * (1/B(a,b)) * exp(a*u) / (1 + exp(u))^(a+b)``
    where ``u = (z - mu) / sigma``.
    """
    u = (z - mu) / sigma
    log_beta = gammaln(a) + gammaln(b) - gammaln(a + b)
    return -jnp.log(sigma) - log_beta + a * u - (a + b) * jax.nn.softplus(u)


@jax.jit
def log_sparse_dirichlet_posterior(position, T, Z, sig, kappa=1.0, a=0.5, b=0.5):
    """Log-posterior for the sparse Dirichlet model.

    Position vector layout (length ``2p + 2``):
        ``theta (p) | log_alpha (p) | mu (1) | log_sigma (1)``

    Args:
        position: packed vector of length ``2p + 2``.
        T: (n,) binary triplet labels.
        Z: (n, 3, p) triplet embeddings.
        sig: noise std (scalar or (p,)).
        kappa: power-likelihood exponent.
        a, b: Z-distribution shape parameters.
    """
    p = (position.shape[0] - 2) // 2
    theta = position[:p]
    log_alpha = position[p : 2 * p]
    mu = position[2 * p]
    log_sigma = position[2 * p + 1]

    sigma = jnp.exp(log_sigma)
    alpha = jnp.exp(log_alpha)
    w = jax.nn.softmax(theta)

    # Likelihood
    ll = kappa * loglik_w(w, T, Z, sig)

    # Dirichlet prior on w given alpha
    sum_alpha = jnp.sum(alpha)
    log_dir = (
        jnp.sum(alpha * jnp.log(w + 1e-12))
        + gammaln(sum_alpha)
        - jnp.sum(gammaln(alpha))
    )

    # Z-distribution prior on log_alpha
    log_prior_alpha = jnp.sum(_log_z_density(log_alpha, a, b, mu, sigma))

    # Hyperpriors: mu ~ N(0, 1), sigma ~ Half-Cauchy(0, 1)
    log_prior_mu = -0.5 * mu**2
    log_prior_sigma = log_sigma - jax.nn.softplus(2.0 * log_sigma)

    return ll + log_dir + log_prior_alpha + log_prior_mu + log_prior_sigma


def make_sparse_dirichlet_logposterior(T, Z, sig, kappa=1.0, a=0.5, b=0.5):
    """Return ``logprob_fn(position) -> scalar`` for sparse Dirichlet.

    Args:
        T: (n,) binary triplet labels.
        Z: (n, 3, p) triplet embeddings.
        sig: noise std.
        kappa: power-likelihood exponent.
        a, b: Z-distribution shape parameters.
    """

    def logprob_fn(position):
        return log_sparse_dirichlet_posterior(position, T, Z, sig, kappa, a, b)

    return logprob_fn
