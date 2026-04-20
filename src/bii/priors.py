"""Prior log-densities as composable factory functions.

Each ``make_*_logposterior`` closes over data and returns a pure
``logprob_fn(position) -> scalar`` suitable for NUTS or VI.
"""

import jax
from jax import numpy as jnp

from bii.inference import loglik_w


def make_dirichlet_logposterior(T, Z, sig, alpha, kappa=1.0, noise_model="additive"):
    """Return ``logprob_fn(theta) -> scalar`` for a Dirichlet prior.

    Args:
        T: (n,) binary triplet labels.
        Z: (n, 3, p) triplet embeddings.
        sig: noise std — scalar or (p,).
        alpha: (p,) Dirichlet concentration.
        kappa: power-likelihood exponent.
        noise_model: ``"additive"`` or ``"multiplicative"``.
    """
    def logprob_fn(theta):
        w = jax.nn.softmax(theta)
        theta_anchor = -0.5 * 1e-4 * jnp.sum(theta ** 2)
        # 2. Mathematically correct Dirichlet log-prior uses (alpha - 1)
        # If alpha=1 (uniform), this term becomes 0, allowing features to gracefully drop to 0
        dirichlet = jnp.sum((alpha - 1.0) * jnp.log(w + 1e-12))
        return kappa * loglik_w(w, T, Z, sig, noise_model) + dirichlet + theta_anchor
    return logprob_fn

# def make_dirichlet_logposterior(T, Z, sig, alpha, kappa=1.0, noise_model="additive"):
#     """Return ``logprob_fn(theta) -> scalar`` for a Dirichlet prior.

#     Args:
#         T: (n,) binary triplet labels.
#         Z: (n, 3, p) triplet embeddings.
#         sig: noise std — scalar or (p,).
#         alpha: (p,) Dirichlet concentration.
#         kappa: power-likelihood exponent.
#         noise_model: ``"additive"`` or ``"multiplicative"``.
#     """

#     def logprob_fn(theta):
#         w = jax.nn.softmax(theta)
#         return kappa * loglik_w(w, T, Z, sig, noise_model) + jnp.sum(alpha * jnp.log(w + 1e-12))

#     return logprob_fn
