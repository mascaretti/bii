"""Prior log-densities as composable factory functions.

Each ``make_*_logposterior`` closes over data and returns a pure
``logprob_fn(position) -> scalar`` suitable for NUTS or VI.
"""

import jax
from jax import numpy as jnp

from bii.inference import loglik_w

def make_dirichlet_logposterior(T, Z, sig, alpha, kappa=1.0, noise_model="additive",
                                triplet_weights=None, clip_s=None, pi_inclusion=None):
    """Return ``logprob_fn(theta) -> scalar`` for a Dirichlet prior.

    Args:
        T: (n,) binary triplet labels.
        Z: (n, 3, p) triplet embeddings.
        sig: noise std — scalar or (p,).
        alpha: (p,) Dirichlet concentration.
        kappa: power-likelihood exponent.
        noise_model: ``"additive"`` or ``"multiplicative"``.
        triplet_weights: optional (n,) per-triplet importance weights forwarded
            to :func:`bii.inference.loglik_w`.
        clip_s: optional float; if set, clips the per-triplet ``s`` statistic
            to ``[-clip_s, clip_s]`` before the normal CDF. Bounded-influence
            (censored-probit) robustifier against saturating triplets.
        pi_inclusion: optional float in (0, 1); enables the inclusion-mixture
            likelihood, see :func:`bii.inference.loglik_w`.
    """

    def logprob_fn(theta):
        w = jax.nn.softmax(theta)
        ll = loglik_w(w, T, Z, sig, noise_model,
                      triplet_weights=triplet_weights, clip_s=clip_s,
                      pi_inclusion=pi_inclusion)
        return kappa * ll + jnp.sum(alpha * jnp.log(w + 1e-12))

    return logprob_fn
