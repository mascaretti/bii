"""
Sparse Dirichlet prior with Z-distributed per-feature concentrations.

The Z-distribution (Barndorff-Nielsen et al. 1982) is the distribution of
log(κ/(1-κ)) where κ ~ Beta(a, b).  Placing it on log(α_d) gives per-feature
Dirichlet concentrations with horseshoe-type shrinkage:

    w ~ Dirichlet(α_1, ..., α_p)
    log α_d ~ Z(a, b, μ, σ)        [iid given μ, σ]
    μ ~ N(0, 1)                     [global sparsity level]
    σ ~ Half-Cauchy(0, 1)           [feature heterogeneity]

For a = b = 1/2, the marginal on κ_d = α_d/(1+α_d) is Beta(1/2, 1/2),
which is U-shaped and concentrates at 0 (feature excluded, α_d ≈ 0)
and 1 (feature included, α_d >> 1).

Position vector layout (length 2p + 2):
    theta       (p)   unconstrained logits for w = softmax(theta)
    log_alpha   (p)   log concentration parameters
    mu          (1)   location hyperparameter
    log_sigma   (1)   log of scale hyperparameter
"""

import jax
from jax import numpy as jnp
from jax.scipy.special import gammaln

from bii.inference import loglik_w


def sparse_dirichlet_dim(p):
    """Total parameter dimension: 2p + 2."""
    return 2 * p + 2


def sparse_dirichlet_to_simplex(position):
    """Extract simplex weights from a sparse Dirichlet position vector."""
    p = (position.shape[0] - 2) // 2
    theta = position[:p]
    return jax.nn.softmax(theta)


def _log_z_density(z, a, b, mu, sigma):
    """Log-density of Z(a, b, mu, sigma).

    f(z; a, b, μ, σ) = (1/σ) · (1/B(a,b)) · exp(a·(z-μ)/σ) / (1 + exp((z-μ)/σ))^(a+b)

    In log-space:
        -log(σ) - log B(a,b) + a·u - (a+b)·log(1 + exp(u))
    where u = (z - μ) / σ.
    """
    u = (z - mu) / sigma
    log_beta = gammaln(a) + gammaln(b) - gammaln(a + b)
    # Use log-sum-exp trick: log(1 + exp(u)) = softplus(u)
    return -jnp.log(sigma) - log_beta + a * u - (a + b) * jax.nn.softplus(u)


@jax.jit
def log_sparse_dirichlet_posterior(position, T, Z, sig, kappa=1.0, a=0.5, b=0.5):
    """Log-posterior for the sparse Dirichlet model.

    Args:
        position: packed vector of length 2p + 2.
        T: (n,) binary triplet labels.
        Z: (n, 3, p) triplet embeddings.
        sig: noise std (scalar or (p,)).
        kappa: power-likelihood exponent.
        a, b: Z-distribution shape parameters (default: horseshoe case 1/2, 1/2).

    Returns:
        Scalar log-posterior value.
    """
    p = (position.shape[0] - 2) // 2
    theta = position[:p]
    log_alpha = position[p : 2 * p]
    mu = position[2 * p]
    log_sigma = position[2 * p + 1]

    sigma = jnp.exp(log_sigma)
    alpha = jnp.exp(log_alpha)
    w = jax.nn.softmax(theta)

    # --- Likelihood ---
    ll = kappa * loglik_w(w, T, Z, sig)

    # --- Dirichlet prior on w given alpha ---
    # log p(w|alpha) = sum (alpha_d-1) log w_d + log Gamma(sum alpha) - sum log Gamma(alpha_d)
    sum_alpha = jnp.sum(alpha)
    log_dir = (
        jnp.sum((alpha - 1.0) * jnp.log(w + 1e-12))
        + gammaln(sum_alpha)
        - jnp.sum(gammaln(alpha))
    )

    # --- Z-distribution prior on log_alpha ---
    log_prior_alpha = jnp.sum(_log_z_density(log_alpha, a, b, mu, sigma))

    # --- Hyperpriors ---
    # mu ~ N(0, 1)
    log_prior_mu = -0.5 * mu**2

    # sigma ~ Half-Cauchy(0, 1), sampled as log_sigma
    # p(sigma) ∝ 1/(1 + sigma²), Jacobian: sigma = exp(log_sigma)
    # log p(log_sigma) = log_sigma - log(1 + exp(2*log_sigma))
    log_prior_sigma = log_sigma - jax.nn.softplus(2.0 * log_sigma)

    return ll + log_dir + log_prior_alpha + log_prior_mu + log_prior_sigma
