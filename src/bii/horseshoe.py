"""
Horseshoe prior with inverse-gamma auxiliaries (Makalic & Schmidt 2015).

Parameterization for NUTS:
  - phi ∈ R^p         log-scale latent weights: phi_d = log(u_d²)
                       → w = softmax(phi)  [avoids 2^p sign symmetry of u²/Σu²]
  - log_lam_sq ∈ R^p  log local scales squared
  - log_nu ∈ R^p      log local auxiliaries
  - log_tau_sq ∈ R    log global scale squared
  - log_xi ∈ R        log global auxiliary

Packed flat vector of length 3p + 2.

Prior on phi_d:
  u_d | lam_sq_d, tau_sq ~ N(0, lam_sq_d * tau_sq)
  phi_d = log(u_d²)  ⟹  log p(phi_d) = phi_d/2 - exp(phi_d)/(2*lam_sq_d*tau_sq) + const

This is the exact density of log(u_d²) under the half-normal prior, obtained by
the change of variables u_d² = exp(phi_d) and folding the ± symmetry.
"""

import jax
from jax import numpy as jnp

from bii.inference import loglik_w


def horseshoe_dim(p):
    """Total parameter dimension for horseshoe: 3p + 2."""
    return 3 * p + 2


def horseshoe_to_simplex(position):
    """Extract simplex weights from a horseshoe position vector.

    w = softmax(phi) where phi = position[:p].
    """
    p = (position.shape[0] - 2) // 3
    phi = position[:p]
    return jax.nn.softmax(phi)


@jax.jit
def log_horseshoe_posterior(position, T, Z, sig, kappa=1.0):
    """
    Horseshoe prior with InvGamma auxiliaries (Makalic & Schmidt 2015).
    All scale parameters in log-space with Jacobian corrections.

    Hierarchy:
        u_d | lam_sq_d, tau_sq  ~ N(0, lam_sq_d * tau_sq)
        phi_d = log(u_d²)        (sampled parameter; unimodal, no sign symmetry)
        w = softmax(phi)

        lam_sq_d | nu_d         ~ InvGamma(1/2, 1/nu_d)
        nu_d                    ~ InvGamma(1/2, 1)
        tau_sq | xi             ~ InvGamma(1/2, 1/xi)
        xi                      ~ InvGamma(1/2, 1)
    """
    p = (position.shape[0] - 2) // 3
    phi = position[:p]
    log_lam_sq = position[p : 2 * p]
    log_nu = position[2 * p : 3 * p]
    log_tau_sq = position[3 * p]
    log_xi = position[3 * p + 1]

    lam_sq = jnp.exp(log_lam_sq)
    nu = jnp.exp(log_nu)
    tau_sq = jnp.exp(log_tau_sq)
    xi = jnp.exp(log_xi)

    w = jax.nn.softmax(phi)

    # --- Likelihood ---
    ll = kappa * loglik_w(w, T, Z, sig)

    # --- Prior on phi_d = log(u_d²) where u_d ~ N(0, lam_sq_d * tau_sq) ---
    # Derived via change of variables: u_d² = exp(phi_d), folding ± symmetry.
    # log p(phi_d) = phi_d/2 - exp(phi_d) / (2 * lam_sq_d * tau_sq) + const
    log_prior_phi = jnp.sum(
        0.5 * phi - jnp.exp(phi) / (2.0 * lam_sq * tau_sq)
    )

    # --- lam_sq_d | nu_d ~ InvGamma(1/2, 1/nu_d), in log-space ---
    log_prior_lam = jnp.sum(-0.5 * log_lam_sq - jnp.exp(-log_lam_sq) / nu)

    # --- nu_d ~ InvGamma(1/2, 1), in log-space ---
    log_prior_nu = jnp.sum(-0.5 * log_nu - jnp.exp(-log_nu))

    # --- tau_sq | xi ~ InvGamma(1/2, 1/xi), in log-space ---
    log_prior_tau = -0.5 * log_tau_sq - jnp.exp(-log_tau_sq) / xi

    # --- xi ~ InvGamma(1/2, 1), in log-space ---
    log_prior_xi = -0.5 * log_xi - jnp.exp(-log_xi)

    return ll + log_prior_phi + log_prior_lam + log_prior_nu + log_prior_tau + log_prior_xi
