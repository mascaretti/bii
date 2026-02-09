"""
Power Likelihood Correction for Triplet Correlations

This module provides the κ correction for Bayesian inference on metric weights
when triplets share destinations (ρ_dest ≈ 0.1 for Gaussian embeddings).

Key results from theory:
- ρ_anchor = 0: Reusing anchors is FREE (no correlation)
- ρ_dest ≈ 0.1: Reusing destinations costs correlation
- κ = 1/(1 + 2*ρ_dest*(k-1)) for balanced designs with k triplets per destination

Usage:
    from bii.power_correction import compute_kappa, sample_posterior_corrected
    
    # Compute correction from triplet indices
    kappa, info = compute_kappa(indices, rho_dest=0.1)
    
    # Sample with correction
    results = sample_posterior_corrected(key, X, Z, sig, alpha, indices)
"""

import jax
from jax import numpy as jnp
from jax import random
from jax.scipy.special import log_ndtr
import numpy as np
from collections import Counter

from bii.data import T_from_X


# =============================================================================
# KAPPA COMPUTATION
# =============================================================================

def compute_kappa(indices, rho_dest=0.1, rho_anchor=0.0):
    """
    Compute power likelihood correction factor κ from triplet indices.
    
    The power posterior is: π_κ(w|Y) ∝ π(w) · [p(Y|w)]^κ
    
    This corrects for correlations induced by destination reuse.
    
    Args:
        indices: Array (n_triplets, 3) with [anchor, dest1, dest2] per row
        rho_dest: Correlation between triplets sharing a destination (~0.1 for Gaussian)
        rho_anchor: Correlation between triplets sharing only anchor (0 for iid)
        
    Returns:
        kappa: Correction factor in (0, 1]
        info: Dictionary with diagnostics
    """
    indices = np.asarray(indices)
    n_triplets = len(indices)
    
    if n_triplets == 0:
        return 1.0, {'lambda': 1.0, 'n_eff': 0, 'n_triplets': 0}
    
    # Count destination usage (columns 1 and 2)
    destinations = indices[:, 1:].flatten()
    dest_counts = Counter(destinations)
    
    # Count anchor usage (column 0)
    anchors = indices[:, 0]
    anchor_counts = Counter(anchors)
    
    # Variance inflation from destinations: Σ_i T_i(T_i - 1) / T
    dest_overlap_sum = sum(t * (t - 1) for t in dest_counts.values())
    lambda_dest = rho_dest * dest_overlap_sum / n_triplets
    
    # Variance inflation from anchors (should be ~0 since rho_anchor ≈ 0)
    anchor_overlap_sum = sum(t * (t - 1) for t in anchor_counts.values())
    lambda_anchor = rho_anchor * anchor_overlap_sum / n_triplets
    
    lambda_total = 1 + lambda_dest + lambda_anchor
    kappa = 1 / lambda_total
    n_eff = n_triplets * kappa
    
    dest_T = np.array(list(dest_counts.values()))
    anchor_T = np.array(list(anchor_counts.values()))
    
    info = {
        'lambda': lambda_total,
        'lambda_dest': lambda_dest,
        'lambda_anchor': lambda_anchor,
        'kappa': kappa,
        'n_eff': n_eff,
        'n_triplets': n_triplets,
        'efficiency': kappa,
        'n_unique_destinations': len(dest_counts),
        'n_unique_anchors': len(anchor_counts),
        'mean_dest_reuse': float(np.mean(dest_T)),
        'max_dest_reuse': int(np.max(dest_T)),
        'mean_anchor_reuse': float(np.mean(anchor_T)),
        'max_anchor_reuse': int(np.max(anchor_T)),
    }
    
    return float(kappa), info


def compute_kappa_balanced(k_dest, rho_dest=0.1):
    """
    Compute κ for a balanced design where each destination appears k times.
    
    For balanced design with single anchor:
        λ = 1 + 2 * ρ_dest * (k - 1)
        κ = 1 / λ
    
    Args:
        k_dest: Number of times each destination is used
        rho_dest: Destination correlation (default 0.1)
        
    Returns:
        kappa: Correction factor
        
    Examples:
        k=1:  κ = 1.000 (independent triplets)
        k=2:  κ = 0.833
        k=5:  κ = 0.556
        k=10: κ = 0.357
    """
    lambda_val = 1 + 2 * rho_dest * (k_dest - 1)
    return 1 / lambda_val


# =============================================================================
# LIKELIHOOD WITH POWER CORRECTION
# =============================================================================

@jax.jit
def _delta_V_one_triplet(zi, zj, zk, w, sig2):
    """Compute mean and variance of distance difference for one triplet."""
    a = zi - zk
    b = zj - zk
    delta = jnp.sum(w * (a*a - b*b))
    w2_sig2 = (w*w) * sig2
    aa = jnp.sum(w2_sig2 * (a*a))
    bb = jnp.sum(w2_sig2 * (b*b))
    ab = jnp.sum(w2_sig2 * (a*b))
    tr = jnp.sum((w*w) * (sig2*sig2))
    V = 8.0 * (aa + bb - ab) + 12.0 * tr
    return delta, V


@jax.jit
def loglik_theta(theta, T, Z, sig):
    """
    Compute log-likelihood for triplet model.
    
    Args:
        theta: Unconstrained parameters (p,)
        T: Triplet outcomes (n,) in {0, 1}
        Z: Embeddings (n, 3, p)
        sig: Noise std (p,)
    """
    sig2 = jnp.square(sig)
    w = jax.nn.softmax(theta)
    zi, zj, zk = Z[:, 1], Z[:, 2], Z[:, 0]
    
    def dv(zi, zj, zk):
        return _delta_V_one_triplet(zi, zj, zk, w, sig2)
    
    delta, V = jax.vmap(dv)(zi, zj, zk)
    s = delta / jnp.sqrt(V + 1e-12)
    logP = log_ndtr(-s)
    log1mP = log_ndtr(s)
    
    return jnp.sum(T * logP + (1.0 - T) * log1mP)


@jax.jit
def loglik_theta_power(theta, T, Z, sig, kappa):
    """Compute power log-likelihood: κ · ℓ(θ)."""
    return kappa * loglik_theta(theta, T, Z, sig)


@jax.jit
def log_prior_dirichlet(theta, alpha):
    """Log prior for Dirichlet(α) on w = softmax(θ)."""
    w = jax.nn.softmax(theta)
    return jnp.sum((alpha - 1.0) * jnp.log(w + 1e-12))


@jax.jit
def log_posterior_power(theta, T, Z, sig, alpha, kappa):
    """Log posterior with power likelihood: κ·ℓ(θ) + log π(θ)."""
    return loglik_theta_power(theta, T, Z, sig, kappa) + log_prior_dirichlet(theta, alpha)


# =============================================================================
# MCMC SAMPLING
# =============================================================================

def sample_posterior_nuts_power(key, X, Z, sig, alpha, kappa=1.0,
                                 num_samples=1000, num_warmup=500, num_chains=4,
                                 step_size=1e-3, target_acceptance_rate=0.8):
    """
    Sample from power posterior using BlackJAX NUTS.
    
    π_κ(θ | T) ∝ π(θ) · [p(T | θ)]^κ
    
    Args:
        key: JAX random key
        X: Triplet data (n, 3, p)
        Z: Normalized embeddings (n, 3, p)
        sig: Noise std (p,)
        alpha: Dirichlet prior concentration (p,)
        kappa: Power correction factor (default 1.0 = no correction)
        num_samples: Samples per chain
        num_warmup: Warmup iterations
        num_chains: Number of chains
        
    Returns:
        Dictionary with theta_samples, w_samples, kappa, diagnostics
    """
    import blackjax
    
    T_obs = T_from_X(X)
    p = Z.shape[2]
    
    def logprob_fn(theta):
        return log_posterior_power(theta, T_obs, Z, sig, alpha, kappa)
    
    init_theta = jnp.zeros(p)
    
    key, *init_keys = random.split(key, num_chains + 1)
    init_positions = jnp.stack([
        init_theta + 0.1 * random.normal(k, shape=(p,))
        for k in init_keys
    ])
    
    warmup = blackjax.window_adaptation(
        blackjax.nuts,
        logprob_fn,
        num_warmup,
        target_acceptance_rate=target_acceptance_rate,
        initial_step_size=step_size,
    )
    
    keys_warmup = random.split(key, num_chains)
    
    all_theta_samples = []
    all_acceptance_rates = []
    
    for chain_idx in range(num_chains):
        (state, params), _ = warmup.run(keys_warmup[chain_idx], init_positions[chain_idx])
        kernel = blackjax.nuts(logprob_fn, **params)
        
        def one_step(state, key):
            new_state, info = kernel.step(key, state)
            return new_state, (new_state.position, info.acceptance_rate)
        
        key, subkey = random.split(key)
        sample_keys = random.split(subkey, num_samples)
        
        _, (theta_samples_chain, acceptance_rates_chain) = jax.lax.scan(
            one_step, state, sample_keys
        )
        
        all_theta_samples.append(theta_samples_chain)
        all_acceptance_rates.append(acceptance_rates_chain)
    
    theta_samples = jnp.stack(all_theta_samples, axis=1)
    acceptance_rates = jnp.stack(all_acceptance_rates, axis=1)
    w_samples = jax.vmap(jax.vmap(jax.nn.softmax))(theta_samples)
    
    return {
        'theta_samples': theta_samples,
        'w_samples': w_samples,
        'kappa': kappa,
        'diagnostics': {
            'acceptance_rate': float(jnp.mean(acceptance_rates)),
            'acceptance_rate_per_chain': jnp.mean(acceptance_rates, axis=0),
        }
    }


def sample_posterior_corrected(key, X, Z, sig, alpha, indices, rho_dest=0.1,
                                num_samples=1000, num_warmup=500, num_chains=4):
    """
    Sample posterior with automatic κ correction from indices.
    
    This is the main entry point for corrected inference.
    
    Args:
        key: JAX random key
        X: Triplet data (n, 3, p)
        Z: Normalized embeddings (n, 3, p)
        sig: Noise std
        alpha: Dirichlet prior
        indices: (n, 3) triplet indices for κ computation
        rho_dest: Destination correlation (default 0.1)
        
    Returns:
        Dictionary with samples and correction info
    """
    kappa, design_info = compute_kappa(indices, rho_dest=rho_dest)
    
    result = sample_posterior_nuts_power(
        key, X, Z, sig, alpha, kappa=kappa,
        num_samples=num_samples, num_warmup=num_warmup, num_chains=num_chains
    )
    
    result['design_info'] = design_info
    return result


# =============================================================================
# DESIGN RECOMMENDATIONS
# =============================================================================

def recommend_triplet_design(n_points, target_n_eff=None, max_triplets=None, rho_dest=0.1):
    """
    Recommend triplet design given constraints.
    
    Args:
        n_points: Total available points
        target_n_eff: Desired effective sample size
        max_triplets: Maximum computational budget
        rho_dest: Destination correlation (default 0.1)
        
    Returns:
        Dictionary with recommended design parameters
    """
    # Optimal: 1 anchor, rest as destinations (since ρ_anchor = 0)
    n_destinations = n_points - 1
    
    # Ceiling on n_eff
    n_eff_ceiling = n_destinations / (2 * rho_dest)
    
    result = {
        'n_anchors': 1,
        'n_destinations': n_destinations,
        'n_eff_ceiling': n_eff_ceiling,
    }
    
    if target_n_eff is not None:
        if target_n_eff >= n_eff_ceiling:
            result['warning'] = f"Target exceeds ceiling {n_eff_ceiling:.0f}"
            k_dest = n_destinations - 1
        else:
            # Solve for k: n_eff = (n_d * k / 2) / (1 + 0.2*(k-1))
            denom = n_destinations / 2 - 2 * rho_dest * target_n_eff
            if denom > 0:
                k_opt = (1 - 2 * rho_dest) * target_n_eff / denom
                k_dest = max(1, int(np.ceil(k_opt)))
            else:
                k_dest = n_destinations - 1
        result['k_dest'] = k_dest
        
    elif max_triplets is not None:
        # T = n_d * k / 2 => k = 2T / n_d
        k_dest = max(1, int(2 * max_triplets / n_destinations))
        result['k_dest'] = min(k_dest, n_destinations - 1)
        
    else:
        # Default: k=5 as good tradeoff
        result['k_dest'] = min(5, n_destinations - 1)
    
    # Compute design statistics
    k = result['k_dest']
    kappa = compute_kappa_balanced(k, rho_dest)
    n_triplets = n_destinations * k // 2
    n_eff = n_triplets * kappa
    
    result.update({
        'kappa': kappa,
        'n_triplets': n_triplets,
        'n_eff': n_eff,
        'efficiency': kappa,
    })
    
    return result
