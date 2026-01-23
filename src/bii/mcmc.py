"""
Posterior sampling for metric alignment using BlackJAX NUTS sampler.

This module implements Bayesian inference with a Dirichlet prior on the weights.
"""

import jax
from jax import numpy as jnp
from jax import random
import blackjax


def T_from_X(X):
    """Convert triplet data to binary comparisons."""
    xi, xj, xk = X[:, 1], X[:, 2], X[:, 0]
    di = jnp.sum((xi - xk)**2, axis=1)
    dj = jnp.sum((xj - xk)**2, axis=1)
    return (di <= dj).astype(jnp.float32)


# Import likelihood functions from your existing code
from jax.scipy.special import log_ndtr


@jax.jit
def delta_V_one_triplet(zi, zj, zk, w, sig2):
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
def logP_log1mP_from_deltaV(delta, V):
    s = delta / jnp.sqrt(V + 1e-12)
    logP = log_ndtr(-s)
    log1mP = log_ndtr(s)
    return logP, log1mP


@jax.jit
def loglik_theta(theta, T, Z, sig):
    """Log likelihood of data given parameters."""
    sig2 = jnp.square(sig)
    w = jax.nn.softmax(theta)
    zi, zj, zk = Z[:, 1], Z[:, 2], Z[:, 0]

    def dv(zi, zj, zk):
        return delta_V_one_triplet(zi, zj, zk, w, sig2)
    delta, V = jax.vmap(dv)(zi, zj, zk)

    logP, log1mP = logP_log1mP_from_deltaV(delta, V)
    return jnp.sum(T * logP + (1.0 - T) * log1mP)


@jax.jit
def log_prior_dirichlet(theta, alpha):
    """
    Log prior for Dirichlet(alpha) on w = softmax(theta).
    
    For uniform prior, use alpha = (1, ..., 1).
    This gives p(w) ∝ 1 on the simplex, so the MAP equals the MLE.
    """
    w = jax.nn.softmax(theta)
    # Dirichlet density: p(w) ∝ ∏ w_i^(α_i - 1)
    log_prior = jnp.sum((alpha - 1.0) * jnp.log(w + 1e-12))
    return log_prior


@jax.jit
def log_posterior(theta, T, Z, sig, alpha):
    """
    Log posterior: log p(theta | data) ∝ log p(data | theta) + log p(theta).
    """
    log_lik = loglik_theta(theta, T, Z, sig)
    log_pri = log_prior_dirichlet(theta, alpha)
    return log_lik + log_pri


def sample_posterior_nuts(key, X, Z, sig, alpha, 
                          num_samples=1000, 
                          num_warmup=500,
                          num_chains=4,
                          init_theta=None,
                          step_size=1e-3,
                          target_acceptance_rate=0.8):
    """
    Sample from posterior using BlackJAX NUTS sampler.
    
    Args:
        key: JAX random key
        X: Triplet data (n, 3, p) - original embeddings
        Z: Normalized embeddings (n, 3, p)
        sig: Noise std (scalar or array of shape (p,))
        alpha: Dirichlet prior concentration (p,)
               Use alpha = ones(p) for uniform prior
        num_samples: Number of posterior samples per chain
        num_warmup: Number of warmup iterations
        num_chains: Number of parallel MCMC chains
        init_theta: Initial theta (p,) or None for zero initialization
        step_size: Initial step size for HMC
        target_acceptance_rate: Target acceptance rate for adaptation
    
    Returns:
        Dictionary with:
            - 'theta_samples': (num_samples, num_chains, p)
            - 'w_samples': (num_samples, num_chains, p) - on simplex
            - 'diagnostics': dict with acceptance rates, ESS, etc.
    """
    T = T_from_X(X)
    p = Z.shape[2]
    
    # Define log posterior function for sampling
    def logprob_fn(theta):
        return log_posterior(theta, T, Z, sig, alpha)
    
    # Initialize theta for each chain
    if init_theta is None:
        init_theta = jnp.zeros(p)  # Corresponds to uniform w
    
    # Initialize chains with small perturbations
    key, *init_keys = random.split(key, num_chains + 1)
    init_positions = jnp.stack([
        init_theta + 0.1 * random.normal(k, shape=(p,))
        for k in init_keys
    ])
    
    # Build NUTS kernel with window adaptation
    warmup = blackjax.window_adaptation(
        blackjax.nuts,
        logprob_fn,
        num_warmup,
        target_acceptance_rate=target_acceptance_rate,
        initial_step_size=step_size,
    )
    
    # Run warmup for each chain
    keys_warmup = random.split(key, num_chains)
    
    def run_warmup_single(init_pos, warmup_key):
        (state, params), info = warmup.run(warmup_key, init_pos)
        return state, params
    
    # Vectorize warmup over chains
    states, params_list = jax.vmap(run_warmup_single)(init_positions, keys_warmup)
    
    # Build sampling kernel for each chain
    def make_kernel(params):
        return blackjax.nuts(logprob_fn, **params)
    
    kernels = jax.vmap(make_kernel)(params_list)
    
    # Sampling loop
    def one_step(states, key):
        keys = random.split(key, num_chains)
        
        def step_single(state, kernel, step_key):
            new_state, info = kernel.step(step_key, state)
            return new_state, (new_state.position, info.acceptance_rate)
        
        new_states, (positions, accept_rates) = jax.vmap(step_single)(
            states, kernels, keys
        )
        return new_states, (positions, accept_rates)
    
    # Generate sampling keys
    key, subkey = random.split(key)
    sample_keys = random.split(subkey, num_samples)
    
    # Run sampling
    final_states, (theta_samples, acceptance_rates) = jax.lax.scan(
        one_step, states, sample_keys
    )
    
    # theta_samples is (num_samples, num_chains, p)
    # Convert to w samples
    w_samples = jax.vmap(jax.vmap(jax.nn.softmax))(theta_samples)
    
    # Compute diagnostics
    mean_acceptance = jnp.mean(acceptance_rates)
    acceptance_per_chain = jnp.mean(acceptance_rates, axis=0)
    
    diagnostics = {
        'acceptance_rate': mean_acceptance,
        'acceptance_rate_per_chain': acceptance_per_chain,
    }
    
    return {
        'theta_samples': theta_samples,
        'w_samples': w_samples,
        'diagnostics': diagnostics
    }


def compute_posterior_statistics(w_samples):
    """
    Compute summary statistics from posterior samples.
    
    Args:
        w_samples: (num_samples, num_chains, p) or (num_samples, p)
    
    Returns:
        Dictionary with posterior mean, std, quantiles
    """
    # Flatten chains if present
    if w_samples.ndim == 3:
        w_flat = w_samples.reshape(-1, w_samples.shape[-1])
    else:
        w_flat = w_samples
    
    return {
        'mean': jnp.mean(w_flat, axis=0),
        'std': jnp.std(w_flat, axis=0),
        'median': jnp.median(w_flat, axis=0),
        'q025': jnp.quantile(w_flat, 0.025, axis=0),
        'q975': jnp.quantile(w_flat, 0.975, axis=0),
        'q05': jnp.quantile(w_flat, 0.05, axis=0),
        'q95': jnp.quantile(w_flat, 0.95, axis=0),
    }


def compute_rhat(samples):
    """
    Compute Gelman-Rubin R-hat convergence diagnostic.
    
    Args:
        samples: (num_samples, num_chains, p)
    
    Returns:
        R-hat values for each parameter (p,)
        
    Values < 1.01 indicate convergence.
    """
    num_samples, num_chains, p = samples.shape
    
    # Between-chain variance
    chain_means = jnp.mean(samples, axis=0)  # (num_chains, p)
    global_mean = jnp.mean(chain_means, axis=0)  # (p,)
    B = num_samples / (num_chains - 1) * jnp.sum(
        (chain_means - global_mean[None, :])**2, axis=0
    )
    
    # Within-chain variance
    chain_vars = jnp.var(samples, axis=0, ddof=1)  # (num_chains, p)
    W = jnp.mean(chain_vars, axis=0)  # (p,)
    
    # Pooled variance estimate
    var_plus = ((num_samples - 1) / num_samples) * W + (1 / num_samples) * B
    
    # R-hat
    rhat = jnp.sqrt(var_plus / (W + 1e-10))
    
    return rhat


def compute_ess(samples):
    """
    Compute effective sample size (ESS) using autocorrelation.
    
    Args:
        samples: (num_samples, num_chains, p)
    
    Returns:
        ESS for each parameter (p,)
    """
    # Flatten chains
    num_samples, num_chains, p = samples.shape
    total_samples = num_samples * num_chains
    
    # Simple ESS estimate: divide by average autocorrelation at lag 1
    # For more accurate ESS, could use arviz or similar
    w_flat = samples.reshape(-1, p)
    
    # Autocorrelation at lag 1
    acf1 = jnp.array([
        jnp.corrcoef(w_flat[:-1, i], w_flat[1:, i])[0, 1]
        for i in range(p)
    ])
    
    # ESS approximation
    ess = total_samples / (1 + 2 * jnp.maximum(acf1, 0))
    
    return ess
