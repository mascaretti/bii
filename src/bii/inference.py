import jax
from jax import numpy as jnp
from jax import random
from bii.data import T_from_X
import optax
from jax.scipy.special import log_ndtr
import blackjax

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
    logP = log_ndtr(-s)   # log Phi(-s)
    log1mP = log_ndtr(s)  # log (1 - Phi(-s))
    return logP, log1mP

@jax.jit
def loglik_theta(theta, T, Z, sig):
    sig2 = jnp.square(sig)
    w = jax.nn.softmax(theta)  # simplex
    zi, zj, zk = Z[:, 1], Z[:, 2], Z[:, 0]

    # vectorized delta/V over triplets
    def dv(zi, zj, zk):
        return delta_V_one_triplet(zi, zj, zk, w, sig2)
    delta, V = jax.vmap(dv)(zi, zj, zk)

    logP, log1mP = logP_log1mP_from_deltaV(delta, V)
    return jnp.sum(T * logP + (1.0 - T) * log1mP)

def fit(key, X, Z, sig, steps=5000, lr=1e-2):
    key_tr, _ = jax.random.split(key)
    n = X.shape[0]
    p = X.shape[2]
    T = T_from_X(X)
    theta = jnp.zeros((p,))   # uniform init
    opt = optax.adam(lr)
    opt_state = opt.init(theta)
    
    def neg_ll(th):
        return -loglik_theta(th, T, Z, sig)
    
    valgrad = jax.jit(jax.value_and_grad(neg_ll))
    
    @jax.jit
    def step(theta, opt_state):
        loss, grads = valgrad(theta)
        updates, opt_state = opt.update(grads, opt_state, theta)
        theta = optax.apply_updates(theta, updates)
        return theta, opt_state, loss
    
    # Store optimization trajectory
    theta_history = [theta]
    loss_history = []
    
    for _ in range(steps):
        theta, opt_state, loss = step(theta, opt_state)
        theta_history.append(theta)
        loss_history.append(loss)
    
    # Convert to arrays
    theta_history = jnp.stack(theta_history)  # (steps+1, p)
    loss_history = jnp.array(loss_history)    # (steps,)
    
    return jax.nn.softmax(theta), theta_history, loss_history


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
    
    # Run warmup and sampling for each chain separately
    keys_warmup = random.split(key, num_chains)
    
    all_theta_samples = []
    all_acceptance_rates = []
    
    for chain_idx in range(num_chains):
        # Run warmup for this chain
        (state, params), warmup_info = warmup.run(keys_warmup[chain_idx], init_positions[chain_idx])
        
        # Build kernel for this chain
        kernel = blackjax.nuts(logprob_fn, **params)
        
        # Sampling loop for this chain
        def one_step(state, key):
            new_state, info = kernel.step(key, state)
            return new_state, (new_state.position, info.acceptance_rate)
        
        # Generate sampling keys for this chain
        key, subkey = random.split(key)
        sample_keys = random.split(subkey, num_samples)
        
        # Run sampling
        final_state, (theta_samples_chain, acceptance_rates_chain) = jax.lax.scan(
            one_step, state, sample_keys
        )
        
        all_theta_samples.append(theta_samples_chain)
        all_acceptance_rates.append(acceptance_rates_chain)
    
    # Stack results: (num_samples, num_chains, p)
    theta_samples = jnp.stack(all_theta_samples, axis=1)
    acceptance_rates = jnp.stack(all_acceptance_rates, axis=1)
    
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
        Dictionary with posterior mean, std, quantiles, and MAP estimate
    """
    # Flatten chains if present
    if w_samples.ndim == 3:
        w_flat = w_samples.reshape(-1, w_samples.shape[-1])
    else:
        w_flat = w_samples
    
    # Compute MAP as the sample with highest posterior density
    # Approximate using kernel density or just use the mode of each component
    # For simplicity, we'll use the median (for unimodal posteriors, close to MAP)
    # A better approach: find the sample closest to the mode
    
    return {
        'mean': jnp.mean(w_flat, axis=0),
        'std': jnp.std(w_flat, axis=0),
        'median': jnp.median(w_flat, axis=0),
        'map': jnp.median(w_flat, axis=0),  # Approximation for unimodal posteriors
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
