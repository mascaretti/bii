"""
Inference with Power Likelihood Correction for Correlated Triplets

The power likelihood adjusts for correlations induced by destination reuse:

    π_κ(w | Y) ∝ π(w) · [∏_t p(Y_t | w)]^κ

where κ = 1/λ and λ = 1 + ρ_dest · Σ_i T_i(T_i-1) / T

This preserves consistency while giving calibrated posteriors.
"""

import jax
from jax import numpy as jnp
from jax import random
from jax.scipy.special import log_ndtr
import optax
import blackjax


# =============================================================================
# CORE LIKELIHOOD FUNCTIONS
# =============================================================================

@jax.jit
def delta_V_one_triplet(zi, zj, zk, w, sig2):
    """
    Compute Δ and V for one triplet.
    
    Δ = E[d(Z_i, Z_k)^2 - d(Z_j, Z_k)^2 | w]
    V = Var[d(Z_i, Z_k)^2 - d(Z_j, Z_k)^2 | w]
    
    The triplet indicator T = 1{d(Z_i, Z_k) < d(Z_j, Z_k)} has
    P(T=1) = Φ(-Δ/√V) under the probit approximation.
    """
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
    """Compute log P(T=1) and log P(T=0) from Δ and V."""
    s = delta / jnp.sqrt(V + 1e-12)
    logP = log_ndtr(-s)   # log Φ(-s)
    log1mP = log_ndtr(s)  # log (1 - Φ(-s))
    return logP, log1mP


@jax.jit
def loglik_theta(theta, T, Z, sig):
    """
    Compute log-likelihood (without power correction).
    
    Args:
        theta: Unconstrained parameters (p,)
        T: Triplet outcomes (n_triplets,) in {0, 1}
        Z: Normalized embeddings (n_triplets, 3, p)
        sig: Noise std (p,) or scalar
        
    Returns:
        Log-likelihood (scalar)
    """
    sig2 = jnp.square(sig)
    w = jax.nn.softmax(theta)  # Map to simplex
    zi, zj, zk = Z[:, 1], Z[:, 2], Z[:, 0]

    def dv(zi, zj, zk):
        return delta_V_one_triplet(zi, zj, zk, w, sig2)
    
    delta, V = jax.vmap(dv)(zi, zj, zk)
    logP, log1mP = logP_log1mP_from_deltaV(delta, V)
    
    return jnp.sum(T * logP + (1.0 - T) * log1mP)


@jax.jit
def loglik_theta_power(theta, T, Z, sig, kappa):
    """
    Compute power log-likelihood: κ · ℓ(θ).
    
    This is the corrected likelihood that accounts for correlations.
    
    Args:
        theta: Unconstrained parameters (p,)
        T: Triplet outcomes (n_triplets,)
        Z: Normalized embeddings (n_triplets, 3, p)
        sig: Noise std
        kappa: Power correction factor in (0, 1]
        
    Returns:
        Power log-likelihood: κ · log p(T | θ)
    """
    return kappa * loglik_theta(theta, T, Z, sig)


# =============================================================================
# PRIOR AND POSTERIOR
# =============================================================================

@jax.jit
def log_prior_dirichlet(theta, alpha):
    """
    Log prior for Dirichlet(α) on w = softmax(θ).
    
    For uniform prior on simplex, use α = (1, ..., 1).
    """
    w = jax.nn.softmax(theta)
    log_prior = jnp.sum((alpha - 1.0) * jnp.log(w + 1e-12))
    return log_prior


@jax.jit
def log_posterior(theta, T, Z, sig, alpha, kappa=1.0):
    """
    Log posterior with power likelihood correction.
    
    log π_κ(θ | T) ∝ κ · ℓ(θ) + log π(θ)
    
    Args:
        theta: Parameters
        T: Triplet outcomes  
        Z: Embeddings
        sig: Noise std
        alpha: Dirichlet concentration
        kappa: Power correction (default 1.0 = no correction)
    """
    log_lik = loglik_theta_power(theta, T, Z, sig, kappa)
    log_pri = log_prior_dirichlet(theta, alpha)
    return log_lik + log_pri


# =============================================================================
# POINT ESTIMATION (MAP / MLE)
# =============================================================================

def fit(key, X, Z, sig, kappa=1.0, steps=5000, lr=1e-2):
    """
    Find MAP estimate with power likelihood correction.
    
    Args:
        key: JAX random key
        X: Triplet data in original space (n, 3, p)
        Z: Normalized embeddings (n, 3, p)
        sig: Noise std
        kappa: Power correction factor (default 1.0)
        steps: Optimization steps
        lr: Learning rate
        
    Returns:
        w_hat: MAP estimate on simplex (p,)
        theta_history: Optimization trajectory
        loss_history: Loss values
    """
    from bii.data import T_from_X
    
    T = T_from_X(X)
    p = X.shape[2]
    theta = jnp.zeros((p,))
    
    opt = optax.adam(lr)
    opt_state = opt.init(theta)
    
    def neg_ll(th):
        return -loglik_theta_power(th, T, Z, sig, kappa)
    
    valgrad = jax.jit(jax.value_and_grad(neg_ll))
    
    @jax.jit
    def step(theta, opt_state):
        loss, grads = valgrad(theta)
        updates, opt_state = opt.update(grads, opt_state, theta)
        theta = optax.apply_updates(theta, updates)
        return theta, opt_state, loss
    
    theta_history = [theta]
    loss_history = []
    
    for _ in range(steps):
        theta, opt_state, loss = step(theta, opt_state)
        theta_history.append(theta)
        loss_history.append(loss)
    
    theta_history = jnp.stack(theta_history)
    loss_history = jnp.array(loss_history)
    
    return jax.nn.softmax(theta), theta_history, loss_history


def fit_with_correction(key, X, Z, sig, indices, rho_dest=0.1, steps=5000, lr=1e-2):
    """
    Fit with automatic κ computation from triplet indices.
    
    Args:
        key: JAX random key
        X: Triplet data (n, 3, p)
        Z: Normalized embeddings (n, 3, p)
        sig: Noise std
        indices: (n, 3) triplet indices for κ computation
        rho_dest: Destination correlation (default 0.1)
        
    Returns:
        w_hat: MAP estimate
        kappa: Computed correction factor
        info: Design diagnostics
    """
    from triplet_design import compute_kappa_from_indices
    
    kappa, info = compute_kappa_from_indices(indices, rho_dest=rho_dest)
    w_hat, theta_hist, loss_hist = fit(key, X, Z, sig, kappa=kappa, steps=steps, lr=lr)
    
    return w_hat, kappa, info, theta_hist, loss_hist


# =============================================================================
# MCMC SAMPLING
# =============================================================================

def sample_posterior_nuts(key, X, Z, sig, alpha, kappa=1.0,
                          num_samples=1000, num_warmup=500, num_chains=4,
                          init_theta=None, step_size=1e-3,
                          target_acceptance_rate=0.8):
    """
    Sample from power posterior using NUTS.
    
    π_κ(θ | T) ∝ π(θ) · [p(T | θ)]^κ
    
    Args:
        key: JAX random key
        X: Triplet data (n, 3, p)
        Z: Normalized embeddings (n, 3, p)  
        sig: Noise std
        alpha: Dirichlet prior concentration (p,)
        kappa: Power correction factor (default 1.0)
        num_samples: Samples per chain
        num_warmup: Warmup iterations
        num_chains: Number of chains
        
    Returns:
        Dictionary with theta_samples, w_samples, diagnostics
    """
    from bii.data import T_from_X
    
    T = T_from_X(X)
    p = Z.shape[2]
    
    def logprob_fn(theta):
        return log_posterior(theta, T, Z, sig, alpha, kappa)
    
    if init_theta is None:
        init_theta = jnp.zeros(p)
    
    # Initialize chains with small perturbations
    key, *init_keys = random.split(key, num_chains + 1)
    init_positions = jnp.stack([
        init_theta + 0.1 * random.normal(k, shape=(p,))
        for k in init_keys
    ])
    
    # Build NUTS with window adaptation
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
            'acceptance_rate': jnp.mean(acceptance_rates),
            'acceptance_rate_per_chain': jnp.mean(acceptance_rates, axis=0),
        }
    }


def sample_with_correction(key, X, Z, sig, alpha, indices, rho_dest=0.1,
                           num_samples=1000, num_warmup=500, num_chains=4):
    """
    Sample posterior with automatic κ correction.
    
    Args:
        indices: (n, 3) triplet indices for κ computation
        
    Returns:
        Posterior samples with calibrated uncertainty
    """
    from triplet_design import compute_kappa_from_indices
    
    kappa, design_info = compute_kappa_from_indices(indices, rho_dest=rho_dest)
    
    result = sample_posterior_nuts(
        key, X, Z, sig, alpha, kappa=kappa,
        num_samples=num_samples, num_warmup=num_warmup, num_chains=num_chains
    )
    
    result['design_info'] = design_info
    return result


# =============================================================================
# POSTERIOR DIAGNOSTICS
# =============================================================================

def compute_coverage(w_samples, w_true, levels=(0.5, 0.9, 0.95)):
    """
    Check if true parameters fall within credible intervals.
    
    Args:
        w_samples: (n_samples, p) posterior samples
        w_true: (p,) true parameters
        levels: Credible levels to check
        
    Returns:
        Dictionary with coverage for each level and each parameter
    """
    results = {}
    
    for level in levels:
        alpha = 1 - level
        lower = jnp.quantile(w_samples, alpha/2, axis=0)
        upper = jnp.quantile(w_samples, 1 - alpha/2, axis=0)
        
        covered = (w_true >= lower) & (w_true <= upper)
        
        results[level] = {
            'covered': covered,
            'fraction_covered': jnp.mean(covered),
            'lower': lower,
            'upper': upper,
        }
    
    return results


def coverage_simulation(key, make_data_fn, fit_fn, w_true, n_reps=100, 
                        level=0.95, use_correction=True, rho_dest=0.1):
    """
    Simulate coverage to validate calibration.
    
    For calibrated posteriors, the (1-α) credible interval should contain
    the true parameter with probability (1-α).
    
    Args:
        make_data_fn: Function (key) -> (X, Z, indices)
        fit_fn: Function (key, X, Z, indices, kappa) -> w_samples
        w_true: True parameter
        n_reps: Number of simulation replicates
        level: Credible level
        use_correction: Whether to apply κ correction
        
    Returns:
        Empirical coverage (should be close to `level` if calibrated)
    """
    from triplet_design import compute_kappa_from_indices
    
    covered_count = 0
    
    for i in range(n_reps):
        key, k_data, k_fit = random.split(key, 3)
        
        X, Z, indices = make_data_fn(k_data)
        
        if use_correction:
            kappa, _ = compute_kappa_from_indices(indices, rho_dest=rho_dest)
        else:
            kappa = 1.0
        
        w_samples = fit_fn(k_fit, X, Z, indices, kappa)
        
        # Check coverage for each parameter
        alpha = 1 - level
        lower = jnp.quantile(w_samples, alpha/2, axis=0)
        upper = jnp.quantile(w_samples, 1 - alpha/2, axis=0)
        
        all_covered = jnp.all((w_true >= lower) & (w_true <= upper))
        covered_count += int(all_covered)
    
    return covered_count / n_reps


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import jax.random as jr
    
    print("=" * 60)
    print("Testing power likelihood inference")
    print("=" * 60)
    
    key = jr.key(42)
    p = 5
    w_true = jnp.array([0.4, 0.3, 0.15, 0.1, 0.05])
    sig = 0.5 * jnp.ones(p)
    tau = 1.0
    
    # Import triplet design
    from triplet_design import make_single_anchor_triplets
    
    # Generate data with different k_dest
    print("\nComparing MLE with and without κ correction:")
    print("-" * 60)
    
    for k_dest in [1, 5, 20]:
        key, k_data, k_fit = jr.split(key, 3)
        
        X, Z, indices, kappa, info = make_single_anchor_triplets(
            k_data, n_destinations=100, k_dest=k_dest,
            p=p, sig=sig, tau=tau, w0=w_true
        )
        
        # Fit without correction
        w_hat_uncorr, _, _ = fit(k_fit, X, Z, sig, kappa=1.0)
        
        # Fit with correction
        w_hat_corr, _, _ = fit(k_fit, X, Z, sig, kappa=kappa)
        
        # L1 errors
        err_uncorr = jnp.sum(jnp.abs(w_hat_uncorr - w_true))
        err_corr = jnp.sum(jnp.abs(w_hat_corr - w_true))
        
        print(f"\nk_dest={k_dest}: T={info['n_triplets']}, κ={kappa:.3f}, n_eff={info['n_eff']:.1f}")
        print(f"  L1 error (uncorrected): {err_uncorr:.4f}")
        print(f"  L1 error (corrected):   {err_corr:.4f}")
        print(f"  Note: Point estimates similar; κ mainly affects posterior width")
