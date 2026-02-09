"""
Optimal Triplet Design for Bayesian Metric Learning

Theory summary:
- ρ_anchor = 0: Triplets sharing only an anchor are uncorrelated (for iid continuous data)
- ρ_dest ≈ 0.1: Triplets sharing a destination have correlation ~0.1 (for Gaussians)

Design principle:
- Reuse anchors freely (no correlation penalty)
- Control destination reuse to manage correlations
- Apply power likelihood correction κ = 1/λ where λ is variance inflation
"""

import jax
from jax import numpy as jnp
from jax import random
from collections import Counter
import numpy as np


# =============================================================================
# TRIPLET CONSTRUCTION
# =============================================================================

def make_optimal_triplets(key, n_anchors, n_destinations, k_dest, p, sig, tau, w0,
                          on_source=False, return_indices=True, return_pool=False):
    """
    Construct triplets with optimal correlation structure.
    
    Design:
    - n_anchors: Number of distinct anchors (can be small, even 1)
    - n_destinations: Number of distinct destinations  
    - k_dest: Max times each destination appears across all triplets
    
    Since ρ_anchor = 0, we reuse each anchor as many times as needed.
    Since ρ_dest ≈ 0.1, we control destination reuse via k_dest.
    
    Total triplets: Approximately (n_destinations * k_dest) / 2
    (Each triplet uses 2 destinations; each destination used k_dest times)
    
    Args:
        key: JAX random key
        n_anchors: Number of anchor points (can be 1)
        n_destinations: Number of destination points
        k_dest: Maximum times each destination is reused
        p: Dimension of embeddings
        sig: Noise std (scalar or array of shape (p,))
        tau: Signal std for data generation
        w0: True weights (p,) for normalization
        on_source: If True, compute distances on noisy Y; else on clean X
        return_indices: If True, return triplet indices
        return_pool: If True, return the full point pools
        
    Returns:
        X: Triplet data (n_triplets, 3, p) in original space
        Z: Normalized embeddings (n_triplets, 3, p)
        indices: (n_triplets, 3) with [anchor_idx, dest1_idx, dest2_idx]
        kappa: Power likelihood correction factor
        design_info: Dictionary with design diagnostics
    """
    pool_size = n_anchors + n_destinations
    
    keys = random.split(key, 5)
    key_x, key_eps, key_anchor, key_shuffle, key_pairs = keys[0], keys[1], keys[2], keys[3], keys[4]
    
    # Generate point cloud
    X_pool = random.multivariate_normal(
        key=key_x, mean=jnp.zeros(p),
        cov=jnp.square(tau) * jnp.eye(p), shape=(pool_size,)
    )
    
    # Handle sig as scalar or array
    if jnp.ndim(sig) == 0:
        sig_diag = jnp.square(sig) * jnp.ones(p)
    else:
        sig_diag = jnp.square(sig)
    
    epsilon = random.multivariate_normal(
        key=key_eps, mean=jnp.zeros(p),
        cov=jnp.diag(sig_diag), shape=(pool_size,)
    )
    Y_pool = X_pool + epsilon
    Z_pool = Y_pool / jnp.sqrt(w0)[None, :]
    
    # Partition into anchors and destinations
    all_indices = jnp.arange(pool_size)
    anchor_indices = random.choice(key_anchor, all_indices, shape=(n_anchors,), replace=False)
    dest_indices = jnp.setdiff1d(all_indices, anchor_indices)
    
    # Distance computation basis
    data = Y_pool if on_source else X_pool
    
    # Build triplets with controlled destination reuse
    # Strategy: for each anchor, pair destinations based on distance ranking
    # Rotate through anchors to spread triplets evenly
    
    T = []
    P = []
    T_indices = []
    
    # Track destination usage
    dest_usage = {int(d): 0 for d in dest_indices}
    
    # Compute all anchor-destination distances
    anchor_dest_dists = {}
    for a_idx in anchor_indices:
        a_idx = int(a_idx)
        anchor_point = data[a_idx]
        dists = jnp.sum((data[dest_indices] - anchor_point[None, :]) ** 2, axis=1)
        # Store sorted destination indices (by distance to this anchor)
        sorted_positions = jnp.argsort(dists)
        anchor_dest_dists[a_idx] = [int(dest_indices[pos]) for pos in sorted_positions]
    
    # Maximum possible triplets given k_dest constraint
    # Each triplet uses 2 destination slots; total slots = n_destinations * k_dest
    max_triplets = (n_destinations * k_dest) // 2
    
    # Generate triplets by cycling through anchors
    anchor_list = [int(a) for a in anchor_indices]
    triplet_count = 0
    anchor_cycle = 0
    
    while triplet_count < max_triplets:
        # Pick anchor (cycle through)
        anchor_idx = anchor_list[anchor_cycle % n_anchors]
        anchor_cycle += 1
        
        # Find two destinations that haven't hit k_dest limit
        sorted_dests = anchor_dest_dists[anchor_idx]
        
        dest1_idx = None
        dest2_idx = None
        
        for d_idx in sorted_dests:
            if dest_usage[d_idx] < k_dest:
                if dest1_idx is None:
                    dest1_idx = d_idx
                elif dest2_idx is None and d_idx != dest1_idx:
                    dest2_idx = d_idx
                    break
        
        # If we can't find two available destinations, we're done
        if dest1_idx is None or dest2_idx is None:
            break
        
        # Record triplet
        dest_usage[dest1_idx] += 1
        dest_usage[dest2_idx] += 1
        
        T.append([X_pool[anchor_idx], X_pool[dest1_idx], X_pool[dest2_idx]])
        P.append([Z_pool[anchor_idx], Z_pool[dest1_idx], Z_pool[dest2_idx]])
        T_indices.append([anchor_idx, dest1_idx, dest2_idx])
        
        triplet_count += 1
    
    T = jnp.asarray(T)
    P = jnp.asarray(P)
    T_indices = jnp.asarray(T_indices)
    
    # Compute kappa and design info
    kappa, design_info = compute_kappa_from_indices(T_indices, rho_dest=0.1)
    
    design_info.update({
        'n_anchors': n_anchors,
        'n_destinations': n_destinations,
        'k_dest_target': k_dest,
        'max_possible_triplets': max_triplets,
    })
    
    results = [T, P]
    if return_indices:
        results.extend([T_indices, kappa, design_info])
    if return_pool:
        results.extend([X_pool, Z_pool])
    
    return tuple(results)


def make_single_anchor_triplets(key, n_destinations, k_dest, p, sig, tau, w0,
                                 on_source=False, return_indices=True):
    """
    Simplified interface: single anchor, many destinations.
    
    This is the theoretically optimal design:
    - 1 anchor (ρ_anchor = 0 means no penalty)
    - n_destinations destinations, each used at most k_dest times
    - Total triplets ≈ n_destinations * k_dest / 2
    
    Args:
        n_destinations: Number of destination points
        k_dest: Max reuse per destination (k_dest=1 gives independent triplets)
        
    For k_dest=1: T = n_destinations/2 independent triplets
    For k_dest=n_destinations-1: T = n_destinations*(n_destinations-1)/2 triplets (all pairs)
    """
    return make_optimal_triplets(
        key, n_anchors=1, n_destinations=n_destinations, k_dest=k_dest,
        p=p, sig=sig, tau=tau, w0=w0, on_source=on_source, 
        return_indices=return_indices
    )


# =============================================================================
# KAPPA COMPUTATION
# =============================================================================

def compute_kappa_from_indices(indices, rho_dest=0.1, rho_anchor=0.0):
    """
    Compute power likelihood correction factor κ from triplet indices.
    
    κ = 1 / λ where λ is the variance inflation factor.
    
    λ = 1 + ρ_dest * Σ_i T_i(T_i - 1) / T
    
    where T_i is the number of triplets containing destination i.
    
    Args:
        indices: (n_triplets, 3) array with [anchor, dest1, dest2] per row
        rho_dest: Correlation between triplets sharing a destination (~0.1 for Gaussian)
        rho_anchor: Correlation between triplets sharing only anchor (0 for iid)
        
    Returns:
        kappa: Correction factor in (0, 1]
        info: Dictionary with diagnostic information
    """
    indices = np.asarray(indices)
    n_triplets = len(indices)
    
    if n_triplets == 0:
        return 1.0, {'lambda': 1.0, 'n_eff': 0}
    
    # Count destination usage
    # Destinations are in columns 1 and 2
    destinations = indices[:, 1:].flatten()
    dest_counts = Counter(destinations)
    
    # Count anchor usage (for diagnostics, though ρ_anchor = 0)
    anchors = indices[:, 0]
    anchor_counts = Counter(anchors)
    
    # Compute Σ_i T_i(T_i - 1) for destinations
    dest_overlap_sum = sum(t * (t - 1) for t in dest_counts.values())
    
    # Compute Σ_a T_a(T_a - 1) for anchors
    anchor_overlap_sum = sum(t * (t - 1) for t in anchor_counts.values())
    
    # Variance inflation factor
    # Note: we divide by T (not 2T) because each pair of triplets sharing
    # a destination contributes once to the covariance sum
    lambda_dest = rho_dest * dest_overlap_sum / n_triplets if n_triplets > 0 else 0
    lambda_anchor = rho_anchor * anchor_overlap_sum / n_triplets if n_triplets > 0 else 0
    
    lambda_total = 1 + lambda_dest + lambda_anchor
    
    kappa = 1 / lambda_total
    n_eff = n_triplets * kappa
    
    # Destination reuse statistics
    dest_T = np.array(list(dest_counts.values()))
    anchor_T = np.array(list(anchor_counts.values()))
    
    info = {
        'lambda': lambda_total,
        'lambda_dest': lambda_dest,
        'lambda_anchor': lambda_anchor,
        'n_eff': n_eff,
        'n_triplets': n_triplets,
        'efficiency': kappa,  # n_eff / n_triplets
        'n_unique_destinations': len(dest_counts),
        'n_unique_anchors': len(anchor_counts),
        'mean_dest_reuse': np.mean(dest_T),
        'max_dest_reuse': np.max(dest_T),
        'mean_anchor_reuse': np.mean(anchor_T),
        'max_anchor_reuse': np.max(anchor_T),
    }
    
    return float(kappa), info


def compute_kappa_balanced(n_destinations, k_dest, rho_dest=0.1):
    """
    Compute κ for a balanced design where each destination appears exactly k times.
    
    In a balanced design:
    - T = n_destinations * k_dest / 2 triplets
    - Each destination appears in exactly k_dest triplets
    - Σ_i T_i(T_i - 1) = n_destinations * k_dest * (k_dest - 1)
    
    λ = 1 + ρ_dest * n_destinations * k_dest * (k_dest - 1) / T
      = 1 + ρ_dest * 2 * (k_dest - 1)
      = 1 + 2 * ρ_dest * (k_dest - 1)
    
    For ρ_dest = 0.1:
    - k=1: λ = 1.0, κ = 1.0 (independent)
    - k=2: λ = 1.2, κ = 0.83
    - k=5: λ = 1.8, κ = 0.56
    - k=10: λ = 2.8, κ = 0.36
    """
    lambda_val = 1 + 2 * rho_dest * (k_dest - 1)
    kappa = 1 / lambda_val
    n_triplets = n_destinations * k_dest // 2
    n_eff = n_triplets * kappa
    
    return {
        'kappa': kappa,
        'lambda': lambda_val,
        'n_triplets': n_triplets,
        'n_eff': n_eff,
        'efficiency': kappa,
    }


# =============================================================================
# DESIGN RECOMMENDATIONS
# =============================================================================

def recommend_design(n_points, target_n_eff=None, max_compute_budget=None, rho_dest=0.1):
    """
    Recommend triplet design given constraints.
    
    Args:
        n_points: Total points available
        target_n_eff: Desired effective sample size (if specified)
        max_compute_budget: Maximum triplets to evaluate (if specified)
        rho_dest: Destination correlation (default 0.1)
        
    Returns:
        Dictionary with recommended design parameters
    """
    # Use 1 anchor (free), rest as destinations
    n_destinations = n_points - 1
    
    # Ceiling on n_eff regardless of triplets
    n_eff_ceiling = n_destinations / (2 * rho_dest)
    
    results = {
        'n_anchors': 1,
        'n_destinations': n_destinations,
        'n_eff_ceiling': n_eff_ceiling,
    }
    
    if target_n_eff is not None:
        # What k_dest gives us target_n_eff?
        # n_eff ≈ T * κ = (n_d * k / 2) / (1 + 0.2(k-1))
        # Solve for k given target n_eff
        # This is a quadratic in k
        
        # n_eff * (1 + 0.2(k-1)) = n_d * k / 2
        # n_eff + 0.2 * n_eff * k - 0.2 * n_eff = n_d * k / 2
        # n_eff - 0.2 * n_eff = k * (n_d/2 - 0.2 * n_eff)
        # k = 0.8 * n_eff / (n_d/2 - 0.2 * n_eff)
        
        if target_n_eff >= n_eff_ceiling:
            results['warning'] = f"Target n_eff={target_n_eff} exceeds ceiling {n_eff_ceiling:.1f}"
            results['k_dest'] = n_destinations - 1  # Use all pairs
        else:
            denom = n_destinations / 2 - 2 * rho_dest * target_n_eff
            if denom > 0:
                k_opt = (1 - 2 * rho_dest) * target_n_eff / denom
                results['k_dest'] = max(1, int(np.ceil(k_opt)))
            else:
                results['k_dest'] = n_destinations - 1
    
    elif max_compute_budget is not None:
        # What k_dest gives us at most max_compute_budget triplets?
        # T = n_d * k / 2 <= budget
        # k <= 2 * budget / n_d
        k_max = int(2 * max_compute_budget / n_destinations)
        results['k_dest'] = max(1, min(k_max, n_destinations - 1))
    
    else:
        # Default: k=5 as sweet spot
        results['k_dest'] = min(5, n_destinations - 1)
    
    # Compute resulting design statistics
    k = results['k_dest']
    design_stats = compute_kappa_balanced(n_destinations, k, rho_dest)
    results.update(design_stats)
    
    return results


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import jax.random as jr
    
    print("=" * 60)
    print("Testing optimal triplet design")
    print("=" * 60)
    
    key = jr.key(42)
    p = 5
    w0 = jnp.ones(p) / p
    
    # Test different k_dest values
    n_dest = 50
    
    print(f"\nn_destinations = {n_dest}, n_anchors = 1")
    print("-" * 60)
    print(f"{'k_dest':>6} {'T':>8} {'n_eff':>8} {'κ':>8} {'efficiency':>10}")
    print("-" * 60)
    
    for k in [1, 2, 5, 10, 20, 49]:
        key, subkey = jr.split(key)
        
        try:
            T, P, indices, kappa, info = make_single_anchor_triplets(
                subkey, n_destinations=n_dest, k_dest=k,
                p=p, sig=0.5, tau=1.0, w0=w0
            )
            
            print(f"{k:>6} {info['n_triplets']:>8} {info['n_eff']:>8.1f} "
                  f"{kappa:>8.3f} {info['efficiency']:>10.3f}")
        except Exception as e:
            print(f"{k:>6} ERROR: {e}")
    
    # Test design recommendations
    print("\n" + "=" * 60)
    print("Design recommendations")
    print("=" * 60)
    
    for n_points in [50, 100, 500]:
        rec = recommend_design(n_points, target_n_eff=100)
        print(f"\nn_points={n_points}, target_n_eff=100:")
        print(f"  Recommended k_dest = {rec['k_dest']}")
        print(f"  Actual n_eff = {rec['n_eff']:.1f}")
        print(f"  Triplets = {rec['n_triplets']}")
        print(f"  Ceiling = {rec['n_eff_ceiling']:.1f}")
