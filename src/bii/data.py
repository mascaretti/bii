import jax
from jax import numpy as jnp
from jax import random
from jax import vmap

def one_anchor(k, X, nns=(1, 2), w=None, idx=None):
    assert len(nns) == 2
    assert nns[0] < nns[1]
    assert nns[0] > 0
    if w is None:
        w = jnp.ones(X.shape[1])
    else:
        assert w.shape[0] == X.shape[1]
    assert jnp.all(w >= 0)
    xk = X[k]
    X = X if idx is None else X[idx]
    diff = X - xk[None, :]
    d = jnp.sum(w[None, :] * (diff*diff), axis=1)
    order = jnp.argsort(d)
    triplet = jnp.array([order[0], order[nns[0]], order[nns[1]]]) if idx is None else jnp.array([k, order[nns[0]], order[nns[1]]])
    return triplet

def make_iid(key, n, p, sig, tau, w0, pool_size=500, on_source=False):
    T = []
    P = []
    for _ in range(n):
        T_curr = []
        P_curr = []
        key, subkey_x, subkey_eps, subkey_choice = random.split(key, 4)
        X = random.multivariate_normal(key=subkey_x, mean=jnp.zeros(p), 
                                       cov=jnp.square(tau) * jnp.eye(p), shape=(pool_size,))
        epsilon = random.multivariate_normal(key=subkey_eps, mean=jnp.zeros(p), 
                                            cov=jnp.diag(jnp.square(sig)), shape=(pool_size,))
        Y = X + epsilon
        Z = Y / jnp.sqrt(w0)[None, :]
        
        idx = random.choice(key=subkey_choice, a=pool_size, replace=False)
        if on_source:
            i, j, k = one_anchor(idx, Y)
        else:
            i, j, k = one_anchor(idx, X)
        T_curr.append(X[i]); T_curr.append(X[j]); T_curr.append(X[k])
        P_curr.append(Z[i]); P_curr.append(Z[j]); P_curr.append(Z[k])
        T.append(T_curr); P.append(P_curr)
    T = jnp.asarray(T); P = jnp.asarray(P)
    return T, P

def make_data(key, n_triplets, p, sig, tau, w0, on_source=False, data_multiplier=20, 
              sample_size=None, origin_ratio=0.1, split_data=False, repeat_pairs=False, 
              return_indices=False, allow_anchor_reuse=False):
    """
    Generate n_triplets triplets from a single dataset.

    Args:
        n_triplets: Number of triplets to generate
        data_multiplier: How many points to generate per triplet (default: 20)
                        Total points generated = n_triplets * data_multiplier
        sample_size: Fixed sample size (alternative to data_multiplier)
        origin_ratio: Fraction of points that can be anchors
        split_data: If True, anchors and destinations come from disjoint sets
        repeat_pairs: If True, allow destination pairs to be reused across triplets
        return_indices: If True, also return the indices of points in each triplet
        allow_anchor_reuse: If True, allow the same anchor to be used multiple times
                           (enables n_triplets > num_anchors for extreme correlation)
    """
    if sample_size is None and data_multiplier is not None:
        assert data_multiplier >= 1.
        n_points = n_triplets * data_multiplier  # Generate enough points
    elif sample_size is not None and data_multiplier is None:
        # Relaxed assertion: only require sample_size >= 3 (minimum for one triplet)
        # When allow_anchor_reuse=True, we can have n_triplets >> sample_size
        assert sample_size >= 3, "Need at least 3 points for a triplet"
        n_points = sample_size
    else:
        raise TypeError("Select either sample_size or data_multiplier")


    key, subkey_x, subkey_eps, subkey_choice, subkey_anchor = random.split(key, 5)

    # Generate dataset with n_points
    X = random.multivariate_normal(key=subkey_x, mean=jnp.zeros(p), 
                                   cov=jnp.square(tau) * jnp.eye(p), shape=(n_points,))
    epsilon = random.multivariate_normal(key=subkey_eps, mean=jnp.zeros(p), 
                                         cov=jnp.diag(jnp.square(sig)), shape=(n_points,))
    Y = X + epsilon
    Z = Y / jnp.sqrt(w0)[None, :]

    # Sample anchor indices
    num_anchors = int(jnp.floor(n_points * origin_ratio))
    num_anchors = max(num_anchors, 1)  # At least one anchor
    
    if not repeat_pairs and not allow_anchor_reuse:
        # Each disjoint triplet uses 3 points (if not split_data) or 2 points (if split_data)
        points_per_triplet = 2 if split_data else 3
        max_possible_triplets = min(num_anchors, n_points // points_per_triplet)
        
        if max_possible_triplets < n_triplets:
            raise ValueError(
                f"Cannot extract {n_triplets} disjoint triplets from {n_points} points "
                f"with origin_ratio={origin_ratio}. Maximum possible: {max_possible_triplets}. "
                f"Suggestions: (1) Increase origin_ratio (currently {origin_ratio}), "
                f"(2) Increase data_multiplier/sample_size, (3) Reduce n_triplets, "
                f"or (4) Set allow_anchor_reuse=True for extreme correlation experiment."
            )
    
    # Sample anchors (without replacement for the base set)
    idx_base = random.choice(key=subkey_choice, a=n_points, shape=(num_anchors,), replace=False)
    
    # If we need more triplets than anchors, resample anchors with replacement
    if allow_anchor_reuse and n_triplets > num_anchors:
        # Repeat anchors as needed, with random selection
        n_repeats = (n_triplets // num_anchors) + 1
        idx_repeated = jnp.tile(idx_base, n_repeats)
        # Shuffle to randomize which anchors get reused
        idx = random.permutation(subkey_anchor, idx_repeated)[:n_triplets + num_anchors]
    else:
        idx = idx_base

    # Generate triplets for all anchors
    data = Y if on_source else X
    nn_1 = n_points // 500 if n_points >= 500 else 1
    nn_2 = (2 * (n_points // 500)) if n_points >= 500 else 2
    vectorized_one_anchor = vmap(one_anchor, in_axes=(0, None, None, None, None))
    col_idx = jnp.setdiff1d(jnp.arange(n_points), idx_base)  # Use base anchors for split
    results = vectorized_one_anchor(idx, data, (nn_1,nn_2), None, None) if split_data is False else vectorized_one_anchor(idx, data, (nn_1,nn_2), None, col_idx)

    # Remove triplets that share ANY points (unless repeat_pairs or allow_anchor_reuse)
    used_points = set()
    T = []
    P = []
    # Also track indices
    T_indices = []    
    for triplet in results:
        i, j, k = int(triplet[0]), int(triplet[1]), int(triplet[2])
        triplet_points = {i, j, k} if split_data is False else {j, k}

        # Check if any point in this triplet has been used before
        # With allow_anchor_reuse, we only check destination pairs (not anchors)
        check_points = triplet_points if not allow_anchor_reuse else {j, k}
        
        if repeat_pairs or allow_anchor_reuse or check_points.isdisjoint(used_points):
            used_points.update(triplet_points)
            if split_data is False:
                T.append([X[i], X[j], X[k]])
                P.append([Z[i], Z[j], Z[k]])
                T_indices.append([i, j, k])
            else:
                T.append([X[i], X[col_idx][j], X[col_idx][k]])
                P.append([Z[i], Z[col_idx][j], Z[col_idx][k]])
                T_indices.append([i, col_idx[j], col_idx[k]])
            if len(T) >= n_triplets:
                break

    # Raise error if we got fewer triplets than requested
    actual_triplets = len(T)
    if actual_triplets < n_triplets:
        raise ValueError(
            f"Generated only {actual_triplets} triplets out of {n_triplets} requested "
            f"(n_points={n_points}, anchors={len(idx)}). "
            f"Increase data_multiplier (current: {data_multiplier}) or reduce n_triplets."
        )

    T = jnp.asarray(T)
    P = jnp.asarray(P)
    if return_indices:
        T_indices = jnp.asarray(T_indices)
        return T, P, T_indices
    else:
        return T, P

def make_hybrid(key, n_triplets, p, sig, tau, w0, on_source=False, 
                triplets_per_dataset=5, points_per_dataset=500):
    """
    Hybrid approach: Generate multiple datasets, extract k disjoint triplets from each.

    Args:
        n_triplets: Total number of triplets to generate
        triplets_per_dataset: How many disjoint triplets to extract from each dataset
        points_per_dataset: Size of each generated dataset
    """
    num_datasets = int(jnp.ceil(n_triplets / triplets_per_dataset))

    T = []
    P = []

    for dataset_idx in range(num_datasets):
        if len(T) >= n_triplets:
            break

        key, subkey_x, subkey_eps, subkey_choice = random.split(key, 4)

        # Generate one dataset
        X = random.multivariate_normal(key=subkey_x, mean=jnp.zeros(p), 
                                       cov=jnp.square(tau) * jnp.eye(p), 
                                       shape=(points_per_dataset,))
        epsilon = random.multivariate_normal(key=subkey_eps, mean=jnp.zeros(p), 
                                            cov=jnp.diag(jnp.square(sig)), 
                                            shape=(points_per_dataset,))
        Y = X + epsilon
        Z = Y / jnp.sqrt(w0)[None, :]

        # Sample anchors for this dataset
        origin_ratio = 0.1
        num_anchors = int(jnp.floor(points_per_dataset * origin_ratio))
        idx = random.choice(key=subkey_choice, a=points_per_dataset, 
                           shape=(num_anchors,), replace=False)

        # Generate triplets
        data = Y if on_source else X
        vectorized_one_anchor = vmap(one_anchor, in_axes=(0, None, None, None, None))
        results = vectorized_one_anchor(idx, data, (1,2), None)

        # Extract disjoint triplets from THIS dataset
        used_points = set()
        triplets_from_this_dataset = 0

        for triplet in results:
            if triplets_from_this_dataset >= triplets_per_dataset:
                break
            if len(T) >= n_triplets:
                break

            i, j, k = int(triplet[0]), int(triplet[1]), int(triplet[2])
            triplet_points = {i, j, k}

            # Check disjointness within this dataset only
            if triplet_points.isdisjoint(used_points):
                used_points.update(triplet_points)
                T.append([X[i], X[j], X[k]])
                P.append([Z[i], Z[j], Z[k]])
                triplets_from_this_dataset += 1

    if len(T) < n_triplets:
        raise ValueError(
            f"Only generated {len(T)} triplets out of {n_triplets} requested. "
            f"Try reducing triplets_per_dataset or increasing points_per_dataset."
        )

    T = jnp.asarray(T)
    P = jnp.asarray(P)
    return T, P


@jax.jit
def T_from_X(X):
    xi, xj, xk = X[:, 1], X[:, 2], X[:, 0]
    di = jnp.sum((xi - xk)**2, axis=1)
    dj = jnp.sum((xj - xk)**2, axis=1)
    return (di <= dj).astype(jnp.float32)


def count_shared_destinations(indices, split_data=True):
    seen_triplets = set()
    counter = 0
    for idx in indices:
        curr = {int(idx[1]), int(idx[2])} if split_data is True else {int(idx[i]) for i in idx}
        if seen_triplets.isdisjoint(curr) is False:
            counter += 1
        seen_triplets.update(curr)
    return counter


def make_data_multi_neighbor(key, n_anchors, triplets_per_anchor, p, sig, tau, w0, 
                              on_source=False, pool_size=None, return_indices=False, return_pool=False):
    """
    Generate multiple distinct triplets per anchor to induce point correlation
    without triplet duplication.
    
    For each anchor, we extract `triplets_per_anchor` triplets using different
    neighbor pairs: (1st, 2nd), (3rd, 4th), (5th, 6th), etc.
    
    This creates correlation through point reuse while keeping all triplets unique.
    
    Args:
        n_anchors: Number of distinct anchor points
        triplets_per_anchor: Number of triplets to form per anchor (k)
                            Total triplets = n_anchors * triplets_per_anchor
        p: Dimension
        sig: Noise std (scalar or array of shape (p,))
        tau: Signal std
        w0: True weights (p,)
        on_source: If True, compute distances on noisy Y; else on clean X
        pool_size: Total number of points to generate. If None, automatically
                   set to ensure enough destinations:
                   pool_size = n_anchors + 2 * n_anchors * triplets_per_anchor
        return_indices: If True, also return point indices for each triplet
    
    Returns:
        X: Triplet data (n_triplets, 3, p) - original embeddings  
        Z: Normalized embeddings (n_triplets, 3, p)
        indices (optional): (n_triplets, 3) array of point indices
        
    Example:
        n_anchors=100, triplets_per_anchor=5 -> 500 triplets from 100 anchors
        Each anchor contributes 5 triplets with neighbors:
          (1st, 2nd), (3rd, 4th), (5th, 6th), (7th, 8th), (9th, 10th)
    """
    n_triplets = n_anchors * triplets_per_anchor
    
    # Need enough points for: anchors + (2 * triplets_per_anchor) destinations per anchor
    # But destinations are shared across anchors, so we need a pool
    # Minimum: n_anchors (for anchors) + 2*triplets_per_anchor (for at least one anchor's destinations)
    # Safe default: enough that destinations aren't too heavily reused
    if pool_size is None:
        # Heuristic: want ~3-5x the minimum to allow some destination reuse but not extreme
        min_destinations = 2 * triplets_per_anchor
        pool_size = n_anchors + 4 * min_destinations
    
    # Ensure pool is large enough
    min_pool = n_anchors + 2 * triplets_per_anchor
    if pool_size < min_pool:
        raise ValueError(
            f"pool_size={pool_size} too small. Need at least {min_pool} "
            f"(n_anchors={n_anchors} + 2*triplets_per_anchor={2*triplets_per_anchor})"
        )
    
    key, subkey_x, subkey_eps, subkey_anchor = random.split(key, 4)
    
    # Generate the point cloud
    X_pool = random.multivariate_normal(
        key=subkey_x, mean=jnp.zeros(p),
        cov=jnp.square(tau) * jnp.eye(p), shape=(pool_size,)
    )
    epsilon = random.multivariate_normal(
        key=subkey_eps, mean=jnp.zeros(p),
        cov=jnp.diag(jnp.square(sig)), shape=(pool_size,)
    )
    Y_pool = X_pool + epsilon
    Z_pool = Y_pool / jnp.sqrt(w0)[None, :]
    
    # Select anchor indices (disjoint from destination pool)
    all_indices = jnp.arange(pool_size)
    anchor_indices = random.choice(subkey_anchor, all_indices, shape=(n_anchors,), replace=False)
    dest_indices = jnp.setdiff1d(all_indices, anchor_indices)
    
    # Distance computation basis
    data = Y_pool if on_source else X_pool
    
    # Neighbor spacing to avoid collapse (same logic as make_data)
    # With many points, nearest neighbors are at nearly identical distances
    n_dest = len(dest_indices)
    spacing = n_dest // 500 if n_dest >= 500 else 1
    
    # For each anchor, find neighbors with proper spacing
    T = []
    P = []
    T_indices = []
    
    for anchor_idx in anchor_indices:
        anchor_idx = int(anchor_idx)
        anchor_point = data[anchor_idx]
        
        # Compute distances to all destinations
        dest_points = data[dest_indices]
        dists = jnp.sum((dest_points - anchor_point[None, :]) ** 2, axis=1)
        
        # Sort destinations by distance
        sorted_dest_positions = jnp.argsort(dists)
        
        # Form triplets with spaced neighbors:
        # k=0: (spacing, 2*spacing), k=1: (3*spacing, 4*spacing), ...
        for k in range(triplets_per_anchor):
            pos1 = (2 * k + 1) * spacing - 1  # 0-indexed: spacing-1, 3*spacing-1, ...
            pos2 = (2 * k + 2) * spacing - 1  # 0-indexed: 2*spacing-1, 4*spacing-1, ...
            
            # Check bounds
            if pos2 >= n_dest:
                raise ValueError(
                    f"Not enough destinations for {triplets_per_anchor} triplets per anchor "
                    f"with spacing={spacing}. Need position {pos2}, but only {n_dest} destinations. "
                    f"Increase pool_size or reduce triplets_per_anchor."
                )
            
            dest1_idx = int(dest_indices[sorted_dest_positions[pos1]])
            dest2_idx = int(dest_indices[sorted_dest_positions[pos2]])
            
            T.append([X_pool[anchor_idx], X_pool[dest1_idx], X_pool[dest2_idx]])
            P.append([Z_pool[anchor_idx], Z_pool[dest1_idx], Z_pool[dest2_idx]])
            T_indices.append([anchor_idx, dest1_idx, dest2_idx])
    
    T = jnp.asarray(T)
    P = jnp.asarray(P)
    
    if return_indices is True:
        T_indices = jnp.asarray(T_indices)
        return T, P, T_indices
    elif return_indices and return_pool:
        return T, P, T_indices, X_pool, Z_pool
    elif return_indices is False and return_pool is False:
        return T, P, X_pool, Z_pool
    else:
        return T, P


def compute_correlation_diagnostics(indices):
    """
    Compute detailed diagnostics for triplet correlation/overlap.
    
    Args:
        indices: Array of shape (n_triplets, 3) with point indices [anchor, dest1, dest2]
    
    Returns:
        Dictionary with:
            - n_triplets: Total number of triplets
            - n_unique_points: Number of unique points used
            - n_unique_anchors: Number of unique anchor points
            - n_unique_destinations: Number of unique destination points
            - avg_anchor_reuse: Average times each anchor is reused
            - max_anchor_reuse: Maximum times any anchor is reused
            - avg_destination_reuse: Average times each destination is reused
            - max_destination_reuse: Maximum times any destination is reused
            - effective_n_ratio: n_unique_points / (3 * n_triplets) 
                                (1.0 = no reuse, lower = more correlation)
            - anchor_entropy: Entropy of anchor distribution (higher = more uniform)
            - destination_entropy: Entropy of destination distribution
    """
    from collections import Counter
    import numpy as np
    
    indices = np.asarray(indices)
    n_triplets = len(indices)
    
    anchors = indices[:, 0]
    destinations = indices[:, 1:].flatten()
    all_points = indices.flatten()
    
    # Count reuse
    anchor_counts = Counter(anchors)
    dest_counts = Counter(destinations)
    all_counts = Counter(all_points)
    
    n_unique_anchors = len(anchor_counts)
    n_unique_destinations = len(dest_counts)
    n_unique_points = len(all_counts)
    
    # Reuse statistics
    anchor_reuse = np.array(list(anchor_counts.values()))
    dest_reuse = np.array(list(dest_counts.values()))
    
    avg_anchor_reuse = np.mean(anchor_reuse)
    max_anchor_reuse = np.max(anchor_reuse)
    avg_destination_reuse = np.mean(dest_reuse)
    max_destination_reuse = np.max(dest_reuse)
    
    # Effective sample size ratio
    # If all points were unique, we'd have 3*n_triplets unique points
    effective_n_ratio = n_unique_points / (3 * n_triplets)
    
    # Entropy (measures uniformity of reuse)
    def entropy(counts):
        probs = counts / counts.sum()
        return -np.sum(probs * np.log(probs + 1e-10))
    
    anchor_entropy = entropy(anchor_reuse)
    dest_entropy = entropy(dest_reuse)
    
    # Pairwise overlap: how many triplet pairs share at least one point?
    # This is O(n^2) so only compute for small n
    if n_triplets <= 1000:
        triplet_sets = [set(indices[i]) for i in range(n_triplets)]
        overlap_count = 0
        for i in range(n_triplets):
            for j in range(i + 1, n_triplets):
                if not triplet_sets[i].isdisjoint(triplet_sets[j]):
                    overlap_count += 1
        pairwise_overlap_fraction = overlap_count / (n_triplets * (n_triplets - 1) / 2)
    else:
        pairwise_overlap_fraction = None  # Too expensive to compute
    
    return {
        'n_triplets': n_triplets,
        'n_unique_points': n_unique_points,
        'n_unique_anchors': n_unique_anchors,
        'n_unique_destinations': n_unique_destinations,
        'avg_anchor_reuse': avg_anchor_reuse,
        'max_anchor_reuse': max_anchor_reuse,
        'avg_destination_reuse': avg_destination_reuse,
        'max_destination_reuse': max_destination_reuse,
        'effective_n_ratio': effective_n_ratio,
        'anchor_entropy': anchor_entropy,
        'destination_entropy': dest_entropy,
        'pairwise_overlap_fraction': pairwise_overlap_fraction,
    }
        
