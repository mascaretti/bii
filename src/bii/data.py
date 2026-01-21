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

def make_iid(key, n, p, sig, tau, w0, on_source=False):
    T = []
    P = []
    for _ in range(n):
        T_curr = []
        P_curr = []
        key, subkey_x, subkey_eps, subkey_choice = random.split(key, 4)
        X = random.multivariate_normal(key=subkey_x, mean=jnp.zeros(p), 
                                       cov=jnp.square(tau) * jnp.eye(p), shape=(500,))
        epsilon = random.multivariate_normal(key=subkey_eps, mean=jnp.zeros(p), 
                                            cov=jnp.diag(jnp.square(sig)), shape=(500,))
        Y = X + epsilon
        Z = Y / jnp.sqrt(w0)[None, :]
        
        idx = random.choice(key=subkey_choice, a=500, replace=False)
        if on_source:
            i, j, k = one_anchor(idx, Y)
        else:
            i, j, k = one_anchor(idx, X)
        T_curr.append(X[i]); T_curr.append(X[j]); T_curr.append(X[k])
        P_curr.append(Z[i]); P_curr.append(Z[j]); P_curr.append(Z[k])
        T.append(T_curr); P.append(P_curr)
    T = jnp.asarray(T); P = jnp.asarray(P)
    return T, P

def make_data(key, n_triplets, p, sig, tau, w0, on_source=False, data_multiplier=20, sample_size=None, origin_ratio=0.1, split_data=False, repeat_pairs=False):
    """
    Generate n_triplets disjoint triplets from a single dataset.

    Args:
        n_triplets: Number of triplets to generate
        data_multiplier: How many points to generate per triplet (default: 20)
                        Total points generated = n_triplets * data_multiplier
    """
    if sample_size is None and data_multiplier is not None:
        assert data_multiplier >= 1.
        n_points = n_triplets * data_multiplier  # Generate enough points
    elif sample_size is not None and data_multiplier is None:
        assert sample_size >= n_triplets
        n_points = sample_size
    else:
        raise TypeError("Select either sample_size or data_multiplier")


    key, subkey_x, subkey_eps, subkey_choice = random.split(key, 4)

    # Generate dataset with n_points
    X = random.multivariate_normal(key=subkey_x, mean=jnp.zeros(p), 
                                   cov=jnp.square(tau) * jnp.eye(p), shape=(n_points,))
    epsilon = random.multivariate_normal(key=subkey_eps, mean=jnp.zeros(p), 
                                         cov=jnp.diag(jnp.square(sig)), shape=(n_points,))
    Y = X + epsilon
    Z = Y / jnp.sqrt(w0)[None, :]

    # Sample anchor indices
    num_anchors = int(jnp.floor(n_points * origin_ratio))
    if not repeat_pairs:
        # Each disjoint triplet uses 3 points (if not split_data) or 2 points (if split_data)
        points_per_triplet = 2 if split_data else 3
        max_possible_triplets = min(num_anchors, n_points // points_per_triplet)
        
        if max_possible_triplets < n_triplets:
            raise ValueError(
                f"Cannot extract {n_triplets} disjoint triplets from {n_points} points "
                f"with origin_ratio={origin_ratio}. Maximum possible: {max_possible_triplets}. "
                f"Suggestions: (1) Increase origin_ratio (currently {origin_ratio}), "
                f"(2) Increase data_multiplier/sample_size, or (3) Reduce n_triplets."
            )
    
    idx = random.choice(key=subkey_choice, a=n_points, shape=(num_anchors,), replace=False)

    # Generate triplets for all anchors
    data = Y if on_source else X
    nn_1 = n_points // 500 if n_points >= 500 else 1
    nn_2 = (2 * (n_points // 500)) if n_points >= 500 else 2
    vectorized_one_anchor = vmap(one_anchor, in_axes=(0, None, None, None, None))
    col_idx = jnp.setdiff1d(jnp.arange(n_points), idx)
    results = vectorized_one_anchor(idx, data, (nn_1,nn_2), None, None) if split_data is False else vectorized_one_anchor(idx, data, (nn_1,nn_2), None, col_idx)

    # Remove triplets that share ANY points
    used_points = set()
    T = []
    P = []
    for triplet in results:
        i, j, k = int(triplet[0]), int(triplet[1]), int(triplet[2])
        triplet_points = {i, j, k} if split_data is False else {j, k}

        # Check if any point in this triplet has been used before
        if repeat_pairs or triplet_points.isdisjoint(used_points):
            used_points.update(triplet_points)
            if split_data is False:
                T.append([X[i], X[j], X[k]])
                P.append([Z[i], Z[j], Z[k]])
            else:
                T.append([X[i], X[col_idx][j], X[col_idx][k]])
                P.append([Z[i], Z[col_idx][j], Z[col_idx][k]])
            if len(T) >= n_triplets:
                break

    # Raise error if we got fewer triplets than requested
    actual_triplets = len(T)
    if actual_triplets < n_triplets:
        raise ValueError(
            f"Generated only {actual_triplets} triplets out of {n_triplets} requested "
            f"(n_points={n_points}, anchors={num_anchors}). "
            f"Increase data_multiplier (current: {data_multiplier}) or reduce n_triplets."
        )

    T = jnp.asarray(T)
    P = jnp.asarray(P)
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

