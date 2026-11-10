"""Triplet formation from paired observation pools — pure functions."""

import jax
from jax import numpy as jnp
from jax import random


@jax.jit
def T_from_X(X):
    """Binary labels from clean-space triplets.

    Convention: ``[:, 0]`` is the anchor.
    ``T = 1`` when column 1 is closer to the anchor than column 2.

    Args:
        X: (n, 3, p) clean embeddings.

    Returns:
        (n,) binary labels in {0, 1}.
    """
    xi, xj, xk = X[:, 1], X[:, 2], X[:, 0]
    di = jnp.sum((xi - xk) ** 2, axis=1)
    dj = jnp.sum((xj - xk) ** 2, axis=1)
    return (di <= dj).astype(jnp.float32)


def kappa_from_triplets(indices):
    """Composite-likelihood temperature from the triplet overlap structure.

    Triplets that share an anchor are correlated, so the product likelihood
    over-counts information. The simplest calibration of the generalised-Bayes
    learning rate is the fraction of distinct anchors,
    ``kappa = n_unique_anchors / n_triplets`` (column 0 is the anchor), which
    deflates the evidence to the effective number of independent comparisons.

    Args:
        indices: (n_triplets, 3) pool indices, column 0 the anchor.

    Returns:
        float ``kappa`` in ``(0, 1]``.
    """
    indices = jnp.asarray(indices)
    n_triplets = indices.shape[0]
    n_anchors = jnp.unique(indices[:, 0]).shape[0]
    return float(n_anchors) / float(n_triplets)


def make_triplets(key, X_pool, Z_pool, n_triplets, anchor_fraction=0.1):
    """Form triplets from paired observation pools.

    Partitions the pool into anchors and destinations, then draws
    ``n_triplets`` random destination pairs per anchor.

    Args:
        key: JAX random key.
        X_pool: (N, p_x) clean reference embeddings.
        Z_pool: (N, p_z) noisy/normalised embeddings.
        n_triplets: destination pairs per anchor.
        anchor_fraction: fraction of pool used as anchors.

    Returns:
        ``(T, X, Z, indices)``
            T: (n_total,) binary labels.
            X: (n_total, 3, p_x) clean triplets.
            Z: (n_total, 3, p_z) noisy triplets.
            indices: (n_total, 3) pool indices.
    """
    N = X_pool.shape[0]
    n_anchors = max(1, int(N * anchor_fraction))

    key, k_split, k_pairs = random.split(key, 3)

    perm = random.permutation(k_split, N)
    anchor_idx = perm[:n_anchors]
    dest_idx = perm[n_anchors:]
    n_dest = dest_idx.shape[0]

    total = n_anchors * n_triplets
    pair_keys = random.split(k_pairs, total)

    def _sample_pair(key):
        return random.choice(key, n_dest, shape=(2,), replace=False)

    pair_positions = jax.vmap(_sample_pair)(pair_keys)

    anchor_repeated = jnp.repeat(anchor_idx, n_triplets)
    indices = jnp.stack(
        [
            anchor_repeated,
            dest_idx[pair_positions[:, 0]],
            dest_idx[pair_positions[:, 1]],
        ],
        axis=1,
    )

    X = X_pool[indices]
    Z = Z_pool[indices]
    T = T_from_X(X)
    return T, X, Z, indices


def make_triplets_local(key, X_pool, Z_pool, sig, n_triplets, anchor_fraction=0.1):
    """Form triplets local in the target space.

    The anchors and destinations are those of :func:`make_triplets` for the same
    key. Each anchor k has its own scale sigma_k, the target distance to its
    ``n_triplets``-th nearest destination. Each of its ``n_triplets`` pairs draws
    one candidate among the destinations within sigma_k and the other among those
    in (sigma_k, 2 sigma_k], or among the next ``n_triplets`` nearest if that shell
    holds fewer. The near candidate goes in column 1 or 2 at random, so the labels
    are balanced; the order of the candidates does not change the likelihood.

    Args:
        key: JAX random key.
        X_pool: (N, p_x) clean reference embeddings.
        Z_pool: (N, p_z) noisy/normalised embeddings.
        sig: unused; the signature of ``fit_bii``'s ``triplet_sampler``.
        n_triplets: destination pairs per anchor; also sets sigma_k.
        anchor_fraction: fraction of pool used as anchors.

    Returns:
        ``(T, X, Z, indices)``, as :func:`make_triplets`.
    """
    del sig
    X_pool = jnp.asarray(X_pool)
    Z_pool = jnp.asarray(Z_pool)
    N = X_pool.shape[0]
    n_anchors = max(1, int(N * anchor_fraction))
    m = n_triplets

    key, k_split, k_pairs = random.split(key, 3)  # as make_triplets
    perm = random.permutation(k_split, N)
    anchor_idx = perm[:n_anchors]
    dest_idx = perm[n_anchors:]
    if dest_idx.shape[0] < 2 * m:
        raise ValueError(f"{dest_idx.shape[0]} destinations; need at least {2 * m}")

    # Destinations of each anchor by target distance, nearest first
    diff = X_pool[anchor_idx][:, None, :] - X_pool[dest_idx][None, :, :]
    dist = jnp.sqrt(jnp.sum(diff * diff, axis=-1))  # (n_anchors, n_dest)
    order = jnp.argsort(dist, axis=1)
    dist = jnp.take_along_axis(dist, order, axis=1)
    sigma = dist[:, m - 1]  # the m nearest are within sigma_k
    n_shell = jnp.sum((dist > sigma[:, None]) & (dist <= 2.0 * sigma[:, None]), axis=1)
    n_shell = jnp.maximum(n_shell, m)  # else the next m nearest

    k_near, k_far, k_swap = random.split(k_pairs, 3)
    near = random.randint(k_near, (n_anchors, m), 0, m)  # positions 0 .. m-1
    far = m + jnp.floor(random.uniform(k_far, (n_anchors, m)) * n_shell[:, None]).astype(int)
    near = dest_idx[jnp.take_along_axis(order, near, axis=1)].reshape(-1)
    far = dest_idx[jnp.take_along_axis(order, far, axis=1)].reshape(-1)
    swap = random.bernoulli(k_swap, 0.5, (n_anchors * m,))

    indices = jnp.stack([jnp.repeat(anchor_idx, m), jnp.where(swap, far, near),
                         jnp.where(swap, near, far)], axis=1)
    X = X_pool[indices]
    Z = Z_pool[indices]
    T = T_from_X(X)
    return T, X, Z, indices
