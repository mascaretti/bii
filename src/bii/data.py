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


def make_triplets_zfar(key, X_pool, Z_pool, sig, n_triplets, anchor_fraction=0.1,
                       rank_i=100, rank_j=200):
    """Form triplets with (i, j) at fixed Z-distance ranks from each anchor.

    For each anchor k, sort destinations by Mahalanobis Z-distance
        ``d2(l, k) = sum_d (z_{l,d} - z_{k,d})^2 / sigma_d^2``
    ascending, then pick (i, j) at sliding ranks ``(rank_i + t, rank_j + t)`` for
    ``t = 0..n_triplets - 1``. Labels are still computed from X-distance (Y).

    Motivation: under flat ``sigma`` on heavy-tailed data (e.g. NHANES nutrients
    with zero-inflated alcohol), random pairs occasionally place a single feature's
    rare-tail outlier in one of (i, j) but not the other. The resulting
    ``|a_d^2 - b_d^2|`` term blows the per-triplet probit statistic ``s`` past the
    sigmoid's saturating regime, and a single saturating-wrong triplet contributes
    ~``log Phi(-5) ≈ -15`` to the loglik. Z-rank-100 destinations are by
    construction close to the anchor in Z — they cannot be "anchor-and-one-pair-
    member-non-outlier vs other-pair-member-outlier" — so the outlier pairs are
    excluded from the triplet set.

    For ``sigma`` flat across features, ``d2`` reduces to plain Euclidean Z-distance
    (up to a global scale). For ``sigma`` proportional to per-feature std, each
    feature contributes equally to the ranking.

    Args:
        key: JAX random key (used to sample anchors).
        X_pool: (N, p_x) reference embeddings (used for labels via X-distance).
        Z_pool: (N, p_z) noisy/normalised embeddings (used for ranking AND loglik).
        sig: per-feature noise std — scalar or (p_z,). Same convention as in
            ``fit_bii``; only the ratio ``1 / sigma_d^2`` matters for the ranking.
        n_triplets: triplets per anchor (sliding window length).
        anchor_fraction: fraction of pool used as anchors.
        rank_i: 1-based starting Z-rank for candidate ``i`` (e.g. 100 = 100th NN).
        rank_j: 1-based starting Z-rank for candidate ``j`` (e.g. 200 = 200th NN).

    Returns:
        ``(T, X, Z, indices)`` — same shapes / semantics as :func:`make_triplets`.

    Raises:
        ValueError: if ``rank_i``/``rank_j`` violate ``1 <= rank_i < rank_j`` or
            ``rank_j + n_triplets > N``.
    """
    if not (1 <= rank_i < rank_j):
        raise ValueError(
            f"need 1 <= rank_i < rank_j; got rank_i={rank_i}, rank_j={rank_j}"
        )
    N = Z_pool.shape[0]
    if rank_j + n_triplets > N:
        raise ValueError(
            f"rank_j + n_triplets = {rank_j + n_triplets} exceeds N = {N}"
        )

    sig2 = jnp.asarray(sig).astype(Z_pool.dtype) ** 2
    n_anchors = max(1, int(N * anchor_fraction))

    key, k_anchors = random.split(key)
    perm = random.permutation(k_anchors, N)
    anchor_idx = perm[:n_anchors]

    offsets = jnp.arange(n_triplets)
    i_positions = (rank_i - 1) + offsets
    j_positions = (rank_j - 1) + offsets

    def one_anchor(k_idx):
        z_k = Z_pool[k_idx]
        diff = Z_pool - z_k[None, :]
        d2 = jnp.sum((diff * diff) / sig2[None, :], axis=1)
        d2 = d2.at[k_idx].set(jnp.inf)
        order = jnp.argsort(d2)
        i_idx = order[i_positions]
        j_idx = order[j_positions]
        k_rep = jnp.full((n_triplets,), k_idx)
        return jnp.stack([k_rep, i_idx, j_idx], axis=1)

    indices = jax.vmap(one_anchor)(anchor_idx)
    indices = indices.reshape(-1, 3)

    X = X_pool[indices]
    Z = Z_pool[indices]
    T = T_from_X(X)
    return T, X, Z, indices
