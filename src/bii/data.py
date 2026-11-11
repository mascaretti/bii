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


def make_triplets_softmax(key, X_pool, Z_pool, sig, n_triplets, anchor_fraction=0.1, rank=15):
    """Form triplets whose candidates are drawn near the anchor in the target space.

    The anchors and destinations are those of :func:`make_triplets` for the same
    key. Anchor k draws each of its ``n_triplets`` pairs of candidates without
    replacement, destination l with probability proportional to
    ``exp(-d(k, l)^2 / lambda_k)``: d is the target distance, and the bandwidth
    ``lambda_k`` is the squared distance from k to its ``rank``-th nearest
    destination. The two candidates are listed in random order, so the labels are
    balanced; the order does not change the likelihood.

    Args:
        key: JAX random key.
        X_pool: (N, p_x) clean reference embeddings.
        Z_pool: (N, p_z) noisy/normalised embeddings.
        sig: unused; the signature of ``fit_bii``'s ``triplet_sampler``.
        n_triplets: destination pairs per anchor.
        anchor_fraction: fraction of pool used as anchors.
        rank: the neighbour of each anchor that sets its bandwidth.

    Returns:
        ``(T, X, Z, indices)``, as :func:`make_triplets`.
    """
    del sig
    X_pool = jnp.asarray(X_pool)
    Z_pool = jnp.asarray(Z_pool)
    N = X_pool.shape[0]
    n_anchors = max(1, int(N * anchor_fraction))

    key, k_split, k_pairs = random.split(key, 3)  # as make_triplets
    perm = random.permutation(k_split, N)
    anchor_idx = perm[:n_anchors]
    dest_idx = perm[n_anchors:]
    n_dest = dest_idx.shape[0]
    if n_dest < 2 or not 1 <= rank <= n_dest:
        raise ValueError(f"rank {rank} with {n_dest} destinations: need two or more "
                         "destinations, and a rank between 1 and their number")

    # Squared target distances, (n_anchors, n_dest), and the log weights
    diff = X_pool[anchor_idx][:, None, :] - X_pool[dest_idx][None, :, :]
    d2 = jnp.sum(diff * diff, axis=-1)
    bandwidth = jnp.sort(d2, axis=1)[:, rank - 1]
    logits = -d2 / (bandwidth[:, None] + 1e-12)

    # Two destinations without replacement: the two largest of the log weights plus
    # Gumbel noise
    k_draw, k_swap = random.split(k_pairs)
    noise = random.gumbel(k_draw, (n_anchors, n_triplets, n_dest))
    pair = jax.lax.top_k(logits[:, None, :] + noise, 2)[1]  # (n_anchors, n_triplets, 2)
    swap = random.bernoulli(k_swap, 0.5, (n_anchors, n_triplets, 1))
    pair = jnp.where(swap, pair[..., ::-1], pair).reshape(-1, 2)

    indices = jnp.column_stack([jnp.repeat(anchor_idx, n_triplets), dest_idx[pair]])
    X = X_pool[indices]
    Z = Z_pool[indices]
    T = T_from_X(X)
    return T, X, Z, indices
