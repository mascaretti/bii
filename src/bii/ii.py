"""
Information Imbalance computation (L2 metric).

Adapted from deepseek-hidden-states/information_imbalance_module.py
(Acevedo, Mascaretti, Rende et al.).

The Information Imbalance Δ(X→Y) measures how well nearest-neighbour
structure in space X predicts nearest-neighbour structure in space Y.
Values close to 0 mean X is highly informative about Y.
"""

import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.stats import rankdata


def L2_distance(x, y):  # noqa: N802
    """Euclidean distance between two vectors."""
    return jnp.linalg.norm(x=x - y, ord=None)


def pairwise_distances(xs, ys):
    """Pairwise L2 distances: (len(ys), len(xs))."""
    return jax.vmap(lambda x: jax.vmap(lambda y: L2_distance(x, y))(xs))(ys).T


def _compute_relative_rank(x_dist, y_dist):
    """For one query point, compute its Y-rank among X-nearest neighbours."""
    x_rank = rankdata(x_dist, method="min")
    y_rank = rankdata(y_dist, method="min")
    rel_ranks = jnp.where(x_rank == 1, y_rank - 1, -1)
    return jnp.sum(rel_ranks, where=rel_ranks != -1), jnp.asarray(rel_ranks != -1).sum()


_batch_relative_ranks = jax.vmap(_compute_relative_rank)


def build_information_imbalance(key, sample_size):
    """Construct a JIT-compiled function that computes II(X→Y) and II(Y→X).

    Uses a row/column subsampling scheme for scalability:
    - 500 row samples (queries) if N >= 2500, else 20% of N
    - remaining points serve as columns (reference set)

    Args:
        key: JAX random key for subsampling.
        sample_size: number of data points N.

    Returns:
        Callable (X, Y) -> (Δ(X→Y), Δ(Y→X)).
    """
    n_row_samples = (
        jnp.array(500) if sample_size >= 2500 else jnp.floor(0.2 * sample_size).astype(int)
    )
    n_col_samples = (sample_size - n_row_samples).astype(int)

    key, subkey = random.split(key)
    indices_rows = random.choice(key=subkey, a=sample_size, shape=(n_row_samples,), replace=False)

    remaining_indices = jnp.setdiff1d(jnp.arange(sample_size), indices_rows)
    key, subkey = random.split(key)
    indices_columns = random.choice(
        key=subkey, a=remaining_indices, shape=(n_col_samples,), replace=False
    )

    def information_imbalance(X, Y):
        """Compute II(X→Y) and II(Y→X).

        Args:
            X: (N, p_x) origin space.
            Y: (N, p_y) destination space.

        Returns:
            (Δ(X→Y), Δ(Y→X)) — both in [0, 1].
        """
        d_X = pairwise_distances(X[indices_rows], X[indices_columns])
        d_Y = pairwise_distances(Y[indices_rows], Y[indices_columns])

        ranks, cards = _batch_relative_ranks(d_X, d_Y)
        total_cards = jnp.sum(cards)
        inf_imb = jnp.where(
            total_cards > 0,
            2.0 * (jnp.sum(ranks) / total_cards) / (n_col_samples - 1.0),
            jnp.nan,
        )

        reciprocal_ranks, reciprocal_cards = _batch_relative_ranks(d_Y, d_X)
        total_reciprocal_cards = jnp.sum(reciprocal_cards)
        reciprocal_inf_imb = jnp.where(
            total_reciprocal_cards > 0,
            2.0 * (jnp.sum(reciprocal_ranks) / total_reciprocal_cards) / (n_col_samples - 1.0),
            jnp.nan,
        )
        return inf_imb, reciprocal_inf_imb

    return jax.jit(information_imbalance)


def compute_ii(X, Z, key, w=None):
    """Compute Information Imbalance with optional weighted Z metric.

    Args:
        X: (N, p_x) reference space.
        Z: (N, p_z) target space.
        key: JAX random key.
        w: optional (p_z,) weight vector on the simplex.
            If provided, distances in Z are computed as
            d(z_i, z_j) = ||sqrt(w) * (z_i - z_j)||_2.

    Returns:
        (Δ(X→Z), Δ(Z→X)).
    """
    if w is not None:
        Z = Z * jnp.sqrt(w)
    N = X.shape[0]
    ii_fn = build_information_imbalance(key, N)
    return ii_fn(X, Z)
