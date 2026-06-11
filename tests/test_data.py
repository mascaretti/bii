"""Tests for triplet formation (data.py)."""

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from bii.data import (
    T_from_X,
    make_triplets,
    make_triplets_random_sparse,
    make_triplets_rank_weighted,
    make_triplets_yfar,
    make_triplets_z_informative,
    make_triplets_z_softmax,
    make_triplets_zfar,
    target_yfar_bump,
)


def _make_pool(key, n=100, p=3, sig=0.1):
    k1, k2 = jr.split(key)
    X = jr.normal(k1, (n, p))
    Z = X + jr.normal(k2, (n, p)) * sig
    return X, Z


def test_T_from_X_binary():
    """T_from_X should produce only 0.0 and 1.0."""
    key = jr.PRNGKey(0)
    X, Z = _make_pool(key, n=50, p=3)
    T, X_trip, Z_trip, _ = make_triplets(key, X, Z, n_triplets=10)
    assert set(np.array(T).tolist()).issubset({0.0, 1.0})


def test_T_from_X_deterministic():
    """Same input should give same output."""
    X = jr.normal(jr.PRNGKey(1), (30, 3, 4))
    T1 = T_from_X(X)
    T2 = T_from_X(X)
    assert jnp.array_equal(T1, T2)


def test_T_from_X_convention():
    """T=1 means column 1 closer to column 0 (anchor) than column 2."""
    X = jnp.array([
        [[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]],  # point 1 closer -> T=1
        [[0.0, 0.0], [10.0, 0.0], [1.0, 0.0]],  # point 2 closer -> T=0
    ])
    T = T_from_X(X)
    assert float(T[0]) == 1.0
    assert float(T[1]) == 0.0


def test_make_triplets_shapes():
    """make_triplets returns correct shapes."""
    key = jr.PRNGKey(1)
    X, Z = _make_pool(key, n=80, p=4)
    T, X_trip, Z_trip, idx = make_triplets(key, X, Z, n_triplets=30)
    n_anchors = max(1, int(80 * 0.1))
    total = n_anchors * 30
    assert T.shape == (total,)
    assert X_trip.shape == (total, 3, 4)
    assert Z_trip.shape == (total, 3, 4)
    assert idx.shape == (total, 3)


def test_make_triplets_binary_labels():
    """Labels should be binary."""
    key = jr.PRNGKey(2)
    X, Z = _make_pool(key, n=50, p=3)
    T, _, _, _ = make_triplets(key, X, Z, n_triplets=20)
    assert set(np.array(T).tolist()).issubset({0.0, 1.0})


def test_make_triplets_returns_X():
    """make_triplets now returns clean triplets X as well."""
    key = jr.PRNGKey(3)
    X_pool, Z_pool = _make_pool(key, n=60, p=3)
    T, X_trip, Z_trip, indices = make_triplets(key, X_pool, Z_pool, n_triplets=10)
    # X_trip should be from X_pool via indices
    assert X_trip.shape[1] == 3
    assert X_trip.shape[2] == 3


def test_make_triplets_indices_valid():
    """All indices should be valid pool indices."""
    key = jr.PRNGKey(4)
    n = 100
    X, Z = _make_pool(key, n=n, p=3)
    _, _, _, idx = make_triplets(key, X, Z, n_triplets=20)
    assert jnp.all(idx >= 0)
    assert jnp.all(idx < n)


def test_make_triplets_zfar_shapes_and_indices():
    """zfar sampler returns correct shapes and valid indices, no self-anchors."""
    key = jr.PRNGKey(0)
    n, p = 500, 4
    X, Z = _make_pool(key, n=n, p=p)
    sig = jnp.ones(p)
    T, X_trip, Z_trip, idx = make_triplets_zfar(
        key, X, Z, sig, n_triplets=5, anchor_fraction=0.1,
        rank_i=10, rank_j=20,
    )
    n_anchors = max(1, int(n * 0.1))
    total = n_anchors * 5
    assert T.shape == (total,)
    assert X_trip.shape == (total, 3, p)
    assert Z_trip.shape == (total, 3, p)
    assert idx.shape == (total, 3)
    assert jnp.all(idx >= 0) and jnp.all(idx < n)
    # i, j must differ from the anchor (we masked the anchor's own distance to inf)
    assert jnp.all(idx[:, 1] != idx[:, 0])
    assert jnp.all(idx[:, 2] != idx[:, 0])
    # i and j should be distinct (rank_i != rank_j by construction)
    assert jnp.all(idx[:, 1] != idx[:, 2])


def test_make_triplets_zfar_respects_z_rank():
    """When sig is uniform, i should be the (rank_i)-th NN in Euclidean Z-space."""
    key = jr.PRNGKey(1)
    n, p = 200, 3
    X, Z = _make_pool(key, n=n, p=p)
    sig = jnp.ones(p)
    rank_i = 5
    rank_j = 15
    T, _, _, idx = make_triplets_zfar(
        key, X, Z, sig, n_triplets=1, anchor_fraction=0.05,
        rank_i=rank_i, rank_j=rank_j,
    )
    # Recompute the (rank_i)-th and (rank_j)-th NNs in Z-space for each anchor
    for row in np.asarray(idx):
        k, i_obs, j_obs = int(row[0]), int(row[1]), int(row[2])
        z_k = np.asarray(Z[k])
        d2 = np.sum((np.asarray(Z) - z_k) ** 2, axis=1)
        d2[k] = np.inf  # mask self
        order = np.argsort(d2)
        assert i_obs == int(order[rank_i - 1])
        assert j_obs == int(order[rank_j - 1])


def test_make_triplets_zfar_rejects_bad_ranks():
    """Sampler raises ValueError on invalid rank arguments."""
    key = jr.PRNGKey(2)
    n, p = 100, 3
    X, Z = _make_pool(key, n=n, p=p)
    sig = jnp.ones(p)
    with pytest.raises(ValueError):
        make_triplets_zfar(key, X, Z, sig, n_triplets=5, rank_i=10, rank_j=5)
    with pytest.raises(ValueError):
        # rank_j + n_triplets > N
        make_triplets_zfar(key, X, Z, sig, n_triplets=10, rank_i=10, rank_j=95)


# --- make_triplets_yfar ---

def test_make_triplets_yfar_shapes_and_indices():
    key = jr.PRNGKey(0)
    n, p = 300, 4
    X, Z = _make_pool(key, n=n, p=p)
    T, X_trip, Z_trip, idx = make_triplets_yfar(
        key, X, Z, sig=None, n_triplets=5, anchor_fraction=0.1,
        rank_i=10, rank_j=20,
    )
    total = max(1, int(n * 0.1)) * 5
    assert T.shape == (total,)
    assert X_trip.shape == (total, 3, p)
    assert Z_trip.shape == (total, 3, p)
    assert idx.shape == (total, 3)
    assert jnp.all(idx >= 0) and jnp.all(idx < n)
    assert jnp.all(idx[:, 1] != idx[:, 0])
    assert jnp.all(idx[:, 2] != idx[:, 0])


def test_make_triplets_yfar_unbalanced_labels_constant():
    """Without label balancing, T == 1 by construction (rank_i < rank_j)."""
    key = jr.PRNGKey(1)
    X, Z = _make_pool(key, n=200, p=3)
    T, _, _, _ = make_triplets_yfar(
        key, X, Z, sig=None, n_triplets=5, anchor_fraction=0.1,
        rank_i=5, rank_j=50, balance_labels=False,
    )
    assert jnp.all(T == 1.0)


def test_make_triplets_yfar_balanced_labels_mixed():
    """With label balancing, both labels should appear."""
    key = jr.PRNGKey(2)
    X, Z = _make_pool(key, n=200, p=3)
    T, _, _, _ = make_triplets_yfar(
        key, X, Z, sig=None, n_triplets=10, anchor_fraction=0.2,
        rank_i=5, rank_j=50, balance_labels=True,
    )
    assert jnp.any(T == 1.0) and jnp.any(T == 0.0)


def test_make_triplets_yfar_rejects_bad_ranks():
    key = jr.PRNGKey(3)
    X, Z = _make_pool(key, n=100, p=3)
    with pytest.raises(ValueError):
        make_triplets_yfar(key, X, Z, sig=None, n_triplets=5, rank_i=10, rank_j=5)
    with pytest.raises(ValueError):
        make_triplets_yfar(key, X, Z, sig=None, n_triplets=10, rank_i=10, rank_j=95)


# --- make_triplets_rank_weighted ---

def test_make_triplets_rank_weighted_returns_5_tuple():
    key = jr.PRNGKey(0)
    n, p = 150, 3
    X, Z = _make_pool(key, n=n, p=p)
    out = make_triplets_rank_weighted(key, X, Z, sig=None, n_triplets=10,
                                      anchor_fraction=0.1)
    assert len(out) == 5
    T, X_trip, Z_trip, idx, weights = out
    total = max(1, int(n * 0.1)) * 10
    assert T.shape == (total,)
    assert weights.shape == (total,)
    assert idx.shape == (total, 3)
    assert jnp.all(idx >= 0) and jnp.all(idx < n)


def test_make_triplets_rank_weighted_uniform_weights():
    """Without a target, all importance weights are 1."""
    key = jr.PRNGKey(1)
    X, Z = _make_pool(key, n=150, p=3)
    *_, weights = make_triplets_rank_weighted(key, X, Z, sig=None, n_triplets=10)
    assert jnp.allclose(weights, 1.0)


def test_make_triplets_rank_weighted_target_self_normalised():
    """With a target, weights vary but stay self-normalised to the count."""
    key = jr.PRNGKey(2)
    X, Z = _make_pool(key, n=150, p=3)
    target = target_yfar_bump(mu_a=20.0, mu_b=50.0, sigma=10.0)
    *_, weights = make_triplets_rank_weighted(
        key, X, Z, sig=None, n_triplets=10, target_logweight_fn=target,
    )
    assert jnp.all(weights >= 0)
    assert not jnp.allclose(weights, 1.0)
    assert jnp.allclose(jnp.sum(weights), weights.shape[0], rtol=1e-4)


def test_make_triplets_rank_weighted_rejects_bad_k_max():
    key = jr.PRNGKey(3)
    X, Z = _make_pool(key, n=100, p=3)
    with pytest.raises(ValueError):
        make_triplets_rank_weighted(key, X, Z, sig=None, n_triplets=5, k_max=0)
    with pytest.raises(ValueError):
        make_triplets_rank_weighted(key, X, Z, sig=None, n_triplets=5, k_max=100)


def test_target_yfar_bump_peaks_at_centre():
    fn = target_yfar_bump(mu_a=10.0, mu_b=20.0, sigma=5.0)
    at_centre = fn(10.0, 20.0)
    off_centre = fn(30.0, 40.0)
    assert at_centre == 0.0
    assert off_centre < at_centre


# --- make_triplets_random_sparse ---

def test_make_triplets_random_sparse_filters_saturating():
    """Survivors must satisfy |s| < z_{1-eps} under the reference (uniform) w."""
    from scipy.stats import norm

    from bii.inference import delta_V_one_triplet

    key = jr.PRNGKey(0)
    n, p = 100, 3
    X, Z = _make_pool(key, n=n, p=p)
    sig = 0.1
    eps = 0.1
    T, X_trip, Z_trip, idx = make_triplets_random_sparse(
        key, X, Z, sig, n_triplets=20, anchor_fraction=0.1,
        eps=eps, oversample=2,
    )
    n_kept = T.shape[0]
    assert X_trip.shape == (n_kept, 3, p)
    assert Z_trip.shape == (n_kept, 3, p)
    assert idx.shape == (n_kept, 3)

    threshold = float(norm.ppf(1.0 - eps))
    w_ref = jnp.full(p, 1.0 / p)
    sig2 = jnp.full(p, sig**2)
    for t in range(min(n_kept, 50)):
        delta, V = delta_V_one_triplet(
            Z_trip[t, 1], Z_trip[t, 2], Z_trip[t, 0], w_ref, sig2, sig2, sig2
        )
        s = float(delta / jnp.sqrt(V + 1e-12))
        assert abs(s) < threshold


# --- make_triplets_z_softmax ---

def test_make_triplets_z_softmax_shapes_and_indices():
    key = jr.PRNGKey(0)
    n, p = 200, 4
    X, Z = _make_pool(key, n=n, p=p)
    T, X_trip, Z_trip, idx = make_triplets_z_softmax(
        key, X, Z, sig=1.0, n_triplets=10, anchor_fraction=0.1,
        lambda_close=1.0, lambda_far=4.0,
    )
    total = max(1, int(n * 0.1)) * 10
    assert T.shape == (total,)
    assert X_trip.shape == (total, 3, p)
    assert Z_trip.shape == (total, 3, p)
    assert idx.shape == (total, 3)
    assert jnp.all(idx >= 0) and jnp.all(idx < n)
    # Anchor's own distance is masked to inf -> never drawn as i or j
    assert jnp.all(idx[:, 1] != idx[:, 0])
    assert jnp.all(idx[:, 2] != idx[:, 0])


def test_make_triplets_z_softmax_rejects_bad_lambdas():
    key = jr.PRNGKey(1)
    X, Z = _make_pool(key, n=100, p=3)
    with pytest.raises(ValueError):
        make_triplets_z_softmax(key, X, Z, sig=1.0, n_triplets=5, lambda_close=0.0)
    with pytest.raises(ValueError):
        make_triplets_z_softmax(key, X, Z, sig=1.0, n_triplets=5, lambda_far=-1.0)


def test_make_triplets_z_softmax_tight_kernel_draws_close():
    """A very tight lambda_close should draw i from the anchor's nearest few."""
    key = jr.PRNGKey(2)
    n, p = 200, 3
    X, Z = _make_pool(key, n=n, p=p)
    T, _, _, idx = make_triplets_z_softmax(
        key, X, Z, sig=1.0, n_triplets=5, anchor_fraction=0.05,
        lambda_close=1e-3, lambda_far=1e-3,
    )
    for row in np.asarray(idx):
        k, i_obs = int(row[0]), int(row[1])
        d2 = np.sum((np.asarray(Z) - np.asarray(Z[k])) ** 2, axis=1)
        d2[k] = np.inf
        nearest = set(np.argsort(d2)[:5].tolist())
        assert i_obs in nearest


# --- make_triplets_z_informative ---

def test_make_triplets_z_informative_shapes_and_indices():
    key = jr.PRNGKey(0)
    n, p = 200, 4
    X, Z = _make_pool(key, n=n, p=p)
    T, X_trip, Z_trip, idx = make_triplets_z_informative(
        key, X, Z, sig=1.0, n_triplets=10, anchor_fraction=0.1,
        k_window=50, n_oversample=5,
    )
    total = max(1, int(n * 0.1)) * 10
    assert T.shape == (total,)
    assert X_trip.shape == (total, 3, p)
    assert Z_trip.shape == (total, 3, p)
    assert idx.shape == (total, 3)
    assert jnp.all(idx >= 0) and jnp.all(idx < n)
    assert jnp.all(idx[:, 1] != idx[:, 0])
    assert jnp.all(idx[:, 2] != idx[:, 0])
    # i == j pairs are scored -inf, so survivors are distinct pairs
    assert jnp.all(idx[:, 1] != idx[:, 2])


def test_make_triplets_z_informative_rejects_big_window():
    key = jr.PRNGKey(1)
    X, Z = _make_pool(key, n=100, p=3)
    with pytest.raises(ValueError):
        make_triplets_z_informative(key, X, Z, sig=1.0, n_triplets=5, k_window=100)
