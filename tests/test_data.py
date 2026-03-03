"""Tests for data generation and triplet utilities."""

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from bii.data import T_from_X, make_data, make_iid


def test_T_from_X_binary():
    """T_from_X should produce only 0.0 and 1.0."""
    key = jr.PRNGKey(0)
    w0 = jnp.ones(3)
    X, Z = make_iid(key, n=50, p=3, sig=0.1 * jnp.ones(3), tau=1.0, w0=w0)
    T = T_from_X(X)
    assert T.shape == (50,)
    assert set(np.array(T).tolist()).issubset({0.0, 1.0})


def test_T_from_X_deterministic():
    """Same input should give same output."""
    key = jr.PRNGKey(1)
    w0 = jnp.ones(4)
    X, _ = make_iid(key, n=30, p=4, sig=0.1 * jnp.ones(4), tau=1.0, w0=w0)
    T1 = T_from_X(X)
    T2 = T_from_X(X)
    assert jnp.array_equal(T1, T2)


def test_T_from_X_convention():
    """T=1 means column 1 closer to column 0 (anchor) than column 2."""
    # Construct a triplet where we know the answer
    # anchor at origin, point 1 at distance 1, point 2 at distance 10
    X = jnp.array([
        [[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]],  # point 1 closer -> T=1
        [[0.0, 0.0], [10.0, 0.0], [1.0, 0.0]],  # point 2 closer -> T=0
    ])
    T = T_from_X(X)
    assert float(T[0]) == 1.0
    assert float(T[1]) == 0.0


def test_make_iid_shapes():
    """make_iid returns correct shapes."""
    key = jr.PRNGKey(2)
    p = 5
    n = 20
    X, Z = make_iid(key, n=n, p=p, sig=0.1 * jnp.ones(p), tau=1.0, w0=jnp.ones(p))
    assert X.shape == (n, 3, p)
    assert Z.shape == (n, 3, p)


def test_make_data_shapes():
    """make_data returns correct shapes."""
    key = jr.PRNGKey(3)
    p = 4
    n = 10
    X, Z = make_data(
        key, n_triplets=n, p=p, sig=0.1 * jnp.ones(p), tau=1.0,
        w0=jnp.ones(p), data_multiplier=20,
    )
    assert X.shape == (n, 3, p)
    assert Z.shape == (n, 3, p)


def test_make_data_with_indices():
    """make_data with return_indices=True returns index array."""
    key = jr.PRNGKey(4)
    p = 3
    n = 8
    X, Z, idx = make_data(
        key, n_triplets=n, p=p, sig=0.1 * jnp.ones(p), tau=1.0,
        w0=jnp.ones(p), data_multiplier=20, return_indices=True,
    )
    assert idx.shape == (n, 3)
    # All indices should be non-negative
    assert jnp.all(idx >= 0)


def test_make_data_insufficient_points_raises():
    """make_data should raise when not enough points for requested triplets."""
    key = jr.PRNGKey(5)
    p = 3
    with pytest.raises(ValueError):
        make_data(
            key, n_triplets=1000, p=p, sig=0.1 * jnp.ones(p), tau=1.0,
            w0=jnp.ones(p), data_multiplier=1,
        )
