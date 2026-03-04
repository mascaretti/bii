"""Tests for the sparse Dirichlet prior with Z-distributed concentrations."""

import jax
import jax.numpy as jnp
import jax.random as jr

from bii.sparse_dirichlet import (
    _log_z_density,
    log_sparse_dirichlet_posterior,
    sparse_dirichlet_dim,
    sparse_dirichlet_to_simplex,
)


def test_sparse_dirichlet_dim():
    assert sparse_dirichlet_dim(5) == 12
    assert sparse_dirichlet_dim(68) == 138


def test_sparse_dirichlet_to_simplex():
    p = 4
    position = jnp.zeros(sparse_dirichlet_dim(p))
    w = sparse_dirichlet_to_simplex(position)
    assert w.shape == (p,)
    assert jnp.allclose(jnp.sum(w), 1.0, atol=1e-6)
    assert jnp.allclose(w, 0.25, atol=1e-6)  # uniform when theta=0


def test_log_z_density_normalizes():
    """Check that the Z-density integrates to ~1 via numerical quadrature."""
    # Coarse grid integration for a = b = 0.5 (horseshoe case)
    z = jnp.linspace(-20, 20, 10000)
    dz = z[1] - z[0]
    log_p = jax.vmap(lambda zi: _log_z_density(zi, 0.5, 0.5, 0.0, 1.0))(z)
    integral = jnp.sum(jnp.exp(log_p)) * dz
    assert jnp.abs(integral - 1.0) < 0.01


def test_log_z_density_symmetric_horseshoe():
    """Z(1/2, 1/2) should be symmetric around 0."""
    z_pos = _log_z_density(2.0, 0.5, 0.5, 0.0, 1.0)
    z_neg = _log_z_density(-2.0, 0.5, 0.5, 0.0, 1.0)
    assert jnp.abs(z_pos - z_neg) < 1e-5


def test_log_z_density_location_shift():
    """Z(a, b, mu, sigma) shifts correctly with mu."""
    val_at_0 = _log_z_density(0.0, 0.5, 0.5, 0.0, 1.0)
    val_at_3 = _log_z_density(3.0, 0.5, 0.5, 3.0, 1.0)
    assert jnp.abs(val_at_0 - val_at_3) < 1e-5


def test_log_posterior_finite():
    """Log-posterior should be finite at a reasonable point."""
    p = 5
    key = jr.PRNGKey(42)
    k1, k2 = jr.split(key)

    # Small synthetic data
    N = 30
    X = jr.normal(k1, (N, p))
    Z = X + 0.1 * jr.normal(k2, (N, p))

    from bii.fit import _random_triplets

    T, Z_trip, _ = _random_triplets(jr.PRNGKey(0), X, Z, n_triplets=20)

    position = jnp.zeros(sparse_dirichlet_dim(p))
    lp = log_sparse_dirichlet_posterior(position, T, Z_trip, sig=0.1)
    assert jnp.isfinite(lp)


def test_log_posterior_gradient_finite():
    """Gradients should be finite (needed for NUTS)."""
    p = 4
    key = jr.PRNGKey(7)
    k1, k2 = jr.split(key)

    N = 20
    X = jr.normal(k1, (N, p))
    Z = X + 0.1 * jr.normal(k2, (N, p))

    from bii.fit import _random_triplets

    T, Z_trip, _ = _random_triplets(jr.PRNGKey(1), X, Z, n_triplets=15)

    position = jnp.zeros(sparse_dirichlet_dim(p))
    grad_fn = jax.grad(lambda pos: log_sparse_dirichlet_posterior(pos, T, Z_trip, sig=0.1))
    g = grad_fn(position)
    assert jnp.all(jnp.isfinite(g))
