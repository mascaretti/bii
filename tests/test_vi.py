"""Tests for mean-field variational inference."""

import jax.numpy as jnp
import jax.random as jr

from bii.fit import fit_bii


def _make_pool(key, n=100, p=3, sig=0.1, tau=1.0, w_star=None):
    """Generate a point-cloud pool with known true weights."""
    k1, k2 = jr.split(key)
    x_pool = jr.normal(k1, (n, p)) * tau
    if w_star is not None:
        # Scale dimensions so that w_star reflects true metric weights
        x_pool = x_pool * jnp.sqrt(w_star)[None, :]
    eps = jr.normal(k2, (n, p)) * sig
    z_pool = x_pool + eps
    return x_pool, z_pool


def test_vi_smoke():
    """VI runs, returns correct keys, w on simplex, WAIC finite."""
    key = jr.PRNGKey(0)
    x_pool, z_pool = _make_pool(key, n=100, p=3)

    result = fit_bii(
        key, x_pool, z_pool, sig=0.1,
        inference_method="vi",
        n_triplets=50,
        vi_steps=500,
        vi_num_samples=100,
    )

    # Expected keys present
    assert "w_samples" in result
    assert "diagnostics" in result
    assert "waic" in result

    # VI-specific diagnostics
    diag = result["diagnostics"]
    assert "elbo_history" in diag
    assert "final_elbo" in diag
    assert "mu" in diag
    assert "log_sigma" in diag

    # w on simplex, shape (S, 1, p)
    w = result["w_samples"]
    assert w.ndim == 3
    assert w.shape[1] == 1  # single "chain" for VI
    sums = jnp.sum(w, axis=-1)
    assert jnp.allclose(sums, 1.0, atol=1e-5)
    assert jnp.all(w >= 0)

    # WAIC finite
    assert jnp.isfinite(result["waic"])


def test_vi_recovers_weights():
    """VI posterior mean is close to known true weights."""
    key = jr.PRNGKey(42)
    w_star = jnp.array([0.7, 0.2, 0.1])
    x_pool, z_pool = _make_pool(key, n=500, p=3, sig=0.05, w_star=w_star)

    result = fit_bii(
        key, x_pool, z_pool, sig=0.05,
        inference_method="vi",
        n_triplets=20,
        anchor_fraction=0.5,
        vi_steps=5000,
        vi_num_samples=2000,
    )

    w_mean = result["w_samples"][:, 0, :].mean(0)
    assert jnp.max(jnp.abs(w_mean - w_star)) < 0.15


def test_vi_vs_nuts_agreement():
    """VI and NUTS posterior means agree on a simple problem."""
    key = jr.PRNGKey(7)
    x_pool, z_pool = _make_pool(key, n=100, p=3, sig=0.1)

    result_vi = fit_bii(
        key, x_pool, z_pool, sig=0.1,
        inference_method="vi",
        n_triplets=100,
        vi_steps=3000,
        vi_num_samples=2000,
    )
    result_nuts = fit_bii(
        key, x_pool, z_pool, sig=0.1,
        inference_method="nuts",
        n_triplets=100,
        num_samples=200,
        num_warmup=200,
        num_chains=1,
    )

    w_vi = result_vi["w_samples"][:, 0, :].mean(0)
    w_nuts = result_nuts["w_samples"][:, 0, :].mean(0)
    assert jnp.max(jnp.abs(w_vi - w_nuts)) < 0.15


def test_vi_elbo_increases():
    """ELBO should increase over the course of optimization."""
    key = jr.PRNGKey(99)
    x_pool, z_pool = _make_pool(key, n=100, p=3)

    result = fit_bii(
        key, x_pool, z_pool, sig=0.1,
        inference_method="vi",
        n_triplets=50,
        vi_steps=1000,
        vi_num_samples=100,
    )

    elbo = result["diagnostics"]["elbo_history"]
    assert elbo[-1] > elbo[0]
