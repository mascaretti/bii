"""Smoke tests for the unified fit_bii pipeline."""

import jax.numpy as jnp
import jax.random as jr
import pytest

from bii.fit import fit_bii

EXPECTED_KEYS = {
    "w_samples",
    "raw_samples",
    "T",
    "triplet_indices",
    "prior",
    "kappa",
    "waic",
    "elapsed_seconds",
    "diagnostics",
}


def _make_pool(key, n=100, p=3, sig=0.1, tau=1.0):
    """Generate a small point-cloud pool for testing."""
    k1, k2 = jr.split(key)
    x_pool = jr.normal(k1, (n, p)) * tau
    eps = jr.normal(k2, (n, p)) * sig
    z_pool = x_pool + eps
    return x_pool, z_pool


@pytest.mark.parametrize("prior", ["dirichlet", "horseshoe"])
def test_fit_smoke(prior):
    key = jr.PRNGKey(0)
    x_pool, z_pool = _make_pool(key, n=100, p=3)

    result = fit_bii(
        key,
        x_pool,
        z_pool,
        sig=0.1,
        prior=prior,
        n_triplets=50,
        num_samples=50,
        num_warmup=50,
        num_chains=1,
    )

    # All expected keys present
    assert set(result.keys()) >= EXPECTED_KEYS

    # w on simplex
    w = result["w_samples"]
    assert w.ndim == 3  # (num_samples, num_chains, p)
    sums = jnp.sum(w, axis=-1)
    assert jnp.allclose(sums, 1.0, atol=1e-5)
    assert jnp.all(w >= 0)

    # WAIC finite
    assert jnp.isfinite(result["waic"])

    # Prior label round-trips
    assert result["prior"] == prior
