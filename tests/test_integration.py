"""End-to-end integration tests for fit_bii."""

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from bii.fit import fit_bii


def _make_pool(key, n=100, p=3, sig=0.1, w_star=None):
    k1, k2 = jr.split(key)
    X = jr.normal(k1, (n, p))
    if w_star is not None:
        X = X * jnp.sqrt(w_star)[None, :]
    Z = X + jr.normal(k2, (n, p)) * sig
    return X, Z


EXPECTED_KEYS = {
    "w_samples", "raw_samples", "T", "triplet_indices",
    "prior", "kappa", "waic", "elapsed_seconds", "diagnostics",
}


@pytest.mark.parametrize("prior,method", [
    ("dirichlet", "nuts"),
    ("dirichlet", "vi"),
    ("horseshoe", "nuts"),
])
def test_end_to_end(prior, method):
    """Full pipeline: data → fit_bii → valid output."""
    key = jr.PRNGKey(42)
    w_star = jnp.array([0.6, 0.3, 0.1])
    X, Z = _make_pool(key, n=100, p=3, sig=0.1, w_star=w_star)

    kwargs = dict(
        key=key, X_pool=X, Z_pool=Z, sig=0.1,
        prior=prior, n_triplets=50, anchor_fraction=0.3,
    )
    if method == "nuts":
        kwargs.update(
            inference_method="nuts",
            num_samples=50, num_warmup=50, num_chains=1,
        )
    else:
        kwargs.update(
            inference_method="vi",
            vi_steps=500, vi_num_samples=100,
        )

    result = fit_bii(**kwargs)

    # All keys present
    assert set(result.keys()) >= EXPECTED_KEYS

    # w on simplex
    w = result["w_samples"]
    assert w.ndim == 3
    sums = jnp.sum(w, axis=-1)
    assert jnp.allclose(sums, 1.0, atol=1e-5)
    assert jnp.all(w >= 0)

    # WAIC finite
    assert jnp.isfinite(result["waic"])

    # Prior label
    assert result["prior"] == prior


def test_different_X_Z_dimensions():
    """X_pool and Z_pool can have different column counts."""
    key = jr.PRNGKey(7)
    k1, k2, k3 = jr.split(key, 3)
    n = 80
    p_x, p_z = 3, 5
    X = jr.normal(k1, (n, p_x))
    Z = jr.normal(k2, (n, p_z))

    result = fit_bii(
        key=k3, X_pool=X, Z_pool=Z, sig=0.1,
        n_triplets=30, num_samples=30, num_warmup=30, num_chains=1,
    )
    # Weights should have dimension p_z (from Z_pool)
    assert result["w_samples"].shape[-1] == p_z


def test_kappa_affects_posterior():
    """Different kappa values should produce different posteriors."""
    key = jr.PRNGKey(99)
    X, Z = _make_pool(key, n=100, p=3, sig=0.1)

    results = []
    for kappa in [0.1, 1.0]:
        r = fit_bii(
            key=key, X_pool=X, Z_pool=Z, sig=0.1,
            kappa=kappa, n_triplets=50,
            inference_method="vi", vi_steps=500, vi_num_samples=100,
        )
        results.append(r["w_samples"][:, 0, :].mean(0))

    # Different kappa → different posterior means
    assert not jnp.allclose(results[0], results[1], atol=1e-3)
