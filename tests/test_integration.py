"""End-to-end integration tests for fit_bii."""

import jax.numpy as jnp
import jax.random as jr

from bii.fit import fit_bii

EXPECTED_KEYS = {
    "w_samples", "raw_samples", "T", "Z", "triplet_indices",
    "prior", "kappa", "waic", "elapsed_seconds", "diagnostics",
}


def _make_pool(key, n=100, p=3, sig=0.1, w_star=None):
    k1, k2 = jr.split(key)
    X = jr.normal(k1, (n, p))
    if w_star is not None:
        X = X * jnp.sqrt(w_star)[None, :]
    Z = X + jr.normal(k2, (n, p)) * sig
    return X, Z


def test_end_to_end_dirichlet_nuts():
    key = jr.PRNGKey(42)
    X, Z = _make_pool(key, n=100, p=3, w_star=jnp.array([0.6, 0.3, 0.1]))
    result = fit_bii(
        key=key, X_pool=X, Z_pool=Z, sig=0.1,
        prior="dirichlet", n_triplets=50, anchor_fraction=0.3,
        num_samples=50, num_warmup=50, num_chains=1,
    )
    assert set(result.keys()) >= EXPECTED_KEYS
    w = result["w_samples"]
    assert jnp.allclose(jnp.sum(w, axis=-1), 1.0, atol=1e-5)
    assert jnp.isfinite(result["waic"])


def test_end_to_end_dirichlet_vi():
    key = jr.PRNGKey(42)
    X, Z = _make_pool(key, n=100, p=3, w_star=jnp.array([0.6, 0.3, 0.1]))
    result = fit_bii(
        key=key, X_pool=X, Z_pool=Z, sig=0.1,
        prior="dirichlet", n_triplets=50, anchor_fraction=0.3,
        inference_method="vi", vi_steps=500, vi_num_samples=100,
    )
    assert set(result.keys()) >= EXPECTED_KEYS
    w = result["w_samples"]
    assert jnp.allclose(jnp.sum(w, axis=-1), 1.0, atol=1e-5)
    assert jnp.isfinite(result["waic"])


def test_different_X_Z_dimensions():
    key = jr.PRNGKey(7)
    k1, k2, k3 = jr.split(key, 3)
    X = jr.normal(k1, (80, 3))
    Z = jr.normal(k2, (80, 5))
    result = fit_bii(
        key=k3, X_pool=X, Z_pool=Z, sig=0.1,
        n_triplets=30, num_samples=30, num_warmup=30, num_chains=1,
    )
    assert result["w_samples"].shape[-1] == 5


def test_kappa_affects_posterior():
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
    assert not jnp.allclose(results[0], results[1], atol=1e-3)
