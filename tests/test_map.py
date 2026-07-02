"""Tests for the MAP (posterior-mode) inference path."""

import jax
import jax.numpy as jnp
import jax.random as jr

from bii import fit_bii, make_dirichlet_logposterior, make_triplets, run_map


def _gaussian(key, n=500, p=8, sig=0.05):
    """Design where the weighted Z-metric at w* matches the X-metric."""
    kx, kw, ke = jr.split(key, 3)
    w = jr.uniform(kw, (p,), minval=0.3, maxval=4.0)
    w = w / w.sum()
    x = jr.normal(kx, (n, p))
    z = (x + sig * jr.normal(ke, (n, p))) / jnp.sqrt(w)
    return x, z, w


def _rmse(a, b):
    return float(jnp.sqrt(jnp.mean((a - b) ** 2)))


def test_run_map_recovers_wstar():
    key = jr.PRNGKey(0)
    x, z, w = _gaussian(key, p=8)
    t, _, zt, _ = make_triplets(key, x, z, n_triplets=15, anchor_fraction=0.5)
    logprob = make_dirichlet_logposterior(t, zt, sig=0.05, alpha=jnp.ones(8), kappa=1 / 15)
    pos, hist = run_map(key, logprob, dim=8, num_steps=1500)
    w_map = jax.nn.softmax(pos)
    assert _rmse(w_map, w) < 0.05
    assert float(hist[-1]) > float(hist[0])          # the log-posterior ascended
    assert jnp.allclose(w_map.sum(), 1.0, atol=1e-5)


def test_fit_bii_map_shape_and_diagnostics():
    key = jr.PRNGKey(1)
    x, z, w = _gaussian(key, p=6)
    r = fit_bii(key, x, z, sig=0.05, inference_method="map",
                n_triplets=15, map_steps=1000, compute_waic_flag=False)
    assert r["w_samples"].shape == (1, 1, 6)          # a single "draw" = the mode
    assert jnp.allclose(r["w_samples"].sum(-1), 1.0, atol=1e-5)
    assert jnp.all(r["w_samples"] >= 0)
    assert "logprob_history" in r["diagnostics"]
    assert _rmse(r["w_samples"][0, 0], w) < 0.05


def test_map_restarts_keep_best():
    key = jr.PRNGKey(2)
    x, z, w = _gaussian(key, p=6)
    t, _, zt, _ = make_triplets(key, x, z, n_triplets=15, anchor_fraction=0.5)
    logprob = make_dirichlet_logposterior(t, zt, sig=0.05, alpha=jnp.ones(6), kappa=1 / 15)
    pos, _ = run_map(key, logprob, dim=6, num_steps=800, n_restarts=3)
    assert _rmse(jax.nn.softmax(pos), w) < 0.06
