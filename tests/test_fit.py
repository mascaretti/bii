import importlib.metadata

import jax
import numpy as np
import pytest
from jax import numpy as jnp
from jax import random

from bii import fit_bii, make_slope_grid

SLOPE_FIT = dict(n_triplets=5, kappa=0.2, slope_grid=make_slope_grid(num=17), num_warmup=100,
                 num_samples=100, num_chains=2, compute_waic_flag=False)


def pools(key, w_true, sigma, n=60):
    """Target stretched by the true weights; source = clean source plus noise."""
    k1, k2 = random.split(key)
    zc = random.normal(k1, (n, len(w_true)))
    return zc * jnp.sqrt(jnp.asarray(w_true)), zc + sigma * random.normal(k2, zc.shape)


@pytest.fixture(scope="module")
def signal():
    # per point and feature noise, as in the NHANES run
    x, z = pools(random.PRNGKey(0), [0.8, 0.15, 0.05], 0.1)
    return fit_bii(random.PRNGKey(1), x, z, jnp.full((60, 3), 0.1), **SLOPE_FIT)


@pytest.fixture(scope="module")
def map_fit():
    x, z = pools(random.PRNGKey(3), [0.7, 0.2, 0.1], 0.1)
    return fit_bii(random.PRNGKey(4), x, z, 0.1, n_triplets=5, kappa=0.2,
                   inference_method="map", map_steps=300)


def test_fit_finds_the_signal(signal):
    w = np.asarray(signal["w_samples"])
    assert w.shape == (100, 2, 3) and np.allclose(w.sum(axis=-1), 1, atol=1e-5)
    assert np.abs(w.reshape(-1, 3).mean(axis=0) - [0.8, 0.15, 0.05]).max() < 0.1
    assert 0.5 < signal["slope_mean"] < 1.5
    assert signal["n_aligned"].shape == (200,)
    assert np.all(signal["diagnostics"]["rhat"] < 1.1)


def test_fit_without_signal_returns_the_prior():
    # shuffled target: no coordinate dominates, and the slope says the labels are noise
    x, z = pools(random.PRNGKey(0), [0.8, 0.15, 0.05], 0.1)
    x = x[random.permutation(random.PRNGKey(2), 60)]
    res = fit_bii(random.PRNGKey(1), x, z, 0.1, **SLOPE_FIT)
    assert res["w_samples"].reshape(-1, 3).mean(axis=0).max() < 0.6
    assert res["slope_mean"] < 0.3


def test_fit_map_returns_one_draw(map_fit):
    assert map_fit["w_samples"].shape == (1, 1, 3) and map_fit["raw_samples"].shape == (1, 1, 3)
    assert int(np.argmax(map_fit["w_samples"])) == 0 and np.isfinite(map_fit["waic"])


def reference_stack():
    versions = {d: importlib.metadata.version(d) for d in ("jax", "blackjax", "optax")}
    return (versions == {"jax": "0.8.2", "blackjax": "1.7.1", "optax": "0.2.8"}
            and jax.default_backend() == "cpu")


@pytest.mark.skipif(not reference_stack(),
                    reason="exact numbers hold on jax 0.8.2, blackjax 1.7.1, optax 0.2.8, CPU")
def test_reproduces_the_reference_numbers(signal, map_fit):
    # The same fits with the bii commit that the experiments pin (aa921d9): this code
    # must give the same numbers, bit for bit, on the same stack.
    assert np.asarray(signal["w_samples"]).mean(axis=(0, 1)).tolist() == [
        0.8064355254173279, 0.11976534128189087, 0.07379871606826782]
    assert signal["slope_mean"] == 0.9332855939865112
    assert np.asarray(map_fit["w_samples"]).ravel().tolist() == [
        0.7084935903549194, 0.18834052979946136, 0.10316580533981323]
    assert float(map_fit["waic"]) == 32.90138244628906
