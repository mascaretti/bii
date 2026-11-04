import numpy as np
from jax import numpy as jnp
from jax import random

from bii import run_map, run_nuts

MEAN = jnp.array([1.0, -2.0])
SD = jnp.array([0.5, 2.0])


def gaussian(x):
    return -0.5 * jnp.sum(((x - MEAN) / SD) ** 2)


def test_nuts_recovers_a_gaussian():
    draws, acc = run_nuts(random.PRNGKey(0), gaussian, jnp.zeros(2),
                          num_samples=500, num_warmup=300, num_chains=2)
    x = np.asarray(draws).reshape(-1, 2)
    assert draws.shape == (500, 2, 2) and acc.shape == (500, 2)
    assert np.all(np.abs(x.mean(axis=0) - MEAN) < 0.25 * SD)
    assert np.all(np.abs(x.std(axis=0) / SD - 1) < 0.15)


def test_map_finds_the_mode():
    mode, history = run_map(random.PRNGKey(0), gaussian, 2, num_steps=600, lr=0.05,
                            n_restarts=3)
    assert np.allclose(mode, MEAN, atol=1e-3)
    assert history.shape == (600,) and history[-1] >= history[0]
