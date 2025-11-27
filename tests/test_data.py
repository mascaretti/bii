import jax
import jax.numpy as jnp
from bii.data import generate_observations


def test_generate_observations_shapes_and_determinism():
    key = jax.random.PRNGKey(0)
    n, p = 5, 3
    w = jnp.array([0.2, 0.5, 1.3])
    sigma = jnp.array([0.1, 0.2, 0.3])

    X1, Z1 = generate_observations(key, n, p, w, sigma)
    X2, Z2 = generate_observations(key, n, p, w, sigma)

    assert X1.shape == (n, p)
    assert Z1.shape == (n, p)
    assert jnp.allclose(X1, X2)
    assert jnp.allclose(Z1, Z2)


def test_generate_observations_invalid_inputs():
    key = jax.random.PRNGKey(0)
    w = jnp.array([1.0, -0.1])
    sigma = jnp.array([0.1, 0.1])
    try:
        generate_observations(key, 3, 2, w, sigma)
    except ValueError:
        return
    assert False, "Expected ValueError for negative weights"
