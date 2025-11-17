import numpy as np
import jax.numpy as jnp
import pytest

from examples.pair_data_utils import generate_toy_pair_data
from examples.fit_mle import run_mle
from examples.fit_mcmc import run_mcmc
from bii import make_loglikelihood

try:  # pragma: no cover - availability check
    import numpyro  # type: ignore
    HAS_NUMPYRO = True
except ImportError:  # pragma: no cover - availability check
    HAS_NUMPYRO = False


def test_generate_toy_pair_data_deterministic():
    data1 = generate_toy_pair_data(seed=0, n=80, row_share=0.3, num_shells=6)
    data2 = generate_toy_pair_data(seed=0, n=80, row_share=0.3, num_shells=6)
    np.testing.assert_array_equal(data1.pair_data.targets, data2.pair_data.targets)
    np.testing.assert_array_equal(data1.pair_data.dag_pairs, data2.pair_data.dag_pairs)


def test_mle_recovers_weights_close():
    toy = generate_toy_pair_data(seed=1, n=120, row_share=0.35, num_shells=8)
    w_hat = run_mle(
        toy.pair_data,
        toy.sigma_true,
        learning_rate=0.08,
        steps=1500,
    )
    assert np.allclose(np.array(w_hat), np.array(toy.w_true), atol=0.15)


@pytest.mark.skipif(not HAS_NUMPYRO, reason="numpyro required")
def test_mcmc_posterior_mean_close():
    toy = generate_toy_pair_data(seed=2, n=150, row_share=0.3, num_shells=8)
    samples = run_mcmc(
        toy.pair_data,
        toy.sigma_true,
        num_warmup=400,
        num_samples=400,
        seed=3,
    )
    mean_est = np.array(samples.mean(axis=0))
    assert np.allclose(mean_est, np.array(toy.w_true), atol=0.2)


def test_loglikelihood_monotone_between_random_weights():
    toy = generate_toy_pair_data(seed=4, n=100, row_share=0.3, num_shells=6)
    llik = make_loglikelihood(toy.pair_data, toy.sigma_true)
    w_true = toy.w_true
    w_random = jnp.array([0.2, 0.8])
    assert float(llik(w_true)) >= float(llik(w_random)) - 1e-3
