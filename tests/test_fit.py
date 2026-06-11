"""Smoke tests for the unified fit_bii pipeline."""

import jax.numpy as jnp
import jax.random as jr

from bii.fit import fit_bii

EXPECTED_KEYS = {
    "w_samples",
    "raw_samples",
    "T",
    "Z",
    "triplet_indices",
    "kappa",
    "waic",
    "elapsed_seconds",
    "diagnostics",
}


def _make_pool(key, n=100, p=3, sig=0.1, tau=1.0):
    k1, k2 = jr.split(key)
    x_pool = jr.normal(k1, (n, p)) * tau
    eps = jr.normal(k2, (n, p)) * sig
    z_pool = x_pool + eps
    return x_pool, z_pool


def test_fit_smoke():
    key = jr.PRNGKey(0)
    x_pool, z_pool = _make_pool(key, n=100, p=3)

    result = fit_bii(
        key, x_pool, z_pool, sig=0.1,
        n_triplets=50,
        num_samples=50, num_warmup=50, num_chains=1,
    )

    assert set(result.keys()) >= EXPECTED_KEYS

    w = result["w_samples"]
    assert w.ndim == 3
    sums = jnp.sum(w, axis=-1)
    assert jnp.allclose(sums, 1.0, atol=1e-5)
    assert jnp.all(w >= 0)
    assert jnp.isfinite(result["waic"])


def test_fit_multiplicative_smoke():
    key = jr.PRNGKey(1)
    x_pool, z_pool = _make_pool(key, n=100, p=3)
    # Make z_pool positive for multiplicative model
    z_pool = jnp.abs(z_pool) + 1.0

    result = fit_bii(
        key, x_pool, z_pool, sig=0.3,
        noise_model="multiplicative",
        n_triplets=50,
        num_samples=50, num_warmup=50, num_chains=1,
    )

    assert set(result.keys()) >= EXPECTED_KEYS
    w = result["w_samples"]
    assert jnp.allclose(jnp.sum(w, axis=-1), 1.0, atol=1e-5)


def test_fit_with_zfar_sampler():
    """fit_bii accepts a custom 4-tuple triplet sampler."""
    import functools

    from bii.data import make_triplets_zfar

    key = jr.PRNGKey(2)
    x_pool, z_pool = _make_pool(key, n=200, p=3)
    sampler = functools.partial(make_triplets_zfar, rank_i=5, rank_j=15)
    result = fit_bii(
        key, x_pool, z_pool, sig=0.1,
        triplet_sampler=sampler, n_triplets=10, anchor_fraction=0.1,
        num_samples=30, num_warmup=30, num_chains=1,
    )
    assert result["triplet_weights"] is None
    assert jnp.allclose(jnp.sum(result["w_samples"], axis=-1), 1.0, atol=1e-5)


def test_fit_with_rank_weighted_sampler():
    """5-tuple sampler protocol: importance weights land in the result."""
    from bii.data import make_triplets_rank_weighted

    key = jr.PRNGKey(3)
    x_pool, z_pool = _make_pool(key, n=150, p=3)
    result = fit_bii(
        key, x_pool, z_pool, sig=0.1,
        triplet_sampler=make_triplets_rank_weighted,
        n_triplets=10, anchor_fraction=0.1,
        num_samples=30, num_warmup=30, num_chains=1,
    )
    assert result["triplet_weights"] is not None
    assert result["triplet_weights"].shape == result["T"].shape


def test_fit_with_clip_s():
    key = jr.PRNGKey(4)
    x_pool, z_pool = _make_pool(key, n=100, p=3)
    result = fit_bii(
        key, x_pool, z_pool, sig=0.1, clip_s=2.5,
        n_triplets=30, num_samples=30, num_warmup=30, num_chains=1,
    )
    assert jnp.isfinite(result["waic"])


def test_fit_with_pi_inclusion():
    """Fixed-pi mixture: inclusion_probs reported per triplet, in [0, 1]."""
    key = jr.PRNGKey(5)
    x_pool, z_pool = _make_pool(key, n=100, p=3)
    result = fit_bii(
        key, x_pool, z_pool, sig=0.1, pi_inclusion=0.8,
        n_triplets=30, num_samples=30, num_warmup=30, num_chains=1,
    )
    probs = result["inclusion_probs"]
    assert probs is not None
    assert probs.shape == result["T"].shape
    assert jnp.all(probs >= 0.0) and jnp.all(probs <= 1.0)


def test_fit_with_pi_prior():
    """Beta prior on pi: pi sampled jointly, reported with posterior mean."""
    key = jr.PRNGKey(6)
    x_pool, z_pool = _make_pool(key, n=100, p=3)
    num_samples, num_chains = 30, 2
    result = fit_bii(
        key, x_pool, z_pool, sig=0.1, pi_prior=(2.0, 2.0),
        n_triplets=30, num_samples=num_samples, num_warmup=30,
        num_chains=num_chains,
    )
    assert result["pi_samples"].shape == (num_samples, num_chains)
    assert jnp.all(result["pi_samples"] > 0.0)
    assert jnp.all(result["pi_samples"] < 1.0)
    assert 0.0 < result["pi_mean"] < 1.0
    # theta part still p-dimensional -> w on the simplex
    assert result["w_samples"].shape == (num_samples, num_chains, 3)
    assert jnp.allclose(jnp.sum(result["w_samples"], axis=-1), 1.0, atol=1e-5)
    assert result["inclusion_probs"] is not None


def test_fit_pi_prior_with_vi_raises():
    import pytest

    key = jr.PRNGKey(7)
    x_pool, z_pool = _make_pool(key, n=100, p=3)
    with pytest.raises(NotImplementedError):
        fit_bii(
            key, x_pool, z_pool, sig=0.1, pi_prior=(2.0, 2.0),
            inference_method="vi", n_triplets=30,
        )


def test_fit_unknown_inference_method_raises():
    import pytest

    key = jr.PRNGKey(8)
    x_pool, z_pool = _make_pool(key, n=100, p=3)
    with pytest.raises(ValueError):
        fit_bii(key, x_pool, z_pool, sig=0.1, inference_method="nope",
                n_triplets=30)


def test_fit_per_point_sig():
    """Pool-level per-point sigmas (N,) are resolved to per-triplet (n, 3).

    Regression test: triplet_accuracy used to break on pre-resolved sigmas
    whenever p != 3.
    """
    key = jr.PRNGKey(9)
    n_pool, p = 100, 4
    x_pool, z_pool = _make_pool(key, n=n_pool, p=p)
    sig_per_point = jnp.full(n_pool, 0.1)
    result = fit_bii(
        key, x_pool, z_pool, sig=sig_per_point,
        n_triplets=20, num_samples=20, num_warmup=20, num_chains=1,
    )
    assert result["w_samples"].shape[-1] == p
    assert jnp.isfinite(result["waic"])
    acc = result["alignment"]["triplet_accuracy"]
    assert jnp.all(acc >= 0.0) and jnp.all(acc <= 1.0)


def test_fit_with_logit_link():
    """fit_bii runs end-to-end with the logistic link; WAIC uses it too."""
    key = jr.PRNGKey(10)
    x_pool, z_pool = _make_pool(key, n=100, p=3)
    result = fit_bii(
        key, x_pool, z_pool, sig=0.1, link="logit",
        n_triplets=30, num_samples=30, num_warmup=30, num_chains=1,
    )
    assert jnp.allclose(jnp.sum(result["w_samples"], axis=-1), 1.0, atol=1e-5)
    assert jnp.isfinite(result["waic"])
    assert jnp.all(jnp.isfinite(result["alignment"]["alignment_index"]))


def test_fit_rejects_uninterpretable_sig():
    """fit_bii fails loudly on sigma shapes it cannot interpret."""
    import pytest

    key = jr.PRNGKey(11)
    x_pool, z_pool = _make_pool(key, n=100, p=3)
    with pytest.raises(ValueError, match="not interpretable"):
        # (p, p) full covariance — not supported
        fit_bii(key, x_pool, z_pool, sig=0.01 * jnp.eye(3), n_triplets=10)
    with pytest.raises(ValueError, match="not interpretable"):
        # 1-D vector that is neither (p,) nor (N,)
        fit_bii(key, x_pool, z_pool, sig=jnp.full(7, 0.1), n_triplets=10)


def test_fit_with_tau_prior():
    """Gamma prior on tau: sampled jointly, reported with posterior mean."""
    key = jr.PRNGKey(12)
    x_pool, z_pool = _make_pool(key, n=100, p=3)
    num_samples, num_chains = 30, 2
    result = fit_bii(
        key, x_pool, z_pool, sig=0.1, tau_prior=(2.0, 2.0),
        n_triplets=30, num_samples=num_samples, num_warmup=30,
        num_chains=num_chains,
    )
    assert result["tau_samples"].shape == (num_samples, num_chains)
    assert jnp.all(result["tau_samples"] > 0.0)
    assert result["tau_mean"] > 0.0
    assert result["w_samples"].shape == (num_samples, num_chains, 3)
    assert jnp.allclose(jnp.sum(result["w_samples"], axis=-1), 1.0, atol=1e-5)
    assert jnp.isfinite(result["waic"])


def test_fit_tau_prior_with_vi_raises():
    import pytest

    key = jr.PRNGKey(13)
    x_pool, z_pool = _make_pool(key, n=100, p=3)
    with pytest.raises(NotImplementedError):
        fit_bii(key, x_pool, z_pool, sig=0.1, tau_prior=(2.0, 2.0),
                inference_method="vi", n_triplets=30)
