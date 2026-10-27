"""Tests for the paper's headline empirical claims.

- ``test_probit_calibration``: the closed-form probit P(T=1) = Phi(-s) matches the
  Monte-Carlo triplet-flip frequency under Gaussian latent noise (Theorem 1).
- ``test_recovery``: BII recovers a known heterogeneous w* on the Gaussian DGP,
  with and without the calibration slope; with it, the fitted slope is near one.
- ``test_null_does_not_collapse_with_slope_grid``: without signal, the calibrated
  posterior stays near the prior instead of collapsing onto one coordinate.
"""

import numpy as np
import pytest
from jax import numpy as jnp
from jax import random
from jax.scipy.special import ndtr

from bii import fit_bii, make_slope_grid
from bii.inference import delta_V_one_triplet


def test_probit_calibration():
    """Phi(-s) tracks the empirical P(Delta <= 0) when w is well spread."""
    p = 20
    sig = 0.5
    sig2 = sig**2
    S = 100_000
    key = random.key(0)
    w = jnp.ones(p) / p  # large effective support -> probit should be accurate

    max_err = 0.0
    for _ in range(6):
        key, ka, kb, kc = random.split(key, 4)
        zk = random.normal(ka, (p,))
        zi = zk + random.normal(kb, (p,))
        zj = zk + random.normal(kc, (p,))

        mu, V = delta_V_one_triplet(zi, zj, zk, w, sig2, sig2, sig2)
        pred = float(ndtr(-mu / jnp.sqrt(V)))  # P(T=1) = P(Delta <= 0)

        key, k1, k2, k3 = random.split(key, 4)
        yi = zi + sig * random.normal(k1, (S, p))
        yj = zj + sig * random.normal(k2, (S, p))
        yk = zk + sig * random.normal(k3, (S, p))
        delta = jnp.sum(w * (yi - yk) ** 2, axis=1) - jnp.sum(w * (yj - yk) ** 2, axis=1)
        emp = float(jnp.mean(delta <= 0.0))
        max_err = max(max_err, abs(emp - pred))

    assert max_err < 0.02, f"probit miscalibrated by {max_err:.4f}"


@pytest.mark.parametrize("calibrated", [False, True])
def test_recovery(calibrated):
    """BII recovers a heterogeneous w* on the Gaussian design z = (x + eps)/sqrt(w*).

    The data follow the model exactly, so with the calibration slope integrated
    out the fitted slope is near one.
    """
    p = 6
    n = 150
    sigma = 0.3
    w_true = jnp.array([0.40, 0.24, 0.16, 0.10, 0.07, 0.03])

    key = random.key(1)
    kx, ke, kf = random.split(key, 3)
    X = random.normal(kx, (n, p))
    Z = (X + sigma * random.normal(ke, (n, p))) / jnp.sqrt(w_true)
    sig = sigma / jnp.sqrt(w_true)  # per-feature noise std on Z

    res = fit_bii(
        kf, X, Z, sig,
        n_triplets=20, anchor_fraction=0.3,
        num_warmup=300, num_samples=200, num_chains=2,
        slope_grid=make_slope_grid() if calibrated else None,
        compute_waic_flag=False,
    )
    w_mean = np.asarray(res["w_samples"]).reshape(-1, p).mean(axis=0)
    wt = np.asarray(w_true)
    cos = float(w_mean @ wt / (np.linalg.norm(w_mean) * np.linalg.norm(wt)))

    assert cos > 0.85, f"recovery cosine only {cos:.3f}"
    assert int(np.argmax(w_mean)) == int(np.argmax(wt)), "top weight not recovered"
    if calibrated:
        assert 0.5 < res["slope_mean"] < 2.0, f"slope {res['slope_mean']:.3f} far from one"


def test_null_does_not_collapse_with_slope_grid():
    """Known answer: without signal the calibrated posterior stays near the prior.

    Three Gaussian source coordinates, the first with half the spread, and a
    target independent of the source. The plain probit collapses onto the
    first coordinate (E[w_0] ~ 1): the direction of w is its only way to lower
    its confidence. With the calibrated slope integrated out, the confidence
    no longer depends on that direction, so the posterior stays near the
    Dirichlet(1, 1, 1) prior and the fitted slope is small.
    """
    rng = np.random.default_rng(0)
    n = 300
    Z = (rng.standard_normal((n, 3)) * np.array([0.5, 1.0, 1.0])).astype(np.float32)
    X = rng.standard_normal((n, 1)).astype(np.float32)

    res = fit_bii(
        random.key(2), jnp.asarray(X), jnp.asarray(Z), 0.3,
        n_triplets=10, anchor_fraction=0.5,
        num_warmup=200, num_samples=200, num_chains=2,
        slope_grid=make_slope_grid(),
    )
    w_mean = np.asarray(res["w_samples"]).reshape(-1, 3).mean(axis=0)
    assert w_mean.max() < 0.6, f"collapsed: E[w] = {np.round(w_mean, 2)}"
    assert res["slope_mean"] < 0.3, f"slope {res['slope_mean']:.3f} is not small"
