"""Tests for the paper's two headline empirical claims.

- ``test_probit_calibration``: the closed-form probit P(T=1) = Phi(-s) matches the
  Monte-Carlo triplet-flip frequency under Gaussian latent noise (Theorem 1).
- ``test_recovery``: BII recovers a known heterogeneous w* on the Gaussian DGP.
"""

import numpy as np
from jax import numpy as jnp
from jax import random
from jax.scipy.special import ndtr

from bii import fit_bii
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


def test_recovery():
    """BII recovers a heterogeneous w* on the Gaussian design z = (x + eps)/sqrt(w*)."""
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
        compute_waic_flag=False,
    )
    w_mean = np.asarray(res["w_samples"]).reshape(-1, p).mean(axis=0)
    wt = np.asarray(w_true)
    cos = float(w_mean @ wt / (np.linalg.norm(w_mean) * np.linalg.norm(wt)))

    assert cos > 0.85, f"recovery cosine only {cos:.3f}"
    assert int(np.argmax(w_mean)) == int(np.argmax(wt)), "top weight not recovered"
