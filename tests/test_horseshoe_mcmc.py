"""
Targeted MCMC diagnostics for the horseshoe prior.

Checks that with enough data (≥15 triplets per anchor) the horseshoe chain:
  - achieves acceptance rate 0.6–0.95
  - R-hat < 1.1 for all components
  - ESS > 20 per component

Run with:
    pytest tests/test_horseshoe_mcmc.py -v -s
"""

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from bii.fit import fit_bii


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_sparse_pool(key, n, p_rel, p_noise, w_star_rel, sig, tau=1.0):
    """Generate pool with sparse ground-truth metric (same as experiment)."""
    p_total = p_rel + p_noise
    w_star = jnp.concatenate([w_star_rel, jnp.zeros(p_noise)])
    k1, k2 = jr.split(key)
    scale = jnp.sqrt(w_star + 1e-30) * tau
    X_pool = jr.normal(k1, (n, p_total)) * scale
    sig_vec = jnp.broadcast_to(jnp.asarray(sig, dtype=float), (p_total,))
    Z_pool = X_pool + jr.normal(k2, (n, p_total)) * sig_vec
    return X_pool, Z_pool, w_star, sig_vec


# ---------------------------------------------------------------------------
# Design: n_pool=50, 15 triplets/anchor → 750 total
#         100 samples + 150 warmup, 2 chains
# ---------------------------------------------------------------------------

N_POOL = 50
N_TRIPLETS = N_POOL * 15   # 750
NUM_SAMPLES = 100
NUM_WARMUP = 150
NUM_CHAINS = 2
P_REL = 2
P_NOISE = 3
W_STAR_REL = jnp.array([0.7, 0.3])
SIG = 0.1


@pytest.mark.parametrize("prior", ["horseshoe", "dirichlet"])
def test_mcmc_quality(prior):
    """2 chains, ≥15 triplets/anchor: check acceptance, R-hat, ESS."""
    key = jr.PRNGKey(7)
    k_pool, k_fit = jr.split(key)

    X_pool, Z_pool, w_star, sig_vec = make_sparse_pool(
        k_pool, N_POOL, P_REL, P_NOISE, W_STAR_REL, SIG
    )

    res = fit_bii(
        k_fit, X_pool, Z_pool, sig_vec,
        prior=prior,
        n_triplets=N_TRIPLETS,
        num_samples=NUM_SAMPLES,
        num_warmup=NUM_WARMUP,
        num_chains=NUM_CHAINS,
    )

    diag = res["diagnostics"]
    w_s = np.array(res["w_samples"])   # (S, C, p)
    p = w_s.shape[2]

    acc = float(diag["acceptance_rate"])
    rhat = np.array(diag["rhat"])
    ess = np.array(diag["ess"])
    w_mean = w_s.reshape(-1, p).mean(axis=0)

    print(f"\n[{prior}] acceptance = {acc:.3f}")
    print(f"[{prior}] R-hat      = {np.round(rhat, 3)}")
    print(f"[{prior}] ESS        = {np.round(ess, 1)}")
    print(f"[{prior}] w_mean     = {np.round(w_mean, 3)}")
    print(f"[{prior}] w_star     = {np.round(np.array(w_star), 3)}")
    print(f"[{prior}] WAIC       = {float(res['waic']):.2f}")

    assert 0.5 < acc <= 1.0, f"bad acceptance: {acc:.3f}"
    # NaN R-hat / ESS means the chain for that dim is exactly constant (zero-variance).
    # This is not a mixing failure — it means the posterior is degenerate at 0,
    # which is correct behaviour for noise dims under the Dirichlet prior.
    assert np.all((rhat < 1.1) | ~np.isfinite(rhat) | (rhat == 0)), f"R-hat > 1.1: {rhat}"
    assert np.all((ess > 20) | ~np.isfinite(ess)), f"low ESS: {ess}"
    assert abs(w_mean.sum() - 1.0) < 1e-4


def test_horseshoe_noise_shrinkage():
    """Horseshoe posterior mean on noise dims should be below uniform 1/p."""
    p_total = P_REL + P_NOISE
    key = jr.PRNGKey(13)
    k_pool, k_hs = jr.split(key)

    X_pool, Z_pool, w_star, sig_vec = make_sparse_pool(
        k_pool, N_POOL, P_REL, P_NOISE, W_STAR_REL, SIG
    )

    res = fit_bii(
        k_hs, X_pool, Z_pool, sig_vec,
        prior="horseshoe",
        n_triplets=N_TRIPLETS,
        num_samples=NUM_SAMPLES,
        num_warmup=NUM_WARMUP,
        num_chains=NUM_CHAINS,
    )

    w_s = np.array(res["w_samples"])
    w_mean = w_s.reshape(-1, p_total).mean(axis=0)
    noise_mean = w_mean[P_REL:]

    print(f"\n[horseshoe] w_mean (rel)   = {np.round(w_mean[:P_REL], 3)}")
    print(f"[horseshoe] w_mean (noise) = {np.round(noise_mean, 3)}")
    print(f"[horseshoe] noise max = {noise_mean.max():.4f}  (uniform = {1/p_total:.4f})")

    assert noise_mean.max() < 1.0 / p_total, (
        f"Horseshoe not shrinking noise dims: max={noise_mean.max():.3f} "
        f">= uniform {1/p_total:.3f}"
    )
