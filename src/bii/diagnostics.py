"""Posterior diagnostics: WAIC, R-hat, ESS, alignment — all pure functions."""

import jax
import numpy as np
from jax import numpy as jnp
from scipy.special import ndtri
from scipy.stats import rankdata

from bii.inference import delta_V_one_triplet, loglik_w_per_triplet


def compute_waic(w_samples_flat, T, Z, sig, noise_model="additive", link="probit",
                 tau2=0.0):
    """Compute WAIC using lax.map to prevent OOM."""
    per_triplet_ll = jax.lax.map(
        lambda w: loglik_w_per_triplet(w, T, Z, sig, noise_model, link=link, tau2=tau2),
        w_samples_flat
    )

    S = per_triplet_ll.shape[0]
    lppd = jnp.sum(jax.scipy.special.logsumexp(per_triplet_ll, axis=0) - jnp.log(S))
    p_waic = jnp.sum(jnp.var(per_triplet_ll, axis=0))

    return -2.0 * (lppd - p_waic)

def _split_chains(x):
    """Split each chain in half: (n, m) -> (n // 2, 2m)."""
    n, m = x.shape
    n2 = n // 2
    return np.concatenate([x[:n2], x[n2:2 * n2]], axis=1)


def _rank_normalize(x):
    """Rank-normalise a (n, m) block to approximate normal scores (rankits)."""
    flat = x.reshape(-1)
    r = rankdata(flat)  # average ranks, 1..N
    z = ndtri((r - 0.375) / (flat.size - 0.25))
    return z.reshape(x.shape)


def _autocov(x):
    """Biased autocovariance of a 1-D series at lags 0..n-1 via FFT."""
    n = x.shape[0]
    x = x - x.mean()
    nfft = 1
    while nfft < 2 * n:
        nfft *= 2
    f = np.fft.rfft(x, nfft)
    acov = np.fft.irfft(f * np.conjugate(f), nfft)[:n].real
    return acov / n


def compute_rhat(samples):
    """Rank-normalised split-R-hat (Vehtari et al., 2021).

    Args:
        samples: (num_samples, num_chains, p).

    Returns:
        R-hat for each parameter (p,), as a numpy array.
    """
    s = np.asarray(samples, dtype=np.float64)
    _, _, p = s.shape
    out = np.empty(p, dtype=np.float64)
    for i in range(p):
        z = _rank_normalize(_split_chains(s[:, :, i]))  # (n2, 2m)
        n2, _ = z.shape
        cmean = z.mean(axis=0)
        W = z.var(axis=0, ddof=1).mean()
        B = n2 * cmean.var(ddof=1)
        var_plus = (n2 - 1) / n2 * W + B / n2
        out[i] = float(np.sqrt(var_plus / W)) if W > 0 else np.nan
    return out


def compute_ess(samples):
    """Bulk effective sample size: rank-normalised, split-chain, multi-lag.

    Uses Geyer's initial monotone positive sequence (the Stan/ArviZ estimator).

    Args:
        samples: (num_samples, num_chains, p).

    Returns:
        ESS for each parameter (p,), as a numpy array.
    """
    s = np.asarray(samples, dtype=np.float64)
    _, _, p = s.shape
    out = np.empty(p, dtype=np.float64)
    for i in range(p):
        x = _rank_normalize(_split_chains(s[:, :, i]))  # (n2, M), M = 2m
        n2, M = x.shape
        N = n2 * M
        acov = np.stack([_autocov(x[:, c]) for c in range(M)], axis=1)  # (n2, M)
        chain_var = acov[0] * n2 / (n2 - 1)                  # unbiased within-chain var
        W = chain_var.mean()
        B = n2 * x.mean(axis=0).var(ddof=1) if M > 1 else 0.0
        var_plus = (n2 - 1) / n2 * W + B / n2
        if var_plus <= 0:
            out[i] = float(N)
            continue
        rho = 1.0 - (W - acov.mean(axis=1)) / var_plus       # rho[0] == 1
        rho[0] = 1.0
        # Geyer initial positive sequence on consecutive pair sums.
        pair_sums = []
        k = 0
        while 2 * k + 1 < n2:
            pk = rho[2 * k] + rho[2 * k + 1]
            if k > 0 and pk <= 0.0:
                break
            pair_sums.append(pk)
            k += 1
        pair_sums = np.array(pair_sums)
        # Enforce the initial monotone (non-increasing) sequence.
        for j in range(1, pair_sums.size):
            if pair_sums[j] > pair_sums[j - 1]:
                pair_sums[j] = pair_sums[j - 1]
        tau = max(-1.0 + 2.0 * pair_sums.sum(), 1.0 / np.log10(max(N, 10)))
        out[i] = N / tau
    return out


def weight_entropy(w_samples):
    """Normalized entropy of weight vectors on the simplex.

    Measures concentration of the weight vector:
    - 0 = uniform (no alignment / all nutrients equally important)
    - 1 = point mass on one nutrient (perfect alignment)

    Args:
        w_samples: (S, p) posterior draws on the simplex.

    Returns:
        (S,) normalized alignment scores in [0, 1].
    """
    p = w_samples.shape[1]
    # Clip to avoid log(0)
    w_safe = jnp.clip(w_samples, 1e-30, None)
    H = -jnp.sum(w_safe * jnp.log(w_safe), axis=1)
    return 1.0 - H / jnp.log(p)


def triplet_accuracy(w_samples, T, Z, sig, noise_model="additive"):
    """Triplet prediction accuracy for each posterior sample.

    Args:
        w_samples: (S, p) posterior draws on the simplex.
        T: (n,) binary triplet labels.
        Z: (n, 3, p) triplet embeddings.
        sig: noise std — scalar or (p,).
        noise_model: ``"additive"`` or ``"multiplicative"``.

    Returns:
        (S,) accuracy values in [0, 1].
    """
    from bii.inference import _make_sig2_fn, _resolve_sig2
    zi, zj, zk = Z[:, 1], Z[:, 2], Z[:, 0]
    sig = jnp.asarray(sig)

    if sig.ndim >= 2:
        # Pre-resolved per-triplet sigmas — (n, 3) or (n, 3, p)
        sig2_i, sig2_j, sig2_k = _resolve_sig2(sig, noise_model, zi, zj, zk)

        def delta_fn(w):
            def dv(zi, zj, zk, s2i, s2j, s2k):
                return delta_V_one_triplet(zi, zj, zk, w, s2i, s2j, s2k)
            delta, _V = jax.vmap(dv)(zi, zj, zk, sig2_i, sig2_j, sig2_k)
            return delta
    else:
        sig2_fn = _make_sig2_fn(sig, noise_model)

        def delta_fn(w):
            def dv(zi, zj, zk):
                return delta_V_one_triplet(zi, zj, zk, w,
                                           sig2_fn(zi), sig2_fn(zj), sig2_fn(zk))
            delta, _V = jax.vmap(dv)(zi, zj, zk)
            return delta

    def accuracy_one(w):
        pred = (delta_fn(w) <= 0.0).astype(jnp.float32)
        return jnp.mean(pred == T)

    return jax.lax.map(accuracy_one, w_samples)


def alignment_index(w_samples, T, Z, sig, noise_model="additive", link="probit",
                    tau2=0.0):
    """Normalised cross-entropy alignment index.

    Maps the mean per-triplet log-likelihood to [0, 1]:
      Δ(w) = 1 + ℓ̄(w) / log(2)

    Args:
        w_samples: (S, p) posterior draws on the simplex.
        T: (n,) binary triplet labels.
        Z: (n, 3, p) triplet embeddings.
        sig: noise std — scalar or (p,).
        noise_model: ``"additive"`` or ``"multiplicative"``.

    Returns:
        (S,) alignment index values in [0, 1].
    """
    def delta_one(w):
        ll = loglik_w_per_triplet(w, T, Z, sig, noise_model, link=link, tau2=tau2)
        mean_ll = jnp.mean(ll)
        return 1.0 + mean_ll / jnp.log(2.0)

    return jax.lax.map(delta_one, w_samples)
