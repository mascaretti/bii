"""Thin composition layer: triplets -> logposterior -> sampling -> diagnostics."""

import time

import jax
from jax import numpy as jnp
from jax import random

from bii.data import make_triplets
from bii.diagnostics import (
    compute_ess,
    compute_rhat,
    compute_waic,
    triplet_accuracy,
    weight_entropy,
)
from bii.priors import make_dirichlet_logposterior
from bii.sampling import run_nuts, run_vi, sample_vi


def fit_bii(
    key,
    X_pool,
    Z_pool,
    sig,
    noise_model="additive",
    n_triplets=15,
    anchor_fraction=0.5,
    # Triplet construction
    triplet_sampler=None,
    # Prior hyperparams
    alpha=None,
    kappa=1.0,
    # Likelihood robustifier
    clip_s=None,
    # Link function for the triplet probability
    link="probit",
    # Inclusion-mixture likelihood (probabilistic triplet inclusion that
    # evolves with w; set to a float in (0, 1) to enable)
    pi_inclusion=None,
    # Optional Beta(a, b) prior on pi (overrides pi_inclusion when given)
    pi_prior=None,
    # Inference method
    inference_method="nuts",
    # NUTS params
    num_samples=1000,
    num_warmup=500,
    num_chains=4,
    step_size=1e-3,
    target_acceptance_rate=0.8,
    # VI params
    vi_steps=5000,
    vi_lr=1e-2,
    vi_elbo_samples=8,
    vi_num_samples=2000,
    # Options
    compute_waic_flag=True,
):
    """Unified Bayesian inference pipeline for metric weights.

    Args:
        key: JAX random key.
        X_pool: (N, p_x) clean embeddings.
        Z_pool: (N, p_z) noisy/normalised embeddings.
        sig: noise std — scalar, per-feature (p,), or pool-level per-point
            (N,) / per-point-diagonal (N, p), which are resolved to
            per-triplet sigmas via the triplet indices. Full (non-diagonal)
            covariance matrices are not supported.
        noise_model: ``"additive"`` or ``"multiplicative"``.
        n_triplets: destination pairs per anchor.
        anchor_fraction: fraction of pool used as anchors.
        triplet_sampler: callable, default ``bii.data.make_triplets`` (ignores
            ``sig``). Signature
            ``(key, X_pool, Z_pool, sig, n_triplets, anchor_fraction) -> ...``.
            Two return protocols are supported:
              * 4-tuple ``(T, X, Z, indices)`` — unweighted loglik.
              * 5-tuple ``(T, X, Z, indices, weights)`` — ``weights`` are
                forwarded to the loglik as per-triplet importance weights.
            Pass e.g. ``functools.partial(make_triplets_zfar, rank_i=10, rank_j=25)``
            for the Z-far sampler, or ``make_triplets_rank_weighted`` for the
            importance-weighted rank-pair sampler.
        alpha: Dirichlet concentration; default ``ones(p)``.
        kappa: power-likelihood correction.
        clip_s: optional float. Clips the per-triplet probit statistic
            ``s = delta / sqrt(V)`` to ``[-clip_s, clip_s]`` inside the
            log-likelihood. Bounded-influence (censored-probit) robustifier
            for saturating triplets; ``clip_s=2.5`` is a sensible default
            ("any confidence stronger than ~99.4% is treated as 99.4%").
            Default ``None`` = no clipping.
        link: ``"probit"`` (normal CDF, default) or ``"logit"``
            (slope-matched logistic CDF). The logistic link has log-linear
            tails ``~ -1.702 |s|`` instead of the probit's ``~ -s^2/2``,
            matching the sub-exponential tails of the exact
            distance-difference statistic; use it when single coordinates
            dominate the metric (anisotropic scales, heavy tails), where
            the Gaussian shape approximation of the probit is least valid.
        pi_inclusion: optional float in (0, 1). Enables an inclusion-mixture
            likelihood: each triplet is treated as "informative" with prob
            ``pi`` and "noise" (label uniform) with prob ``1 - pi``. The
            posterior inclusion probability ``P(m_t = 1 | T_t, w)`` is
            reported in the result dict as ``inclusion_probs`` (mean over
            NUTS draws). Adaptive analogue of the static ``[eps, 1-eps]``
            filter from :func:`bii.data.make_triplets_random_sparse`.
            Default ``None`` = plain probit (no mixture).
        pi_prior: optional ``(a, b)`` Beta hyperparameters that put a
            ``Beta(a, b)`` prior on ``pi`` and sample it jointly with
            ``theta``. When given, ``pi_inclusion`` is ignored and the
            result dict includes a new field ``pi_samples`` of shape
            ``(num_samples, num_chains)`` plus a posterior-mean
            ``pi_mean``. Use ``Beta(2, 2)`` for a weakly informative
            default that keeps mass away from 0 and 1.
        inference_method: ``"nuts"`` or ``"vi"``.
        num_samples: posterior draws per chain (NUTS).
        num_warmup: NUTS warmup steps.
        num_chains: number of MCMC chains.
        step_size: initial NUTS step size.
        target_acceptance_rate: NUTS target acceptance.
        vi_steps: VI optimization steps.
        vi_lr: Adam learning rate for VI.
        vi_elbo_samples: MC samples per ELBO estimate.
        vi_num_samples: samples drawn from fitted variational posterior.

    Returns:
        dict with keys ``w_samples``, ``raw_samples``, ``T``, ``Z``,
        ``triplet_indices``, ``kappa``, ``waic``,
        ``elapsed_seconds``, ``diagnostics``.
    """
    t0 = time.perf_counter()
    p = Z_pool.shape[1]

    if alpha is None:
        alpha = jnp.ones(p)

    # Step 1 — form triplets
    key, key_trip = random.split(key)
    triplet_weights = None
    if triplet_sampler is None:
        T, X, Z, indices = make_triplets(key_trip, X_pool, Z_pool, n_triplets, anchor_fraction)
    else:
        out = triplet_sampler(key_trip, X_pool, Z_pool, sig, n_triplets, anchor_fraction)
        if len(out) == 5:
            T, X, Z, indices, triplet_weights = out
        else:
            T, X, Z, indices = out

    # Step 1b — resolve per-point sigmas to per-triplet if needed
    sig_arr = jnp.asarray(sig)
    N = X_pool.shape[0]
    if sig_arr.ndim == 1 and sig_arr.shape[0] == N:
        # Pool-level per-point sigmas (N,) -> triplet-level (n_triplets, 3)
        sig_resolved = sig_arr[indices]
    elif sig_arr.ndim == 2 and sig_arr.shape == (N, p):
        # Pool-level per-point diagonal sigmas (N, p) -> (n_triplets, 3, p)
        sig_resolved = sig_arr[indices]
    elif sig_arr.ndim == 0 or (sig_arr.ndim == 1 and sig_arr.shape[0] == p):
        sig_resolved = sig
    else:
        raise ValueError(
            f"sig of shape {sig_arr.shape} is not interpretable: expected a "
            f"scalar, per-feature (p,) = ({p},), per-point (N,) = ({N},), or "
            f"per-point diagonal (N, p) = ({N}, {p}). Full (non-diagonal) "
            f"covariance matrices are not supported."
        )

    # Step 2 — build log-posterior. With pi_prior, the position vector grows
    # by one entry (logit_pi at the end), and pi_inclusion is ignored.
    logprob_fn = make_dirichlet_logposterior(
        T, Z, sig_resolved, alpha, kappa, noise_model,
        triplet_weights=triplet_weights, clip_s=clip_s,
        pi_inclusion=pi_inclusion, pi_prior=pi_prior, link=link,
    )
    position_dim = p + 1 if pi_prior is not None else p
    init_position = jnp.zeros(position_dim)

    # Step 3 — run inference
    if inference_method == "nuts":
        key, key_nuts = random.split(key)
        raw_samples, acceptance_rates = run_nuts(
            key_nuts, logprob_fn, init_position,
            num_samples, num_warmup, num_chains, step_size, target_acceptance_rate,
        )

        if pi_prior is not None:
            theta_samples = raw_samples[..., :p]
            logit_pi_samples = raw_samples[..., p]
            pi_samples = jax.nn.sigmoid(logit_pi_samples)
        else:
            theta_samples = raw_samples
            pi_samples = None
        w_samples = jax.vmap(jax.vmap(jax.nn.softmax))(theta_samples)

        diagnostics = {
            "acceptance_rate": jnp.mean(acceptance_rates),
            "acceptance_rate_per_chain": jnp.mean(acceptance_rates, axis=0),
        }
        if num_chains > 1:
            diagnostics["rhat"] = compute_rhat(w_samples)
            diagnostics["ess"] = compute_ess(w_samples)

    elif inference_method == "vi":
        if pi_prior is not None:
            raise NotImplementedError(
                "VI with a Beta prior on pi is not yet supported; use NUTS."
            )
        key, key_vi, key_sample = random.split(key, 3)

        mu, log_sigma, elbo_history = run_vi(
            key_vi, logprob_fn, p,
            num_steps=vi_steps, lr=vi_lr, num_elbo_samples=vi_elbo_samples,
        )
        theta_samples, w_flat_vi = sample_vi(key_sample, mu, log_sigma, vi_num_samples)

        raw_samples = theta_samples[:, None, :]
        w_samples = w_flat_vi[:, None, :]
        pi_samples = None

        diagnostics = {
            "elbo_history": elbo_history,
            "final_elbo": elbo_history[-1],
            "mu": mu,
            "log_sigma": log_sigma,
        }
    else:
        raise ValueError(f"Unknown inference_method: {inference_method!r}")

    # WAIC (optional — can OOM with large triplet sets + many samples)
    w_flat = w_samples.reshape(-1, p)
    waic = (compute_waic(w_flat, T, Z, sig_resolved, noise_model, link=link)
            if compute_waic_flag else None)

    # Posterior-mean inclusion probability per triplet (only meaningful when
    # the mixture likelihood is in use; otherwise we skip the computation).
    # With a Beta prior on pi we use the posterior-mean pi as the reference.
    incl_probs = None
    pi_mean = None
    if pi_prior is not None and pi_samples is not None:
        pi_mean = float(jnp.mean(pi_samples))
    pi_for_incl = pi_mean if pi_mean is not None else pi_inclusion
    if pi_for_incl is not None:
        from bii.inference import inclusion_probs as _inclusion_probs
        sub = w_flat[:: max(1, w_flat.shape[0] // 512)]
        per_draw = jax.vmap(
            lambda w_: _inclusion_probs(w_, T, Z, sig_resolved, pi_for_incl,
                                        noise_model, clip_s, link=link)
        )(sub)
        incl_probs = jnp.mean(per_draw, axis=0)

    # Alignment measures
    from bii.diagnostics import alignment_index as _alignment_index
    entropy_scores = weight_entropy(w_flat)
    accuracy_scores = triplet_accuracy(w_flat, T, Z, sig_resolved, noise_model)
    alignment_idx = _alignment_index(w_flat, T, Z, sig_resolved, noise_model, link=link)

    elapsed = time.perf_counter() - t0

    return {
        "w_samples": w_samples,
        "raw_samples": raw_samples,
        "T": T,
        "Z": Z,
        "triplet_indices": indices,
        "triplet_weights": triplet_weights,
        "inclusion_probs": incl_probs,
        "pi_inclusion": pi_inclusion,
        "pi_prior": pi_prior,
        "pi_samples": pi_samples,
        "pi_mean": pi_mean,
        "kappa": kappa,
        "waic": waic,
        "alignment": {
            "entropy": entropy_scores,
            "triplet_accuracy": accuracy_scores,
            "alignment_index": alignment_idx,
        },
        "elapsed_seconds": elapsed,
        "diagnostics": diagnostics,
    }
