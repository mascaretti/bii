# How it works

## The problem

We observe the same $N$ objects in two representations: a clean **target** $X$ and
a noisy or rescaled **source** $Z$. We want a weighted metric on the source,

$$
\lVert z_i - z_j \rVert_w^2 = \sum_{d=1}^{p} w_d\,(z_{id} - z_{jd})^2,
\qquad w \in \Delta^{p-1},
$$

whose neighbourhood structure matches that of the target $X$ — and we want a
*posterior* over $w$, not just a point estimate.

## Triplet likelihood

Comparisons are encoded as triplets $(i, j; k)$: "under the target, is $i$ or $j$
closer to anchor $k$?". For the weighted source metric, the decision is governed
by the **gap**

$$
\Delta(w) = \lVert z_i - z_k \rVert_w^2 - \lVert z_j - z_k \rVert_w^2 .
$$

Under independent, symmetric source noise the gap has mean $\mu(w)$ and variance
$V(w)$ available in closed form (the package implements them in
{func}`bii.inference.delta_V_one_triplet`). Standardising, $s = \mu(w)/\sqrt{V(w)}$,
the probability that the source agrees with the target ordering is well
approximated by a **probit**,

$$
\Pr(\text{source agrees}) \approx \Phi\!\big(-s\big),
$$

with an error that vanishes as the effective support $m_{\mathrm{eff}}(w) =
\lVert w\rVert_2^{-2}$ grows. A slope-matched **logistic** link is also available
(`link="logit"`) for heavier tails.

## Prior and posterior

A **Dirichlet** prior keeps $w$ on the simplex; its concentration `alpha`
controls sparsity (`alpha < 1` favours a few dominant coordinates). The
log-posterior is assembled by {func}`bii.make_dirichlet_logposterior` and sampled
in an unconstrained space via a softmax transform.

Because triplets that share objects are **dependent**, a naive likelihood is
overconfident. A scalar **power-likelihood correction** `kappa` $\in (0, 1]$
tempers the likelihood so the posterior precision matches the Godambe
(sandwich) information; `kappa = 1` is recovered when triplets are independent.
See {func}`bii.kappa_from_triplets`.

## Inference

- **NUTS** (default): {func}`bii.run_nuts`, multi-chain, with warmup adaptation.
- **Variational inference**: {func}`bii.run_vi` / {func}`bii.sample_vi`, a fast
  mean-field approximation.

{func}`bii.fit_bii` wires triplets → log-posterior → sampler → diagnostics and
returns posterior draws plus R-hat, ESS, WAIC, and alignment scores.

## Robustness extensions

- `clip_s` — a bounded-influence (censored-probit) robustifier that caps the
  per-triplet statistic, defusing saturating triplets.
- `pi_inclusion` / `pi_prior` — an inclusion-mixture likelihood that lets each
  triplet be "off-model" with probability that evolves with $w$.
- `tau_prior` — an extra relation-noise std $\tau$ ($V \to V + \tau^2$) sampled
  jointly.

## Reference

The model, the probit approximation, and the Godambe calibration are developed in
the Bayesian Information Imbalance paper (Mascaretti, in preparation).
