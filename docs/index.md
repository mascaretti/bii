# bii — Bayesian Information Imbalance

`bii` performs **Bayesian estimation of metric weights from triplet comparisons**.
Given a clean target representation $X$ and a noisy or rescaled source $Z$ of the
same objects, it infers a weight vector $w$ on the simplex such that the weighted
metric $\lVert z_i - z_j\rVert_w^2 = \sum_d w_d (z_{id}-z_{jd})^2$ reproduces the
neighbourhood structure of $X$ — together with **honest uncertainty** on $w$.

Inference is fully Bayesian: triplet comparisons enter a probit (or logistic)
likelihood, a Dirichlet prior keeps $w$ on the simplex, and the posterior is drawn
with NUTS (or approximated with mean-field variational inference). A Godambe
power-likelihood correction (`kappa`) calibrates the posterior width for the
dependence between triplets.

```python
from jax import random
from bii import fit_bii

result = fit_bii(random.PRNGKey(0), X_pool, Z_pool, sig=0.1)
w_samples = result["w_samples"]   # (num_samples, num_chains, p), each row on the simplex
```

```{toctree}
:maxdepth: 2
:caption: Guide

installation
quickstart
theory
```

```{toctree}
:maxdepth: 2
:caption: Reference

api
```

## Indices

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
