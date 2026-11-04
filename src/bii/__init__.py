"""Bayesian information imbalance: metric weights from triplet comparisons."""

from bii.data import T_from_X, kappa_from_triplets, make_triplets
from bii.diagnostics import aligned_triplets, compute_ess, compute_rhat, compute_waic
from bii.fit import fit_bii
from bii.inference import (
    delta_V_one_triplet,
    loglik_w,
    loglik_w_slope_grid,
    slope_posterior,
    slope_scale,
)
from bii.priors import make_dirichlet_logposterior, make_slope_grid
from bii.sampling import run_map, run_nuts

__all__ = [
    "fit_bii",
    "T_from_X",
    "kappa_from_triplets",
    "make_triplets",
    "delta_V_one_triplet",
    "loglik_w",
    "loglik_w_slope_grid",
    "slope_scale",
    "slope_posterior",
    "make_dirichlet_logposterior",
    "make_slope_grid",
    "run_nuts",
    "run_map",
    "compute_rhat",
    "compute_ess",
    "compute_waic",
    "aligned_triplets",
]
