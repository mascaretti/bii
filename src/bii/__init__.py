"""Bayesian Information Imbalance — metric weight inference from triplets."""

from bii.data import T_from_X, make_triplets
from bii.diagnostics import (
    alignment_index,
    compute_ess,
    compute_rhat,
    compute_waic,
    triplet_accuracy,
    weight_entropy,
)
from bii.fit import fit_bii
from bii.priors import make_dirichlet_logposterior
from bii.sampling import run_nuts, run_vi, sample_vi

__all__ = [
    "fit_bii",
    "T_from_X",
    "make_triplets",
    "make_dirichlet_logposterior",
    "run_nuts",
    "run_vi",
    "sample_vi",
    "compute_waic",
    "compute_rhat",
    "compute_ess",
    "weight_entropy",
    "triplet_accuracy",
    "alignment_index",
]
