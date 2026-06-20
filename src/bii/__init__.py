"""Bayesian Information Imbalance — metric weight inference from triplets."""

from bii.data import (
    T_from_X,
    kappa_from_triplets,
    make_triplets,
    make_triplets_random_sparse,
    make_triplets_rank_weighted,
    make_triplets_yfar,
    make_triplets_z_informative,
    make_triplets_z_softmax,
    make_triplets_zfar,
    target_yfar_bump,
)
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
    "kappa_from_triplets",
    "make_triplets",
    "make_triplets_random_sparse",
    "make_triplets_rank_weighted",
    "make_triplets_yfar",
    "make_triplets_z_informative",
    "make_triplets_z_softmax",
    "make_triplets_zfar",
    "target_yfar_bump",
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
