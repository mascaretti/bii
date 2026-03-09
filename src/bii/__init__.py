"""Bayesian Information Imbalance — metric weight inference from triplets."""

from bii.data import T_from_X
from bii.fit import fit_bii
from bii.ii import compute_ii

__all__ = ["fit_bii", "T_from_X", "compute_ii"]
