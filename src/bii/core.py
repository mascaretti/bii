from __future__ import annotations
from typing import Iterable, List
import math

def normalize(xs: Iterable[float]) -> List[float]:
    """Scale a sequence so it sums to 1. Raises ValueError on empty or all-zero."""
    lst = list(xs)
    if not lst:
        raise ValueError("Cannot normalize empty sequence")
    total = sum(lst)
    if math.isclose(total, 0.0):
        raise ValueError("Cannot normalize sequence with zero sum")
    return [x / total for x in lst]

