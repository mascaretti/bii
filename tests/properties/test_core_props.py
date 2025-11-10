import math
from hypothesis import given, strategies as st
from bii import normalize

finite = st.floats(allow_nan=False, allow_infinity=False, width=64,
                   min_value=-1e9, max_value=1e9)

@given(st.lists(finite, min_size=1))
def test_normalize_sum_to_one(xs):
    if all(x == 0.0 for x in xs):
        # Expect error for all-zero input
        try:
            normalize(xs)
        except ValueError:
            return
        assert False, "Expected ValueError for all-zero input"
    else:
        ys = normalize(xs)
        assert len(ys) == len(xs)
        assert math.isclose(sum(ys), 1.0, rel_tol=1e-12, abs_tol=1e-12)
        # Scale invariance for c>0
        c = 2.5
        ys2 = normalize([c * x for x in xs])
        for a, b in zip(ys, ys2):
            assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)

