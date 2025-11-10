import pytest
from bii import normalize

def test_normalize_basic():
    assert normalize([1, 1, 2]) == [0.25, 0.25, 0.5]

def test_normalize_errors():
    with pytest.raises(ValueError):
        normalize([])
    with pytest.raises(ValueError):
        normalize([0.0, 0.0])

