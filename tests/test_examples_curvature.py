import json
import pytest

from pathlib import Path

from examples.analyze_curvature import analyze_n, parse_args

try:  # pragma: no cover
    import numpyro  # type: ignore
    HAS_NUMPYRO = True
except ImportError:  # pragma: no cover
    HAS_NUMPYRO = False


@pytest.mark.skipif(not HAS_NUMPYRO, reason="numpyro required")
def test_analyze_curvature_small(tmp_path: Path):
    class Args:
        seed = 0
        p = 4
        row_share = 0.5
        num_shells = 4
        quantile_outer = 0.2
        warmup = 50
        samples = 50
        outdir = tmp_path
    args = Args()
    n = 40
    analyze_n(n, args, tmp_path)
    summary_file = tmp_path / f"summary_n{n}.json"
    assert summary_file.exists()
    data = json.loads(summary_file.read_text())
    assert "coverage" in data and "rmse" in data and len(data["rmse"]) == args.p
