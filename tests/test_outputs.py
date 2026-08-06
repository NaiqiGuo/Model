"""
Frozen-golden spot checks for all six cases (numpy only, no solver).

Structural soundness validation for committed ground-truth samples in
tests/golden/ (correct channel counts, finite values, consistent time
axes, sane frequencies)

Model is NOT re-run, for speed. If model code is changed; use the
'slow' test_model_solve.py tests to spot check models. 
Another option is to regenerate results and diff against tests/golden/
deliberately (see tests/refresh_golden.py).
"""
from pathlib import Path

import numpy as np
import pytest

GOLDEN = Path(__file__).resolve().parent / "golden"

OUT_CH = {"frame": 6, "bridge": 3}         # output (structure) channel counts
FIELD_IN_CH = {"frame": 2, "bridge": 3}    # field ground-input channel counts

EVENT = {"frame": "226", "bridge": "1"}
STRUCTURES = ["frame", "bridge"]
MODEL_CASES = [(s, c) for s in STRUCTURES for c in ("elastic", "inelastic")]


def load(rel, ndmin=2):
    return np.loadtxt(GOLDEN / rel, ndmin=ndmin)


@pytest.mark.parametrize("structure", STRUCTURES)
def test_field_channel_counts(structure):
    ev = EVENT[structure]
    assert load(f"{structure}/field/acceleration/ground/{ev}.csv").shape[0] == FIELD_IN_CH[structure]
    assert load(f"{structure}/field/displacement/structure/{ev}.csv").shape[0] == OUT_CH[structure]


@pytest.mark.parametrize("structure,case", MODEL_CASES)
def test_model_output_channels_finite(structure, case):
    ev = EVENT[structure]
    for q in ("displacement", "acceleration"):
        a = load(f"{structure}/{case}/{q}/structure/{ev}.csv")
        assert a.shape[0] == OUT_CH[structure]
        assert np.all(np.isfinite(a))


@pytest.mark.parametrize("structure,case", MODEL_CASES)
def test_frequencies_are_sane(structure, case):
    ev = EVENT[structure]
    for name in ("frequency_pre_eq", "frequency_post_eq"):
        f = load(f"{structure}/{case}/{name}/structure/{ev}.csv", ndmin=1)
        assert f.shape[0] == 5                 # n_modes = 5
        assert np.all(np.isfinite(f))
        assert np.all(f > 0)                    # positive frequencies
        assert np.all(np.diff(f) >= 0)          # ascending


@pytest.mark.parametrize("structure,case", MODEL_CASES + [("frame", "field"), ("bridge", "field")])
def test_time_dt_consistency(structure, case):
    ev = EVENT[structure]
    # field stores displacement under structure; model too -- both share the axis
    disp = load(f"{structure}/{case}/displacement/structure/{ev}.csv")
    time = load(f"{structure}/{case}/time/ground/{ev}.csv", ndmin=1)
    dt = float(load(f"{structure}/{case}/dt/ground/{ev}.csv", ndmin=1)[0])
    nt = disp.shape[1]
    assert time.shape[0] == nt
    np.testing.assert_allclose(time, np.arange(nt) * dt, rtol=0, atol=1e-9)
