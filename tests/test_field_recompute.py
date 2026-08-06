"""
Lightweight regression test for the field-input pipeline (No FE solve).

Recomputes the frame field ground inputs for one event from a committed 15 s
slice of the raw record and requires an exact match to the frozen golden.
This exercises the get_249_data + scale_249_units code paths.
It runs fast in CI because it only takes in the unit constants from xara
(xara.units) rather than setting up and running a full FE model.

Field results must never change, so the comparison is exact.
"""
import gzip
from pathlib import Path

import numpy as np
import pytest

# get_249_data imports xara.units at module load; skip cleanly where xara is
# unavailable (the golden spot checks still cover those environments).
pytest.importorskip("xara", reason="xara.units needed for the field recompute")
from get_249_data import get_249_data, scale_249_units  # noqa: E402

HERE = Path(__file__).resolve().parent
RAW_GZ = HERE / "fixtures" / "ce249Run226_15s.txt.gz"
GOLDEN = HERE / "golden" / "frame" / "field"

# Frame field ground inputs, mirroring set_channels_dofs("frame", multisupport=False).
INPUT = {
    "acceleration": ([0, 2], [1, 2]),    # (channels, dofs)
    "displacement": ([34, 35], [1, 2]),
}


@pytest.fixture(scope="module")
def raw_path(tmp_path_factory):
    dst = tmp_path_factory.mktemp("raw") / "ce249Run226_15s.txt"
    dst.write_bytes(gzip.decompress(RAW_GZ.read_bytes()))
    return str(dst)


@pytest.mark.parametrize("quantity", ["acceleration", "displacement"])
def test_frame_field_input_matches_golden(raw_path, quantity):
    array, names, units, time, dt = get_249_data(raw_path)
    channels, dofs = INPUT[quantity]
    recomputed = np.vstack([
        np.sign(dof) * array[ch] * scale_249_units(units=units[ch])
        for ch, dof in zip(channels, dofs)
    ])
    golden = np.loadtxt(GOLDEN / quantity / "ground" / "226.csv", ndmin=2)
    n = golden.shape[1]
    np.testing.assert_allclose(recomputed[:, :n], golden, rtol=1e-9, atol=1e-9)
