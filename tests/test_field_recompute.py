"""
Lightweight regression tests for the field-input pipeline (No FE solve).

Frame: recomputes the frame field ground inputs from a committed 15 s slice of
the raw record (get_249_data + scale_249_units). Only needs xara's unit
constants, so it runs fast in CI without a full FE model.

Bridge: recomputes the bridge field ground inputs from one event in events.pkl
(get_measurements -> extract_channels).

Field results must never change, so the comparisons are exact.
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
ROOT = HERE.parent
FRAME_FIXTURE = HERE / "fixtures" / "ce249Run226_15s.txt.gz"
GOLDEN_FRAME = HERE / "golden" / "frame" / "field"
BRIDGE_FIXTURE = HERE / "fixtures" / "bridge_event_1.pkl.gz"
GOLDEN_BRIDGE = HERE / "golden" / "bridge" / "field"
EVENTS_PKL = ROOT / "events.pkl"

# Frame field ground inputs, mirroring set_channels_dofs("frame", multisupport=False).
INPUT = {
    "acceleration": ([0, 2], [1, 2]),    # (channels, dofs)
    "displacement": ([34, 35], [1, 2]),
}

# Bridge field ground inputs, mirroring set_channels_dofs("bridge", multisupport=False)
# and the units used in EventAnalysis.load_measurements.
# (channels, dofs, xara.units.iks attribute, extract_channels response)
BRIDGE_INPUT = {
    "acceleration": ([3, 17, 20], [2, 2, 2], "cmps2", "accel"),
    "displacement": ([3, 17, 20], [2, 2, 2], "cm", "displ"),
}


@pytest.fixture(scope="module")
def raw_path(tmp_path_factory):
    dst = tmp_path_factory.mktemp("raw") / "ce249Run226_15s.txt"
    dst.write_bytes(gzip.decompress(FRAME_FIXTURE.read_bytes()))
    return str(dst)


@pytest.mark.parametrize("quantity", ["acceleration", "displacement"])
def test_frame_field_input_matches_golden(raw_path, quantity):
    array, names, units, time, dt = get_249_data(raw_path)
    channels, dofs = INPUT[quantity]
    recomputed = np.vstack([
        np.sign(dof) * array[ch] * scale_249_units(units=units[ch])
        for ch, dof in zip(channels, dofs)
    ])
    golden = np.loadtxt(GOLDEN_FRAME / quantity / "ground" / "226.csv", ndmin=2)
    n = golden.shape[1]
    np.testing.assert_allclose(recomputed[:, :n], golden, rtol=1e-9, atol=1e-9)


def _load_bridge_event():
    """Prefer the committed single-event fixture (works in CI); fall back to the
    local full events.pkl; skip if neither is available."""
    import gzip
    import pickle
    if BRIDGE_FIXTURE.exists():
        with gzip.open(BRIDGE_FIXTURE, "rb") as f:
            return pickle.load(f)
    if EVENTS_PKL.exists():
        with open(EVENTS_PKL, "rb") as f:
            return pickle.load(f)[0]  # sorted by peak_accel -> event_id "1"
    pytest.skip("no bridge event: run tests/make_bridge_fixture.py, or provide events.pkl")


@pytest.mark.parametrize("quantity", ["acceleration", "displacement"])
def test_bridge_field_input_matches_golden(quantity):
    pytest.importorskip("quakeio", reason="unpickling bridge events needs quakeio")
    pytest.importorskip("mdof", reason="get_measurements -> extract_channels needs mdof")
    import xara.units.iks as units
    from utilities import get_measurements

    channels, dofs, unit_attr, response = BRIDGE_INPUT[quantity]
    scale = getattr(units, unit_attr)

    event = _load_bridge_event()
    measurements, dt = get_measurements(event, channels=channels, scale=scale, response=response)
    recomputed = np.vstack([np.sign(dof) * measurements[ch] for ch, dof in zip(channels, dofs)])

    golden = np.loadtxt(GOLDEN_BRIDGE / quantity / "ground" / "1.csv", ndmin=2)
    n = golden.shape[1]
    np.testing.assert_allclose(recomputed[:, :n], golden, rtol=1e-9, atol=1e-9)
