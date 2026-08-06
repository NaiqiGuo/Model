"""
Opt-in model regression: runs the FE solve on a short (15 s) event and
compares the model outputs to the frozen golden.

This is the only true regression for the elastic/inelastic results, but it needs
xara and runs the solver, so it is marked `slow` and excluded from the default
(and CI) run. Run it deliberately after touching model code:

    pytest -m slow

Compare displacement, acceleration, and the pre-earthquake frequencies (the
initial eigenvalue analysis).

All heavy imports are inside the test so the default (no-xara) collection stays
clean; the test drives the real EventAnalysis.run_fe using the golden model input,
so it exercises the production model-building + solve + post-processing
code path without re-reading the raw record or running all events.
"""
import argparse
from pathlib import Path

import numpy as np
import pytest

GOLDEN = Path(__file__).resolve().parent / "golden"
EVENT = {"frame": "226", "bridge": "1"}


@pytest.mark.slow
@pytest.mark.parametrize("structure,case", [
    ("frame", "elastic"), ("frame", "inelastic"),
    ("bridge", "elastic"), ("bridge", "inelastic"),
])
def test_model_solve_matches_golden(structure, case):
    pytest.importorskip("xara", reason="FE solve needs xara")
    from get_data import RunConfig, EventAnalysis
    from utilities import get_node_outputs

    ev = EVENT[structure]
    base = f"{structure}/{case}"
    model_input = np.loadtxt(GOLDEN / f"{base}/acceleration/ground/{ev}.csv", ndmin=2)
    dt = float(np.loadtxt(GOLDEN / f"{base}/dt/ground/{ev}.csv", ndmin=1)[0])

    args = argparse.Namespace(
        structure=structure, multisupport=False, elastic=(case == "elastic"),
        field_only=False, from_scratch=False, frame_coupons=True,
        frame_zerolength="section", verbose=0,
    )
    cfg = RunConfig.from_args(args)

    # Drive the real pipeline with the golden model input (skip the raw
    # read / field processing that load_measurements would normally do).
    event = "ce249Run226.txt" if structure == "frame" else ""
    ea = EventAnalysis(cfg, event, 0)
    ea.inputs["field"]["dt"] = dt
    ea.inputs["model"] = {"dt": dt, "acceleration": model_input}
    ea.nt = model_input.shape[1]

    assert ea.run_fe() is True, "FE solve failed"

    nodes = cfg.channels_dofs["output"]["nodes"]
    dofs = cfg.channels_dofs["output"]["dofs"]
    got = {
        "displacement": get_node_outputs(ea.displ, nodes=nodes, dofs=dofs)[:, 1:],
        "acceleration": get_node_outputs(ea.accel, nodes=nodes, dofs=dofs)[:, 1:],
    }
    # The last couple of integration steps are a boundary artifact.
    # Drop a small tail margin.
    TAIL_TRIM = 5
    for quantity, arr in got.items():
        gold = np.loadtxt(GOLDEN / f"{base}/{quantity}/structure/{ev}.csv", ndmin=2)
        n = gold.shape[1] - TAIL_TRIM
        np.testing.assert_allclose(arr[:, :n], gold[:, :n], rtol=1e-6, atol=1e-6)

    # Pre-earthquake frequencies (initial eigen analysis)
    fpre = np.loadtxt(GOLDEN / f"{base}/frequency_pre_eq/structure/{ev}.csv", ndmin=1)
    freqs = np.asarray(ea.freqs_before, dtype=float).ravel()
    np.testing.assert_allclose(freqs[:fpre.shape[0]], fpre, rtol=1e-6, atol=1e-6)
