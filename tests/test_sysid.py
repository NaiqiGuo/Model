"""Simple three-story MCK example for testing system identification."""

import pickle
from pathlib import Path

import numpy as np


ELCENTRO = Path(__file__).resolve().parent.parent / "uploads" / "elcentro.txt"
PREDICTION = Path(__file__).resolve().parent / "results" / "sysid" / "elcentro" / "u_pred.csv"
ABCD_PATH = Path(__file__).resolve().parent / "results" / "sysid" / "elcentro" / "abcd.pkl"
BRIDGE_GOLDEN = Path(__file__).resolve().parent / "golden" / "bridge" / "inelastic"
BRIDGE_RESULTS = Path(__file__).resolve().parent / "results" / "sysid" / "bridge" / "event1" / "inelastic" / "displacement"
BRIDGE_PREDICTION = BRIDGE_RESULTS / "u_pred.csv"
BRIDGE_ABCD = BRIDGE_RESULTS / "abcd.pkl"
DT = 0.02


def _get_mck():
    """Define and return the M, C, and K matrices from the example."""
    M = np.diag([400.0, 400.0, 200.0]) / 386.0

    K = 610.0 * np.array([
        [2.0, -1.0, 0.0],
        [-1.0, 2.0, -1.0],
        [0.0, -1.0, 1.0],
    ])

    C = 0.9198 * M + 0.0021 * K

    return M, C, K


def _mck_response(M, C, K, f, dt):
    """Solve M*u_ddot + C*u_dot + K*u = [f, f, f] with Newmark."""
    f = np.asarray(f)
    nt = f.size
    force = np.vstack([f, f, f])

    u = np.zeros((3, nt))
    u_dot = np.zeros((3, nt))
    u_ddot = np.zeros((3, nt))

    # Newmark average-acceleration constants.
    beta = 1 / 4
    gamma = 1 / 2
    a0 = 1 / (beta * dt**2)
    a1 = gamma / (beta * dt)
    a2 = 1 / (beta * dt)
    a3 = 1 / (2 * beta) - 1
    a4 = gamma / beta - 1
    a5 = dt * (gamma / (2 * beta) - 1)

    u_ddot[:, 0] = np.linalg.solve(M, force[:, 0])
    effective_K = K + a0 * M + a1 * C

    for i in range(nt - 1):
        effective_force = (
            force[:, i + 1]
            + M @ (a0 * u[:, i] + a2 * u_dot[:, i] + a3 * u_ddot[:, i])
            + C @ (a1 * u[:, i] + a4 * u_dot[:, i] + a5 * u_ddot[:, i])
        )

        u[:, i + 1] = np.linalg.solve(effective_K, effective_force)
        u_ddot[:, i + 1] = (
            a0 * (u[:, i + 1] - u[:, i])
            - a2 * u_dot[:, i]
            - a3 * u_ddot[:, i]
        )
        u_dot[:, i + 1] = u_dot[:, i] + dt * (
            (1 - gamma) * u_ddot[:, i] + gamma * u_ddot[:, i + 1]
        )

    return u


def _relative_error(u_pred, u_true):
    """Return the relative error between predicted and true displacement."""
    return np.linalg.norm(u_pred - u_true) / np.linalg.norm(u_true)


def test_elcentro_sysid():
    from mdof import predict, sysid
    from mdof.utilities.testing import intensity_bounds, truncate_by_bounds

    M, C, K = _get_mck()
    f = np.loadtxt(ELCENTRO)
    u = _mck_response(M, C, K, f, DT)

    # Use one set of bounds to keep the input and output time-aligned.
    bounds = intensity_bounds(u[0], lb=0.01, ub=0.99)
    f_truncated = truncate_by_bounds(f[None, :], bounds)
    u_truncated = truncate_by_bounds(u, bounds)

    assert f_truncated.shape[1] == u_truncated.shape[1]

    # Three physical DOFs give six states: three displacements + velocities.
    n = 3
    ABCD = sysid(
        f_truncated,
        u_truncated,
        method="srim",
        order=2 * n,
        horizon=50,
        threads=1,
        verbose=False,
    )

    # Prediction using only the truncated input.
    u_pred = predict(ABCD, f_truncated)

    PREDICTION.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(PREDICTION, u_pred)
    with open(ABCD_PATH, "wb") as file:
        pickle.dump(ABCD, file)


def test_elcentro_prediction():
    from mdof.utilities.testing import intensity_bounds, truncate_by_bounds

    M, C, K = _get_mck()
    f = np.loadtxt(ELCENTRO)
    u = _mck_response(M, C, K, f, DT)

    bounds = intensity_bounds(u[0], lb=0.01, ub=0.99)
    u_truncated = truncate_by_bounds(u, bounds)
    u_pred = np.loadtxt(PREDICTION)

    assert u_pred.shape == u_truncated.shape
    assert np.all(np.isfinite(u_pred))

    relative_error = _relative_error(u_pred, u_truncated)

    assert relative_error < 0.20


def test_simulate():
    from mdof.simulate import simulate
    from mdof.utilities.testing import intensity_bounds, truncate_by_bounds

    M, C, K = _get_mck()
    f = np.loadtxt(ELCENTRO)
    u = _mck_response(M, C, K, f, DT)

    bounds = intensity_bounds(u[0], lb=0.01, ub=0.99)
    f_truncated = truncate_by_bounds(f[None, :], bounds)

    with open(ABCD_PATH, "rb") as file:
        ABCD = pickle.load(file)

    u_simulated = simulate(ABCD, f_truncated)
    u_pred = np.loadtxt(PREDICTION)

    assert u_simulated.shape == u_pred.shape
    assert np.all(np.isfinite(u_simulated))
    np.testing.assert_allclose(u_simulated, u_pred, rtol=1e-10, atol=1e-12)


def test_bridge_inelastic_event1_sysid():
    from mdof import predict, sysid
    from mdof.utilities.testing import intensity_bounds, truncate_by_bounds

    f = np.loadtxt(
        BRIDGE_GOLDEN / "acceleration" / "ground" / "1.csv",
        ndmin=2,
    )
    u = np.loadtxt(
        BRIDGE_GOLDEN / "displacement" / "structure" / "1.csv",
        ndmin=2,
    )

    bounds = intensity_bounds(u[0], lb=0.01, ub=0.99)
    f_truncated = truncate_by_bounds(f, bounds)
    u_truncated = truncate_by_bounds(u, bounds)

    assert f_truncated.shape[1] == u_truncated.shape[1]

    n = 4
    ABCD = sysid(
        f_truncated,
        u_truncated,
        method="srim",
        order=2 * n,
        horizon=190,
        threads=1,
        verbose=False,
    )
    u_pred = predict(ABCD, f_truncated)

    BRIDGE_RESULTS.mkdir(parents=True, exist_ok=True)
    np.savetxt(BRIDGE_PREDICTION, u_pred)
    with open(BRIDGE_ABCD, "wb") as file:
        pickle.dump(ABCD, file)


def test_bridge_inelastic_event1_prediction():
    from mdof.utilities.testing import intensity_bounds, truncate_by_bounds

    u = np.loadtxt(
        BRIDGE_GOLDEN / "displacement" / "structure" / "1.csv",
        ndmin=2,
    )
    bounds = intensity_bounds(u[0], lb=0.01, ub=0.99)
    u_truncated = truncate_by_bounds(u, bounds)
    u_pred = np.loadtxt(BRIDGE_PREDICTION, ndmin=2)

    assert u_pred.shape == u_truncated.shape
    assert np.all(np.isfinite(u_pred))

    relative_error = _relative_error(u_pred, u_truncated)
    assert relative_error < 0.30


def test_bridge_inelastic_event1_simulate():
    from mdof.simulate import simulate
    from mdof.utilities.testing import intensity_bounds, truncate_by_bounds

    f = np.loadtxt(
        BRIDGE_GOLDEN / "acceleration" / "ground" / "1.csv",
        ndmin=2,
    )
    u = np.loadtxt(
        BRIDGE_GOLDEN / "displacement" / "structure" / "1.csv",
        ndmin=2,
    )
    bounds = intensity_bounds(u[0], lb=0.01, ub=0.99)
    f_truncated = truncate_by_bounds(f, bounds)

    with open(BRIDGE_ABCD, "rb") as file:
        ABCD = pickle.load(file)

    u_simulated = simulate(ABCD, f_truncated)
    u_pred = np.loadtxt(BRIDGE_PREDICTION, ndmin=2)

    assert u_simulated.shape == u_pred.shape
    assert np.all(np.isfinite(u_simulated))
    np.testing.assert_allclose(u_simulated, u_pred, rtol=1e-10, atol=1e-12)
