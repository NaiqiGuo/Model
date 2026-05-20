"""
Computes damage metrics using the current repo layout.

Current layout:
  - Dk uses Modeling/<structure>/<source>/strain_stress/structure/<event>.csv
  - Dk_fd uses Modeling/<structure>/<source>/force_deformation/structure/<event>.csv
  - Baseline intensity uses Modeling/<structure>/field/acceleration/ground/<event>.csv
  - Df uses Modeling/<structure>/<source>/frequency_post_eq/structure/<event>.csv
  - Dr uses System ID/<structure>/<source>/<output_quantity>/...

Outputs are saved under:
  Damage/<structure>/<source>/<output_quantity>/
"""

from pathlib import Path
import os
import numpy as np
import pandas as pd


# -----------------------------
# User config
# -----------------------------
STRUCTURE = os.environ.get("DAMAGE_STRUCTURE", "bridge")  # "frame", "bridge"
SOURCE = os.environ.get("DAMAGE_SOURCE", "inelastic")    # "elastic", "inelastic", "field"
OUTPUT_QUANTITY = os.environ.get("DAMAGE_OUTPUT_QUANTITY", "displacement")
SID_METHOD = os.environ.get("DAMAGE_SID_METHOD", "srim")

# Toggle metrics. Dk is the active one for now; Df/Dr stay available for later.
COMPUTE_DK = os.environ.get("DAMAGE_COMPUTE_DK", "1") == "1"
COMPUTE_DK_FD = os.environ.get(
    "DAMAGE_COMPUTE_DK_FD",
    "1" if STRUCTURE == "bridge" else "0",
) == "1"
COMPUTE_DF = os.environ.get("DAMAGE_COMPUTE_DF", "0") == "1"
COMPUTE_DR = os.environ.get("DAMAGE_COMPUTE_DR", "0") == "1"
COMPUTE_ERROR = os.environ.get("DAMAGE_COMPUTE_ERROR", "1") == "1"

# Elements used for Dk
if STRUCTURE == "bridge":
    ELEMENTS = [3]
    FD_ELEMENTS = [107]
else:
    ELEMENTS = [1, 5, 9]
    FD_ELEMENTS = []

# Baseline selection
N_BASELINE = 5
BASELINE_METRIC = "pga"  # "pga" or "rms"

# Df frequency source
FREQ_STAGE = "frequency_post_eq"

# Dr tail window (seconds)
DR_TAIL_SEC = 10.0


MODELING_CASE_DIR = Path("Modeling") / STRUCTURE / SOURCE
FIELD_INPUT_DIR = Path("Modeling") / STRUCTURE / "field" / "acceleration" / "ground"
FIELD_DT_DIR = Path("Modeling") / STRUCTURE / "field" / "dt" / "ground"
SYSTEM_ID_CASE_DIR = Path("System ID") / STRUCTURE / SOURCE / OUTPUT_QUANTITY
OUTPUT_DIR = Path("Damage") / STRUCTURE / SOURCE / OUTPUT_QUANTITY


def compute_intensity(inputs: np.ndarray, metric: str) -> float:
    if metric == "pga":
        return float(np.max(np.abs(inputs)))
    if metric == "rms":
        rms_each = np.sqrt(np.mean(inputs**2, axis=1))
        return float(np.max(rms_each))
    raise ValueError("BASELINE_METRIC must be 'pga' or 'rms'")


def compute_ksec_for_element(df_ss: pd.DataFrame, ele_id: int, eps_stab: float = 1e-12) -> float:
    stress_col = f"ele{ele_id}_stress"
    strain_col = f"ele{ele_id}_strain"
    if stress_col not in df_ss.columns or strain_col not in df_ss.columns:
        raise KeyError(f"Missing columns {stress_col} or {strain_col}")

    stress = df_ss[stress_col].to_numpy(dtype=float)
    strain = df_ss[strain_col].to_numpy(dtype=float)
    idx = int(np.argmax(np.abs(stress)))
    sigma = float(stress[idx])
    eps = float(strain[idx])
    return abs(sigma) / (abs(eps) + eps_stab)


def compute_Dk_event(df_ss: pd.DataFrame, elements, kref_ele: dict, eps_stab: float = 1e-12) -> dict:
    out = {}
    dk_list = []
    for ele in elements:
        ksec = compute_ksec_for_element(df_ss, ele)
        kref = float(kref_ele[ele])
        dk = 1.0 - (ksec / (kref + eps_stab))
        out[f"ksec_ele{ele}"] = float(ksec)
        out[f"Dk_ele{ele}"] = float(dk)
        dk_list.append(dk)
    out["Dk_median"] = float(np.median(np.asarray(dk_list, dtype=float)))
    return out


def compute_ksec_fd_for_element(df_fd: pd.DataFrame, ele_id: int, eps_stab: float = 1e-12) -> float:
    force_col = f"ele{ele_id}_force"
    deformation_col = f"ele{ele_id}_deformation"
    if force_col not in df_fd.columns or deformation_col not in df_fd.columns:
        raise KeyError(f"Missing columns {force_col} or {deformation_col}")

    force = df_fd[force_col].to_numpy(dtype=float)
    deformation = df_fd[deformation_col].to_numpy(dtype=float)
    idx = int(np.argmax(np.abs(force)))
    f_val = float(force[idx])
    u_val = float(deformation[idx])
    return abs(f_val) / (abs(u_val) + eps_stab)


def compute_Dk_fd_event(df_fd: pd.DataFrame, elements, kref_ele_fd: dict, eps_stab: float = 1e-12) -> dict:
    out = {}
    dk_list = []
    for ele in elements:
        ksec_fd = compute_ksec_fd_for_element(df_fd, ele)
        kref_fd = float(kref_ele_fd[ele])
        dk_fd = 1.0 - (ksec_fd / (kref_fd + eps_stab))
        out[f"ksec_fd_ele{ele}"] = float(ksec_fd)
        out[f"Dk_fd_ele{ele}"] = float(dk_fd)
        dk_list.append(dk_fd)
    out["Dk_fd_median"] = float(np.median(np.asarray(dk_list, dtype=float)))
    return out


def compute_Df_event(f_event: np.ndarray, fbase_per_mode: np.ndarray, eps_stab: float = 1e-12) -> dict:
    if f_event.size < 3:
        raise ValueError(f"Expected 3 modes, got {f_event.size}")
    if fbase_per_mode.size < 3:
        raise ValueError(f"Expected 3 baseline modes, got {fbase_per_mode.size}")

    f_event = f_event[:3].astype(float)
    fbase = fbase_per_mode[:3].astype(float)
    df_modes = 1.0 - (f_event / (fbase + eps_stab))
    return {
        "f1_event": float(f_event[0]),
        "f2_event": float(f_event[1]),
        "f3_event": float(f_event[2]),
        "f1_base": float(fbase[0]),
        "f2_base": float(fbase[1]),
        "f3_base": float(fbase[2]),
        "Df_mode1": float(df_modes[0]),
        "Df_mode2": float(df_modes[1]),
        "Df_mode3": float(df_modes[2]),
        "Df_median": float(np.median(df_modes)),
    }


def compute_Dr_residual_tail(y_true: np.ndarray, y_pred: np.ndarray, dt: float, tail_sec: float) -> dict:
    nt = y_true.shape[1]
    k_tail = int(round(tail_sec / dt))
    k_tail = max(1, min(k_tail, nt))
    y_true_tail = y_true[:, -k_tail:]
    y_pred_tail = y_pred[:, -k_tail:]
    res = np.mean(np.abs(y_true_tail - y_pred_tail), axis=1)
    out = {
        "Dr_tail_sec": float(tail_sec),
        "Dr_tail_samples": int(k_tail),
        "Dr_residual_mean": float(np.mean(res)),
        "Dr_residual_max": float(np.max(res)),
    }
    for i, val in enumerate(res, start=1):
        out[f"Dr_residual_ch{i}"] = float(val)
    return out


def output_labels_for(model: str, source: str, quantity: str):
    if source == "field":
        if model == "bridge":
            return ["Channel 4 (Y)", "Channel 7 (Y)", "Channel 9 (Y)"]
        if model == "frame":
            if quantity == "acceleration":
                return [
                    "Channel 3 (X)",
                    "Channel 4 (Y)",
                    "Channel 6 (X)",
                    "Channel 7 (Y)",
                    "Channel 9 (X)",
                    "Channel 10 (Y)",
                ]
            return [
                "Channel 21",
                "Channel 22",
                "Channel 23",
                "Channel 24",
                "Channel 25",
                "Channel 26",
            ]

    if model == "frame":
        return ["Floor 1, X", "Floor 1, Y", "Floor 2, X", "Floor 2, Y", "Floor 3, X", "Floor 3, Y"]
    return ["West Deck Interface, Y", "Column 1 Top, Y", "East Deck Interface, Y"]


def sanitize_label(label: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in label)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


def load_prediction_error(event_id: str) -> np.ndarray:
    path = SYSTEM_ID_CASE_DIR / "System ID Results" / "prediction error" / f"{event_id}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction error file: {path}")
    err = np.loadtxt(path, dtype=float, delimiter=",")
    return np.atleast_1d(err).astype(float)


def compute_error_event(err_vec: np.ndarray, labels) -> dict:
    if err_vec.size != len(labels):
        raise ValueError(f"Prediction error length {err_vec.size} does not match label count {len(labels)}")

    out = {}
    for label, val in zip(labels, err_vec):
        out[f"Error_{sanitize_label(label)}"] = float(val)
    out["Error_mean"] = float(np.mean(err_vec))
    out["Error_median"] = float(np.median(err_vec))
    out["Error_max"] = float(np.max(err_vec))
    return out


def list_event_ids_from_modeling(case_dir: Path):
    stress_strain_dir = case_dir / "strain_stress" / "structure"
    if not stress_strain_dir.exists():
        raise FileNotFoundError(f"Missing stress-strain directory: {stress_strain_dir}")
    event_ids = sorted(
        [path.stem for path in stress_strain_dir.glob("*.csv") if path.stem.isdigit()],
        key=int,
    )
    return event_ids


def load_field_inputs(event_id: str) -> np.ndarray:
    path = FIELD_INPUT_DIR / f"{event_id}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing field input file: {path}")
    inputs = np.loadtxt(path, dtype=float)
    if inputs.ndim == 1:
        inputs = inputs[None, :]
    return inputs


def load_strain_stress_df(event_id: str) -> pd.DataFrame:
    path = MODELING_CASE_DIR / "strain_stress" / "structure" / f"{event_id}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing stress-strain file: {path}")
    return pd.read_csv(path)


def load_force_deformation_df(event_id: str) -> pd.DataFrame:
    path = MODELING_CASE_DIR / "force_deformation" / "structure" / f"{event_id}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing force-deformation file: {path}")
    return pd.read_csv(path)


def load_freq_vector(event_id: str) -> np.ndarray:
    path = MODELING_CASE_DIR / FREQ_STAGE / "structure" / f"{event_id}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing frequency file: {path}")
    freq = np.loadtxt(path, dtype=float)
    return np.atleast_1d(freq).astype(float)


def load_dt(event_id: str) -> float:
    sid_dt_path = SYSTEM_ID_CASE_DIR / "System ID Training Data" / "dt" / f"{event_id}.csv"
    field_dt_path = FIELD_DT_DIR / f"{event_id}.csv"
    for path in [sid_dt_path, field_dt_path]:
        if path.exists():
            dt = np.loadtxt(path, dtype=float)
            return float(np.atleast_1d(dt).reshape(-1)[0])
    raise FileNotFoundError(f"Missing dt file for event {event_id} in {sid_dt_path} or {field_dt_path}")


def load_pred_true_processed(event_id: str):
    pred_dir = SYSTEM_ID_CASE_DIR / "System ID Results" / "prediction" / SID_METHOD / event_id
    p_true = pred_dir / "outputs_true_processed.csv"
    p_pred = pred_dir / "outputs_pred_processed.csv"
    if not p_true.exists() or not p_pred.exists():
        raise FileNotFoundError(f"Missing processed outputs under {pred_dir}")

    y_true = np.loadtxt(p_true, dtype=float)
    y_pred = np.loadtxt(p_pred, dtype=float)
    if y_true.ndim == 1:
        y_true = y_true[None, :]
    if y_pred.ndim == 1:
        y_pred = y_pred[None, :]
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}")
    return y_true, y_pred


def select_baseline_events(event_ids, n_baseline: int, metric: str):
    rows = []
    for event_id in event_ids:
        inputs = load_field_inputs(event_id)
        intensity = compute_intensity(inputs, metric)
        rows.append({"event": int(event_id), "intensity": intensity})
    df_int = pd.DataFrame(rows).sort_values("intensity").reset_index(drop=True)
    baseline_ids = df_int.loc[: n_baseline - 1, "event"].astype(int).tolist()
    return baseline_ids, df_int


def compute_kref_from_event_ids(baseline_ids, elements) -> dict:
    kref_ele = {}
    for ele in elements:
        k_list = []
        for event_id in baseline_ids:
            df_ss = load_strain_stress_df(str(event_id))
            k_list.append(compute_ksec_for_element(df_ss, ele))
        kref_ele[ele] = float(np.median(np.asarray(k_list, dtype=float)))
    return kref_ele


def compute_kref_fd_from_event_ids(baseline_ids, elements) -> dict:
    kref_ele_fd = {}
    for ele in elements:
        k_list = []
        for event_id in baseline_ids:
            df_fd = load_force_deformation_df(str(event_id))
            k_list.append(compute_ksec_fd_for_element(df_fd, ele))
        kref_ele_fd[ele] = float(np.median(np.asarray(k_list, dtype=float)))
    return kref_ele_fd


def compute_fbase_from_event_ids(baseline_ids) -> np.ndarray:
    f_mat = []
    for event_id in baseline_ids:
        f_vec = load_freq_vector(str(event_id))
        if f_vec.size < 3:
            raise ValueError(f"Event {event_id} has fewer than 3 frequency modes")
        f_mat.append(f_vec[:3])
    return np.median(np.vstack(f_mat).astype(float), axis=0)


if __name__ == "__main__":
    if not MODELING_CASE_DIR.exists():
        raise FileNotFoundError(f"MODELING_CASE_DIR not found: {MODELING_CASE_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    event_ids = list_event_ids_from_modeling(MODELING_CASE_DIR)
    if not event_ids:
        raise RuntimeError(f"No event csv files found under {MODELING_CASE_DIR/'strain_stress'/'structure'}")

    baseline_ids, df_int = select_baseline_events(event_ids, N_BASELINE, BASELINE_METRIC)
    df_int["is_baseline"] = df_int["event"].isin(baseline_ids)
    df_int.to_csv(OUTPUT_DIR / "damage_baseline_selection.csv", index=False)

    print("\nAuto baseline selection")
    print(f"  structure = {STRUCTURE}")
    print(f"  source = {SOURCE}")
    print(f"  output_quantity = {OUTPUT_QUANTITY}")
    print(f"  metric = {BASELINE_METRIC}")
    print(f"  N_BASELINE = {N_BASELINE}")
    print(f"  baseline events = {baseline_ids}")
    print(f"  saved: {OUTPUT_DIR/'damage_baseline_selection.csv'}")

    kref_ele = compute_kref_from_event_ids(baseline_ids, ELEMENTS) if COMPUTE_DK else {}
    kref_ele_fd = compute_kref_fd_from_event_ids(baseline_ids, FD_ELEMENTS) if COMPUTE_DK_FD else {}
    fbase = compute_fbase_from_event_ids(baseline_ids) if COMPUTE_DF else None
    error_labels = output_labels_for(STRUCTURE, SOURCE, OUTPUT_QUANTITY) if COMPUTE_ERROR else []

    if COMPUTE_DK:
        print("\nBaseline references for Dk")
        for ele in ELEMENTS:
            print(f"  k_ref ele{ele} = {kref_ele[ele]:.6g}")
    if COMPUTE_DK_FD:
        print("\nBaseline references for Dk_fd")
        for ele in FD_ELEMENTS:
            print(f"  k_ref_fd ele{ele} = {kref_ele_fd[ele]:.6g}")
    if COMPUTE_DF:
        print("\nBaseline references for Df")
        print(f"  f_base: mode1={fbase[0]:.6g}, mode2={fbase[1]:.6g}, mode3={fbase[2]:.6g}")

    rows_k, rows_k_fd, rows_f, rows_r, rows_error, rows_all = [], [], [], [], [], []

    for event_id in event_ids:
        ev = int(event_id)
        row_all = {
            "event": ev,
            "is_baseline": bool(ev in baseline_ids),
            "intensity_metric": BASELINE_METRIC,
        }

        if COMPUTE_DK:
            df_ss = load_strain_stress_df(event_id)
            dk_dict = compute_Dk_event(df_ss, ELEMENTS, kref_ele)
            dk_row = {"event": ev, **dk_dict}
            for ele in ELEMENTS:
                dk_row[f"kref_ele{ele}"] = float(kref_ele[ele])
            rows_k.append(dk_row)
            row_all["Dk_median"] = dk_dict["Dk_median"]

        if COMPUTE_DK_FD:
            df_fd = load_force_deformation_df(event_id)
            dk_fd_dict = compute_Dk_fd_event(df_fd, FD_ELEMENTS, kref_ele_fd)
            dk_fd_row = {"event": ev, **dk_fd_dict}
            for ele in FD_ELEMENTS:
                dk_fd_row[f"kref_fd_ele{ele}"] = float(kref_ele_fd[ele])
            rows_k_fd.append(dk_fd_row)
            row_all["Dk_fd_median"] = dk_fd_dict["Dk_fd_median"]

        if COMPUTE_DF:
            f_event = load_freq_vector(event_id)
            df_dict = compute_Df_event(f_event, fbase)
            rows_f.append({"event": ev, **df_dict})
            row_all["Df_median"] = df_dict["Df_median"]

        if COMPUTE_DR:
            dt = load_dt(event_id)
            y_true, y_pred = load_pred_true_processed(event_id)
            dr_dict = compute_Dr_residual_tail(y_true, y_pred, dt, DR_TAIL_SEC)
            rows_r.append({"event": ev, **dr_dict})
            row_all["Dr_residual_mean_30s"] = dr_dict["Dr_residual_mean"]
            row_all["Dr_residual_max_30s"] = dr_dict["Dr_residual_max"]

        if COMPUTE_ERROR:
            err_vec = load_prediction_error(event_id)
            err_dict = compute_error_event(err_vec, error_labels)
            rows_error.append({"event": ev, **err_dict})
            row_all["Error_mean"] = err_dict["Error_mean"]
            row_all["Error_median"] = err_dict["Error_median"]
            row_all["Error_max"] = err_dict["Error_max"]

        rows_all.append(row_all)

    if rows_k:
        pd.DataFrame(rows_k).sort_values("event").reset_index(drop=True).to_csv(
            OUTPUT_DIR / "damage_Dk.csv", index=False
        )
    if rows_k_fd:
        pd.DataFrame(rows_k_fd).sort_values("event").reset_index(drop=True).to_csv(
            OUTPUT_DIR / "damage_Dk_fd.csv", index=False
        )
    if rows_f:
        pd.DataFrame(rows_f).sort_values("event").reset_index(drop=True).to_csv(
            OUTPUT_DIR / "damage_Df.csv", index=False
        )
    if rows_r:
        pd.DataFrame(rows_r).sort_values("event").reset_index(drop=True).to_csv(
            OUTPUT_DIR / "damage_Dr.csv", index=False
        )
    if rows_error:
        pd.DataFrame(rows_error).sort_values("event").reset_index(drop=True).to_csv(
            OUTPUT_DIR / "damage_error.csv", index=False
        )
    if rows_all:
        pd.DataFrame(rows_all).sort_values("event").reset_index(drop=True).to_csv(
            OUTPUT_DIR / "damage_all.csv", index=False
        )

    print("\nSaved outputs")
    if rows_k:
        print(f"  {OUTPUT_DIR/'damage_Dk.csv'}")
    if rows_k_fd:
        print(f"  {OUTPUT_DIR/'damage_Dk_fd.csv'}")
    if rows_f:
        print(f"  {OUTPUT_DIR/'damage_Df.csv'}")
    if rows_r:
        print(f"  {OUTPUT_DIR/'damage_Dr.csv'}")
    if rows_error:
        print(f"  {OUTPUT_DIR/'damage_error.csv'}")
    if rows_all:
        print(f"  {OUTPUT_DIR/'damage_all.csv'}")
