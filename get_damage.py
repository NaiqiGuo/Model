"""
get_damage.py (inelastic only, auto baseline by low input intensity)

Directory layout assumed:
  /Users/guonaiqi/Documents/UCB/299/Example5-python/Model/frame/inelastic/{1..22}/
    strain_stress.csv
    post_eq_natural_frequencies.csv
    inputs.csv
    dt.txt
    srim/
      outputs_true_processed.csv
      outputs_pred_processed.csv

Computes:
  Dk: stress-strain equivalent secant stiffness degradation (median across elements)
  Df: modal frequency change (per-mode + median across 3 modes) using post_eq_natural_frequencies.csv
  Dr: residual-focused metric using last DR_TAIL_SEC seconds:
      residual per channel = |mean(y_true_tail) - mean(y_pred_tail)|
      Dr_residual_mean / Dr_residual_max

Baseline:
  Automatically selected as N_BASELINE events with lowest input intensity (PGA from inputs.csv).

Outputs saved under:
  .../frame/inelastic/
    damage_baseline_selection.csv
    damage_Dk.csv
    damage_Df.csv
    damage_Dr.csv
    damage_all.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd

EPS = 1e-12

# -----------------------------
# User config
# -----------------------------
CASE_DIR = Path("/Users/guonaiqi/Documents/UCB/299/Example5-python/Model/frame/inelastic")
SID_METHOD = "srim"

# Elements used for Dk (must exist as columns in strain_stress.csv)
ELEMENTS = [1, 5, 9]

# Baseline selection
N_BASELINE = 5                 # pick lowest-intensity N events as baseline
BASELINE_METRIC = "pga"        # "pga" (peak abs accel), or "rms"

# Df frequency source
FREQ_FILE = "post_eq_natural_frequencies.csv"   # has 3 modes per event

# Dr tail window (seconds)
DR_TAIL_SEC = 30.0


# -----------------------------
# Helpers: listing + loading
# -----------------------------
def list_event_dirs(case_dir: Path):
    ev_dirs = [p for p in case_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    return sorted(ev_dirs, key=lambda x: int(x.name))

def load_dt(event_dir: Path) -> float:
    p = event_dir / "dt.txt"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    return float(p.read_text().strip())

def load_inputs(event_dir: Path) -> np.ndarray:
    p = event_dir / "inputs.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    x = np.loadtxt(p, dtype=float)
    # expected shape (n_in, nt)
    if x.ndim == 1:
        x = x[None, :]
    return x

def load_strain_stress_df(event_dir: Path) -> pd.DataFrame:
    p = event_dir / "strain_stress.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    return pd.read_csv(p)

def load_freq_vector(event_dir: Path, fname: str) -> np.ndarray:
    p = event_dir / fname
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    f = np.loadtxt(p, dtype=float)
    f = np.atleast_1d(f).astype(float)
    return f

def load_pred_true_processed(event_dir: Path, sid_method: str):
    pred_dir = event_dir / sid_method
    p_true = pred_dir / "outputs_true_processed.csv"
    p_pred = pred_dir / "outputs_pred_processed.csv"
    if not p_true.exists() or not p_pred.exists():
        raise FileNotFoundError(f"Missing processed outputs under {pred_dir}")

    y_true = np.loadtxt(p_true, dtype=float)
    y_pred = np.loadtxt(p_pred, dtype=float)

    # enforce shape (n_out, nt)
    if y_true.ndim == 1:
        y_true = y_true[None, :]
    if y_pred.ndim == 1:
        y_pred = y_pred[None, :]

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}")

    return y_true, y_pred


# -----------------------------
# Baseline selection by input intensity
# -----------------------------
def compute_intensity(inputs: np.ndarray, metric: str) -> float:
    """
    inputs: (n_in, nt)
    metric:
      - "pga": max over channels of max(abs(input))
      - "rms": max over channels of rms(input)
    """
    if metric == "pga":
        return float(np.max(np.abs(inputs)))
    if metric == "rms":
        rms_each = np.sqrt(np.mean(inputs**2, axis=1))
        return float(np.max(rms_each))
    raise ValueError("BASELINE_METRIC must be 'pga' or 'rms'")

def select_baseline_events(case_dir: Path, n_baseline: int, metric: str):
    ev_dirs = list_event_dirs(case_dir)
    rows = []
    for ev_dir in ev_dirs:
        ev = int(ev_dir.name)
        inputs = load_inputs(ev_dir)
        inten = compute_intensity(inputs, metric)
        rows.append({"event": ev, "intensity": inten})
    df_int = pd.DataFrame(rows).sort_values("intensity").reset_index(drop=True)
    baseline_ids = df_int.loc[: n_baseline - 1, "event"].tolist()
    return baseline_ids, df_int


# -----------------------------
# Metric computations
# -----------------------------
def compute_ksec_for_element(df: pd.DataFrame, ele_id: int) -> float:
    s_col = f"ele{ele_id}_stress"
    e_col = f"ele{ele_id}_strain"
    if s_col not in df.columns or e_col not in df.columns:
        raise KeyError(f"Missing columns {s_col} or {e_col} in strain_stress.csv")

    stress = df[s_col].to_numpy(dtype=float)
    strain = df[e_col].to_numpy(dtype=float)

    idx = int(np.argmax(np.abs(stress)))
    sigma = float(stress[idx])
    eps = float(strain[idx])

    return abs(sigma) / (abs(eps) + EPS)

def compute_kref_ele(case_dir: Path, baseline_ids, elements) -> dict:
    """
    For each element:
      kref_ele = median over baseline events of ksec_ele
    """
    kref_ele = {}
    for ele in elements:
        k_list = []
        for ev in baseline_ids:
            ev_dir = case_dir / str(ev)
            df_ss = load_strain_stress_df(ev_dir)
            k_list.append(compute_ksec_for_element(df_ss, ele))
        kref_ele[ele] = float(np.median(np.asarray(k_list, dtype=float)))
    return kref_ele

def compute_Dk_event(df_ss: pd.DataFrame, elements, kref_ele: dict) -> dict:
    """
    Per event:
      - compute ksec for each element
      - compute Dk_ele = 1 - ksec_ele/kref_ele
      - aggregate Dk_event = median(Dk_ele across elements)
    """
    out = {}
    dk_list = []
    for ele in elements:
        ksec = compute_ksec_for_element(df_ss, ele)
        kref = float(kref_ele[ele])
        dk = 1.0 - (ksec / (kref + EPS))
        out[f"ksec_ele{ele}"] = float(ksec)
        out[f"Dk_ele{ele}"] = float(dk)
        dk_list.append(dk)
    out["Dk_median"] = float(np.median(np.asarray(dk_list, dtype=float)))
    return out

def compute_fbase_per_mode(case_dir: Path, baseline_ids, freq_file: str) -> np.ndarray:
    """
    Baseline frequency per mode:
      fbase_modei = median over baseline events of f_event_modei
    """
    f_mat = []
    for ev in baseline_ids:
        ev_dir = case_dir / str(ev)
        f_vec = load_freq_vector(ev_dir, freq_file)
        if f_vec.size < 3:
            raise ValueError(f"{ev_dir/freq_file} has <3 modes")
        f_mat.append(f_vec[:3])
    f_mat = np.vstack(f_mat).astype(float)  # (n_base, 3)
    return np.median(f_mat, axis=0)

def compute_Df_event(f_event: np.ndarray, fbase_per_mode: np.ndarray) -> dict:
    """
    Per event:
      - Df_modei = 1 - f_event_i / fbase_i
      - Df_median = median(Df_mode1..3)
    """
    if f_event.size < 3:
        raise ValueError(f"Expected 3 modes, got {f_event.size}")
    if fbase_per_mode.size < 3:
        raise ValueError(f"Expected 3 baseline modes, got {fbase_per_mode.size}")

    f_event = f_event[:3].astype(float)
    fbase = fbase_per_mode[:3].astype(float)

    df_modes = 1.0 - (f_event / (fbase + EPS))
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
    """
    Residual-focused Dr using LAST tail_sec:
      residual per channel = |mean(y_true_tail) - mean(y_pred_tail)|
      Dr_residual_mean = mean(residual per channel)
      Dr_residual_max  = max(residual per channel)
    """
    nt = y_true.shape[1]
    k_tail = int(round(tail_sec / dt))
    k_tail = max(1, min(k_tail, nt))

    y_true_tail = y_true[:, -k_tail:]
    y_pred_tail = y_pred[:, -k_tail:]

    mu_true = np.mean(y_true_tail, axis=1)
    mu_pred = np.mean(y_pred_tail, axis=1)
    res = np.abs(mu_true - mu_pred)

    out = {
        "Dr_tail_sec": float(tail_sec),
        "Dr_tail_samples": int(k_tail),
        "Dr_residual_mean": float(np.mean(res)),
        "Dr_residual_max": float(np.max(res)),
    }
    for i, val in enumerate(res, start=1):
        out[f"Dr_residual_ch{i}"] = float(val)
    return out


# -----------------------------
# Main runner
# -----------------------------
def main():
    if not CASE_DIR.exists():
        raise FileNotFoundError(f"CASE_DIR not found: {CASE_DIR}")

    ev_dirs = list_event_dirs(CASE_DIR)
    if not ev_dirs:
        raise RuntimeError(f"No event folders found under {CASE_DIR}")

    # 1) auto select baseline by low intensity
    baseline_ids, df_int = select_baseline_events(CASE_DIR, N_BASELINE, BASELINE_METRIC)
    df_int["is_baseline"] = df_int["event"].isin(baseline_ids)
    df_int.to_csv(CASE_DIR / "damage_baseline_selection.csv", index=False)

    print("\nAuto baseline selection")
    print(f"  metric = {BASELINE_METRIC}")
    print(f"  N_BASELINE = {N_BASELINE}")
    print(f"  baseline events = {baseline_ids}")
    print(f"  saved: {CASE_DIR/'damage_baseline_selection.csv'}")

    # 2) build baseline references using medians
    kref_ele = compute_kref_ele(CASE_DIR, baseline_ids, ELEMENTS)
    fbase = compute_fbase_per_mode(CASE_DIR, baseline_ids, FREQ_FILE)

    print("\nBaseline references (medians over baseline events)")
    for ele in ELEMENTS:
        print(f"  k_ref ele{ele} = {kref_ele[ele]:.6g}")
    print(f"  f_base: mode1={fbase[0]:.6g}, mode2={fbase[1]:.6g}, mode3={fbase[2]:.6g}")

    # 3) compute per-event indices
    rows_k, rows_f, rows_r, rows_all = [], [], [], []

    for ev_dir in ev_dirs:
        ev = int(ev_dir.name)

        # Dk
        df_ss = load_strain_stress_df(ev_dir)
        dk_dict = compute_Dk_event(df_ss, ELEMENTS, kref_ele)
        dk_row = {"event": ev, **dk_dict}
        for ele in ELEMENTS:
            dk_row[f"kref_ele{ele}"] = float(kref_ele[ele])
        rows_k.append(dk_row)

        # Df
        f_event = load_freq_vector(ev_dir, FREQ_FILE)
        df_dict = compute_Df_event(f_event, fbase)
        df_row = {"event": ev, **df_dict}
        rows_f.append(df_row)

        # Dr (tail residual)
        dt = load_dt(ev_dir)
        y_true, y_pred = load_pred_true_processed(ev_dir, SID_METHOD)
        dr_dict = compute_Dr_residual_tail(y_true, y_pred, dt, DR_TAIL_SEC)
        dr_row = {"event": ev, **dr_dict}
        rows_r.append(dr_row)

        # merged quick table
        rows_all.append({
            "event": ev,
            "is_baseline": bool(ev in baseline_ids),
            "intensity_metric": BASELINE_METRIC,
            "Dk_median": dk_dict["Dk_median"],
            "Df_median": df_dict["Df_median"],
            "Dr_residual_mean_30s": dr_dict["Dr_residual_mean"],
            "Dr_residual_max_30s": dr_dict["Dr_residual_max"],
        })

    df_k = pd.DataFrame(rows_k).sort_values("event").reset_index(drop=True)
    df_f = pd.DataFrame(rows_f).sort_values("event").reset_index(drop=True)
    df_r = pd.DataFrame(rows_r).sort_values("event").reset_index(drop=True)
    df_all = pd.DataFrame(rows_all).sort_values("event").reset_index(drop=True)

    # 4) save results
    df_k.to_csv(CASE_DIR / "damage_Dk.csv", index=False)
    df_f.to_csv(CASE_DIR / "damage_Df.csv", index=False)
    df_r.to_csv(CASE_DIR / "damage_Dr.csv", index=False)
    df_all.to_csv(CASE_DIR / "damage_all.csv", index=False)

    print("\nSaved outputs")
    print(f"  {CASE_DIR/'damage_Dk.csv'}")
    print(f"  {CASE_DIR/'damage_Df.csv'}")
    print(f"  {CASE_DIR/'damage_Dr.csv'}")
    print(f"  {CASE_DIR/'damage_all.csv'}")

if __name__ == "__main__":
    main()
