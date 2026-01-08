"""
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
from model_utils import (
    list_event_dirs, load_dt, load_inputs, load_strain_stress_df, load_freq_vector, load_pred_true_processed,
    compute_intensity, select_baseline_events, compute_ksec_for_element, compute_kref_ele, compute_Dk_event,
    compute_fbase_per_mode, compute_Df_event, compute_Dr_residual_tail
)

# -----------------------------
# User config
# -----------------------------
CASE_DIR = Path("/Users/guonaiqi/Documents/UCB/299/Example5-python/Model/bridge/inelastic") #frame bridge
SID_METHOD = "srim"

# Elements used for Dk 
ELEMENTS = [2, 3] #(1, 5, 9) (2, 3, 5)

# Baseline selection
N_BASELINE = 5                 # pick lowest-intensity N events as baseline
BASELINE_METRIC = "pga"        # "pga" (peak abs accel), or "rms"

# Df frequency source
FREQ_FILE = "post_eq_natural_frequencies.csv"   # has 3 modes per event

# Dr tail window (seconds)
DR_TAIL_SEC = 10.0


if __name__ == "__main__":
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


    
