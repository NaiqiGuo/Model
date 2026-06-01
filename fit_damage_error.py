from pathlib import Path
import os
import csv

import numpy as np
import matplotlib.pyplot as plt


DEFAULT_DAMAGE_CSV = Path("Damage/bridge/inelastic/displacement/damage_all.csv")
DAMAGE_CSV = Path(os.environ.get("FIT_DAMAGE_CSV", str(DEFAULT_DAMAGE_CSV)))
USE_DK_FD = os.environ.get("FIT_USE_DK_FD", "1") == "1"

DK_COLUMN = "Dk_fd_median" if USE_DK_FD else "Dk_median"
DF_COLUMN = os.environ.get("FIT_DF_COLUMN", "Df_median")
DR_COLUMN = os.environ.get("FIT_DR_COLUMN", "Dr_residual_mean_30s")
Y_COLUMN = os.environ.get("FIT_Y_COLUMN", "Error_mean")

OUT_DIR = Path(os.environ.get("FIT_OUT_DIR", str(DAMAGE_CSV.parent / "fit_error")))
VERBOSE = os.environ.get("FIT_VERBOSE", "1") == "1"


def load_rows(path: Path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def as_float(row, key):
    value = row.get(key, "")
    if value in ("", None):
        return np.nan
    try:
        return float(value)
    except ValueError:
        return np.nan


def linear_fit(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    residuals = y - y_pred
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = np.nan if np.isclose(ss_tot, 0.0) else 1.0 - ss_res / ss_tot
    return slope, intercept, y_pred, residuals, r_squared


def save_summary(path: Path, slope, intercept, r_squared, n_samples):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["damage_csv", str(DAMAGE_CSV)])
        writer.writerow(["dk_column", DK_COLUMN])
        writer.writerow(["df_column", DF_COLUMN])
        writer.writerow(["dr_column", DR_COLUMN])
        writer.writerow(["y_column", Y_COLUMN])
        writer.writerow(["n_samples", n_samples])
        writer.writerow(["slope_a", float(slope)])
        writer.writerow(["intercept_b", float(intercept)])
        writer.writerow(["r_squared", float(r_squared)])
        writer.writerow([
            "equation",
            f"{Y_COLUMN} = {float(slope):.6g} * D_comb + {float(intercept):.6g}",
        ])


def save_predictions(path: Path, events, dk_vals, df_vals, dr_vals, d_comb, y_true, y_pred, residuals):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "event",
            DK_COLUMN,
            DF_COLUMN,
            DR_COLUMN,
            "D_comb",
            "actual_error_mean",
            "predicted_error_mean",
            "residual",
        ])
        for row in zip(events, dk_vals, df_vals, dr_vals, d_comb, y_true, y_pred, residuals):
            writer.writerow(row)


def save_scatter(path: Path, d_comb, y_true, y_pred, slope, intercept, r_squared):
    fig, ax = plt.subplots(figsize=(7.2, 5.8), constrained_layout=True)
    ax.scatter(d_comb, y_true, s=55, color="black", alpha=0.85, label="Events")

    x_line = np.linspace(float(np.min(d_comb)), float(np.max(d_comb)), 200)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, color="red", linestyle="--", linewidth=2.0, label="Linear fit")

    ax.set_xlabel(r"$D_{\mathrm{comb}}$", fontsize=16)
    ax.set_ylabel("Error_mean", fontsize=16)
    ax.set_title(
        rf"$D_{{\mathrm{{comb}}}} = \sqrt{{{DF_COLUMN}^2 + {DK_COLUMN}^2 + {DR_COLUMN}^2}}$"
        + "\n"
        + rf"$E_{{\mathrm{{mean}}}} = {slope:.4g} D_{{\mathrm{{comb}}}} + {intercept:.4g}$, $R^2 = {r_squared:.4f}$",
        fontsize=15,
        fontweight="bold",
    )
    ax.legend(frameon=False, fontsize=12)
    ax.tick_params(axis="both", labelsize=12)
    fig.savefig(path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    if not DAMAGE_CSV.exists():
        raise FileNotFoundError(f"Damage table not found: {DAMAGE_CSV}")

    rows = load_rows(DAMAGE_CSV)
    required = ["event", DK_COLUMN, DF_COLUMN, DR_COLUMN, Y_COLUMN]
    missing = [col for col in required if rows and col not in rows[0]]
    if missing:
        raise KeyError(f"Missing required columns in {DAMAGE_CSV}: {missing}")

    events = []
    dk_vals = []
    df_vals = []
    dr_vals = []
    y_vals = []
    skipped = 0
    for row in rows:
        dk = as_float(row, DK_COLUMN)
        df = as_float(row, DF_COLUMN)
        dr = as_float(row, DR_COLUMN)
        y = as_float(row, Y_COLUMN)
        if any(np.isnan(v) for v in [dk, df, dr, y]):
            skipped += 1
            continue
        events.append(row["event"])
        dk_vals.append(dk)
        df_vals.append(df)
        dr_vals.append(dr)
        y_vals.append(y)

    if len(events) < 2:
        raise ValueError(f"Not enough valid rows to fit. Found {len(events)} valid rows after skipping {skipped}.")

    dk_vals = np.asarray(dk_vals, dtype=float)
    df_vals = np.asarray(df_vals, dtype=float)
    dr_vals = np.asarray(dr_vals, dtype=float)
    y_vals = np.asarray(y_vals, dtype=float)

    d_comb = np.sqrt(df_vals**2 + dk_vals**2 + dr_vals**2)
    slope, intercept, y_pred, residuals, r_squared = linear_fit(d_comb, y_vals)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "dkfd" if USE_DK_FD else "dk"
    summary_path = OUT_DIR / f"fit_summary_{suffix}.csv"
    pred_path = OUT_DIR / f"fit_predictions_{suffix}.csv"
    fig_path = OUT_DIR / f"fit_dcomb_vs_error_{suffix}.png"

    save_summary(summary_path, slope, intercept, r_squared, len(events))
    save_predictions(pred_path, events, dk_vals, df_vals, dr_vals, d_comb, y_vals, y_pred, residuals)
    save_scatter(fig_path, d_comb, y_vals, y_pred, slope, intercept, r_squared)

    if VERBOSE:
        print(f"Loaded damage table: {DAMAGE_CSV}")
        print(f"Using D_comb = sqrt({DF_COLUMN}^2 + {DK_COLUMN}^2 + {DR_COLUMN}^2)")
        print(f"Using y column: {Y_COLUMN}")
        print(f"Valid rows: {len(events)}; skipped rows: {skipped}")
        print(f"Slope a: {slope:.6g}")
        print(f"Intercept b: {intercept:.6g}")
        print(f"R^2: {r_squared:.6g}")
        print(f"Saved summary: {summary_path}")
        print(f"Saved predictions: {pred_path}")
        print(f"Saved figure: {fig_path}")
