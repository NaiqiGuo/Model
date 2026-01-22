import os
import glob
import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compute_damage_intensity(strain, stress, tol=0.05, n0=200):
    """
    Compute damage intensity using actual time history.
    Linear stiffness is estimated from the initial portion of the record.
    """
    strain = np.asarray(strain)
    stress = np.asarray(stress)

    d_eps = np.diff(strain)
    d_sig = np.diff(stress)

    valid = np.abs(d_eps) > 1e-12
    slopes = d_sig[valid] / d_eps[valid]

    # ---- Estimate initial elastic stiffness from early cycles ----
    k0 = np.mean(slopes[:n0])   # initial linear stiffness

    lower = (1.0 - tol) * k0
    upper = (1.0 + tol) * k0

    nonlinear_mask = (slopes < lower) | (slopes > upper)

    damage_intensity = nonlinear_mask.sum() / slopes.size

    return damage_intensity, slopes, nonlinear_mask

if __name__ == "__main__":
    # ----------------------------------------------------------------------
    # Material properties
    # ----------------------------------------------------------------------
    fc  = 4.0  # ksi, specified unconfined concrete strength
    fpc = 5.0  # ksi, confined concrete strength (not used here for Ec)

    # Ec in ksi, using ACI-type empirical formula:
    # Ec(psi) = 57000 * sqrt(fc(psi))
    # fc is given in ksi, so we convert fc -> psi, then convert Ec back to ksi.
    Ec = 57000.0 * math.sqrt(fc * 1000.0) / 1000.0

    print(f"Ec = {Ec:.2f} ksi")

    # ----------------------------------------------------------------------
    # Loop over all inelastic events and compute damage intensity
    # ----------------------------------------------------------------------
    folder = "event_strain_stress_inelastic"   # folder with your CSV files
    pattern = os.path.join(folder, "event_*_strain_stress.csv")

    rows = []

    for fname in sorted(glob.glob(pattern)):
        df = pd.read_csv(fname)

        # Change the column names below to match your CSV.
        # For example: "ele1_strain" / "ele1_stress" or "strain" / "stress".
        strain = df["ele1_strain"].values
        stress = df["ele1_stress"].values

        damage_intensity, slopes, nonlinear_mask = compute_damage_intensity(strain, stress)

        # Extract event ID from file name, e.g. event_22_strain_stress.csv -> 22
        base = os.path.basename(fname)
        m = re.search(r"event_(\d+)_", base)
        event_id = int(m.group(1)) if m else base

        rows.append({
            "event_id": event_id,
            "file": base,
            "damage_intensity": damage_intensity,
            "n_nonlinear_steps": int(nonlinear_mask.sum()),
            "n_total_steps": int(slopes.size),
        })

    summary = pd.DataFrame(rows).sort_values("event_id")
    summary.to_csv("damage_intensity_inelastic_ele1.csv", index=False)
    print(summary)


    # ----------------------------------------------------------------------
    # Plot 1: Damage intensity vs. event ID (recommended for the paper)
    # ----------------------------------------------------------------------
    plt.figure(figsize=(10, 4))
    plt.bar(summary["event_id"], summary["damage_intensity"], width=0.8)

    plt.xlabel("Event ID")
    plt.ylabel("Damage intensity\n(# nonlinear steps / # total steps)")
    plt.title("Damage intensity from element-level stress-strain response")

    plt.xticks(summary["event_id"],  # positions
            [f"E{e:02d}" for e in summary["event_id"]],  # labels E01, E02...
            rotation=0)

    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("damage_intensity_inelastic_ele1.png", dpi=300)
    plt.close()
