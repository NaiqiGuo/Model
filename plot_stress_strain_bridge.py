import os
import glob
import csv
import numpy as np
import matplotlib.pyplot as plt


DATA_DIR = "event_strain_stress_bridge_elastic"   # Folder containing event_XX_strain_stress.csv
ELEMENT_ID = 2  


def load_stress_strain_for_event(filepath, element_id):
    """
    Load the stress-strain data for a specific event and a specific element.

    Returns:
        strain (ndarray), stress (ndarray)
    """
    with open(filepath, "r") as f:
        reader = csv.reader(f)
        header = next(reader)

        # Column names for this element
        stress_col = f"ele{element_id}_stress"
        strain_col = f"ele{element_id}_strain"

        try:
            idx_stress = header.index(stress_col)
            idx_strain = header.index(strain_col)
        except ValueError:
            print(f"[WARN] {filepath} does not contain required columns for element {element_id}.")
            return None, None

        strain_list, stress_list = [], []

        for row in reader:
            if not row:
                continue
            try:
                strain_list.append(float(row[idx_strain]))
                stress_list.append(float(row[idx_stress]))
            except ValueError:
                # skip bad lines
                continue

    if not strain_list:
        return None, None

    return np.array(strain_list), np.array(stress_list)


def plot_stress_strain_each_event(element_id=ELEMENT_ID, data_dir=DATA_DIR):
    """
    Plot one stress-strain curve per event for the Painter bridge model.
    Each event produces one figure PNG in fig_stress_strain_bridge_each_event.
    """
    pattern = os.path.join(data_dir, "event_*_strain_stress.csv")
    paths = sorted(glob.glob(pattern))

    if not paths:
        raise FileNotFoundError(f"No files found matching {pattern}")

    # Output folder
    out_dir = "fig_stress_strain_bridge_each_event"
    os.makedirs(out_dir, exist_ok=True)

    for filepath in paths:
        # Extract event label from filename
        # event_01_strain_stress.csv → "01"
        filename = os.path.basename(filepath)
        parts = filename.split("_")
        if len(parts) < 3:
            print(f"[WARN] Unexpected filename format: {filename}")
            continue
        event_label = parts[1]

        strain, stress = load_stress_strain_for_event(filepath, element_id)
        if strain is None:
            print(f"[Skip] No valid data in {filename}")
            continue

        # idx = np.argsort(strain)
        # strain = strain[idx]; stress = stress[idx]

        # Plot
        plt.figure(figsize=(6, 4))
        plt.plot(strain, stress, linewidth=1.5)

        plt.xlabel("Strain")
        plt.ylabel("Stress")
        plt.title(f"Bridge - Event {event_label} - Element {element_id} Stress-Strain")
        plt.grid(True)
        plt.tight_layout()

        # Save figure
        outpath = os.path.join(out_dir, f"bridge_E{event_label}_ele{element_id}.png")
        plt.savefig(outpath, dpi=300)
        plt.close()

        print(f"Saved: {outpath}")


if __name__ == "__main__":
    plot_stress_strain_each_event(ELEMENT_ID, DATA_DIR)
