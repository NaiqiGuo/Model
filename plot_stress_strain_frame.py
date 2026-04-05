import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


STRUCTURE_TYPE = "frame"
RESPONSE_TYPE = "inelastic"
ELEMENT_ID = 1


def load_stress_strain_for_event(filepath, element_id):
    """
    Load the stress-strain data for a specific event and element.

    Returns:
        strain (ndarray), stress (ndarray)
    """
    with open(filepath, "r") as f:
        reader = csv.reader(f)
        header = next(reader)

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
                continue

    if not strain_list:
        return None, None

    return np.array(strain_list), np.array(stress_list)


def iter_event_dirs(data_dir):
    return sorted(
        [d for d in data_dir.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name),
    )


def plot_stress_strain_each_event(
    element_id=ELEMENT_ID,
    structure_type=STRUCTURE_TYPE,
    response_type=RESPONSE_TYPE,
):
    """
    Plot one stress-strain curve per event for the selected structure.
    The figure is saved inside each event directory.
    """
    strain_stress_dir = Path("Modeling") / structure_type / response_type / "strain_stress"
    data_dir = strain_stress_dir / "structure"
    csv_paths = sorted(data_dir.glob("*.csv"), key=lambda p: int(p.stem))

    if not csv_paths:
        raise FileNotFoundError(f"No event csv files found in {data_dir}")

    for filepath in csv_paths:
        event_label = filepath.stem
        strain, stress = load_stress_strain_for_event(filepath, element_id)
        if strain is None:
            print(f"[Skip] No valid data in {filepath}")
            continue

        plt.figure(figsize=(6, 4))
        plt.plot(strain, stress, linewidth=1.5)
        plt.xlabel("Strain")
        plt.ylabel("Stress")
        plt.title(
            f"{structure_type.capitalize()} - Event {event_label} - "
            f"Element {element_id} Stress-Strain"
        )
        plt.grid(True)
        plt.tight_layout()

        outpath = data_dir / f"{structure_type}_{response_type}_E{event_label}_ele{element_id}.png"
        plt.savefig(outpath, dpi=300)
        plt.close()

        print(f"Saved: {outpath}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--structure-type",
        choices=["bridge", "frame"],
        default=STRUCTURE_TYPE,
        help="Which structure folder to read from.",
    )
    parser.add_argument(
        "--response-type",
        default=RESPONSE_TYPE,
        help="Response type folder, for example elastic or inelastic.",
    )
    parser.add_argument(
        "--element-id",
        type=int,
        default=ELEMENT_ID,
        help="Element id to plot.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_stress_strain_each_event(
        element_id=args.element_id,
        structure_type=args.structure_type,
        response_type=args.response_type,
    )
