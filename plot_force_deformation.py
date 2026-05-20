import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


STRUCTURE_TYPE = "bridge"  # bridge, frame
RESPONSE_TYPE = "inelastic"  # elastic, inelastic
ELEMENT_ID = 107


DEFAULTS_BY_STRUCTURE = {
    "bridge": {"response_type": "inelastic", "element_id": 107},
    "frame": {"response_type": "inelastic", "element_id": 1},
}


def load_force_deformation_for_event(filepath, element_id):
    """
    Load force-deformation data for one event and element.

    Returns:
        deformation (ndarray), force (ndarray)
    """
    with open(filepath, "r") as f:
        reader = csv.reader(f)
        header = next(reader)

        force_col = f"ele{element_id}_force"
        deformation_col = f"ele{element_id}_deformation"

        try:
            idx_force = header.index(force_col)
            idx_deformation = header.index(deformation_col)
        except ValueError:
            print(f"[WARN] {filepath} does not contain required columns for element {element_id}.")
            return None, None

        deformation_list, force_list = [], []
        for row in reader:
            if not row:
                continue
            try:
                deformation_list.append(float(row[idx_deformation]))
                force_list.append(float(row[idx_force]))
            except ValueError:
                continue

    if not deformation_list:
        return None, None

    return np.array(deformation_list), np.array(force_list)


def iter_modeling_event_csvs(data_dir):
    return sorted(
        [p for p in data_dir.glob("*.csv") if p.stem.isdigit()],
        key=lambda p: int(p.stem),
    )


def detect_layout(structure_type, response_type, requested_layout="auto"):
    layout_paths = {
        "modeling": Path("Modeling") / structure_type / response_type / "force_deformation" / "structure",
    }

    if requested_layout != "auto":
        data_dir = layout_paths[requested_layout]
        if not data_dir.exists():
            raise FileNotFoundError(
                f"Requested {requested_layout} layout, but data directory does not exist: {data_dir}"
            )
        return requested_layout, data_dir

    modeling_dir = layout_paths["modeling"]
    if modeling_dir.exists() and iter_modeling_event_csvs(modeling_dir):
        return "modeling", modeling_dir

    raise FileNotFoundError(
        "Could not find force-deformation data in the supported layout:\n"
        f"  - {modeling_dir}"
    )


def iter_event_sources(data_dir, layout):
    if layout == "modeling":
        for filepath in iter_modeling_event_csvs(data_dir):
            yield filepath.stem, filepath
        return


def build_output_path(structure_type, response_type, element_id, event_label):
    output_dir = Path("Modeling") / structure_type / response_type / "force_deformation" / "structure"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{structure_type}_{response_type}_E{event_label}_ele{element_id}_fd.png"


def plot_force_deformation_each_event(structure_type, response_type, element_id, layout="auto"):
    """
    Plot one force-deformation curve per event for the selected structure.
    """
    resolved_layout, data_dir = detect_layout(
        structure_type=structure_type,
        response_type=response_type,
        requested_layout=layout,
    )
    print(f"[Info] Using {resolved_layout} layout from {data_dir}")

    plotted_any = False

    for event_label, filepath in iter_event_sources(data_dir, resolved_layout):
        if not filepath.exists():
            print(f"[Skip] Missing {filepath}")
            continue

        deformation, force = load_force_deformation_for_event(filepath, element_id)
        if deformation is None:
            print(f"[Skip] No valid data in {filepath}")
            continue

        plt.figure(figsize=(6, 4))
        plt.plot(deformation, force, linewidth=1.5)
        plt.xlabel("Deformation")
        plt.ylabel("Force")
        plt.title(
            f"{structure_type.capitalize()} - Event {event_label} - "
            f"Element {element_id} Force-Deformation"
        )
        plt.grid(True)
        plt.tight_layout()

        outpath = build_output_path(
            structure_type=structure_type,
            response_type=response_type,
            element_id=element_id,
            event_label=event_label,
        )
        plt.savefig(outpath, dpi=300)
        plt.close()

        plotted_any = True
        print(f"Saved: {outpath}")

    if not plotted_any:
        raise RuntimeError(
            f"No plots were created for structure={structure_type}, "
            f"response={response_type}, element={element_id}"
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--structure-type",
        choices=sorted(DEFAULTS_BY_STRUCTURE),
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
    parser.add_argument(
        "--layout",
        choices=["auto", "modeling"],
        default="auto",
        help="Which input directory layout to use. Default: auto-detect.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    plot_force_deformation_each_event(
        structure_type=args.structure_type,
        response_type=args.response_type,
        element_id=args.element_id,
        layout=args.layout,
    )


if __name__ == "__main__":
    main()
