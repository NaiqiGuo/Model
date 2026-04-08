import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


STRUCTURE_TYPE = "bridge" # bridge, frame
RESPONSE_TYPE = "elastic" # elastic, inelastic 
ELEMENT_ID = 3


DEFAULTS_BY_STRUCTURE = {
    "bridge": {"response_type": "elastic", "element_id": 3},
    "frame": {"response_type": "inelastic", "element_id": 1},
}


def load_stress_strain_for_event(filepath, element_id):
    """
    Load stress-strain data for one event and element.

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


def iter_legacy_event_dirs(data_dir):
    return sorted(
        [d for d in data_dir.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name),
    )


def iter_modeling_event_csvs(data_dir):
    return sorted(
        [p for p in data_dir.glob("*.csv") if p.stem.isdigit()],
        key=lambda p: int(p.stem),
    )


def detect_layout(structure_type, response_type, requested_layout="auto"):
    layout_paths = {
        "legacy": Path(structure_type) / response_type,
        "modeling": Path("Modeling") / structure_type / response_type / "strain_stress" / "structure",
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

    legacy_dir = layout_paths["legacy"]
    if legacy_dir.exists() and iter_legacy_event_dirs(legacy_dir):
        return "legacy", legacy_dir

    raise FileNotFoundError(
        "Could not find stress-strain data in either supported layout:\n"
        f"  - {legacy_dir}\n"
        f"  - {modeling_dir}"
    )


def iter_event_sources(data_dir, layout):
    if layout == "modeling":
        for filepath in iter_modeling_event_csvs(data_dir):
            yield filepath.stem, filepath
        return

    for event_dir in iter_legacy_event_dirs(data_dir):
        yield event_dir.name, event_dir / "strain_stress.csv"


def build_output_path(data_dir, layout, structure_type, response_type, element_id, event_label):
    output_dir = Path("Modeling") / structure_type / response_type / "strain_stress" / "structure"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{structure_type}_{response_type}_E{event_label}_ele{element_id}.png"


def plot_stress_strain_each_event(structure_type, response_type, element_id, layout="auto"):
    """
    Plot one stress-strain curve per event for the selected structure.
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

        outpath = build_output_path(
            data_dir=data_dir,
            layout=resolved_layout,
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
        help="Response type folder, for example elastic, inelastic, or field.",
    )
    parser.add_argument(
        "--element-id",
        type=int,
        default=ELEMENT_ID,
        help="Element id to plot.",
    )
    parser.add_argument(
        "--layout",
        choices=["auto", "legacy", "modeling"],
        default="auto",
        help="Which input directory layout to use. Default: auto-detect.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    plot_stress_strain_each_event(
        structure_type=args.structure_type,
        response_type=args.response_type,
        element_id=args.element_id,
        layout=args.layout,
    )


if __name__ == "__main__":
    main()
