import matplotlib.pyplot as plt
import numpy as np
from mdof.utilities.testing import intensity_bounds, truncate_by_bounds
from pathlib import Path
import plotly.graph_objects as go
import utilities_visualization

# Analysis configuration
WINDOWED_PLOT = True
MODEL = "bridge" # "frame", "bridge"
ELASTIC = True
MULTISUPPORT = False
VERBOSE = 1

# Main output directory
BASE_DIR = Path("Modeling")
SOURCE = "elastic" if ELASTIC else "inelastic"
MODEL_OUT_DIR = BASE_DIR / MODEL / SOURCE
FIELD_OUT_DIR = BASE_DIR / MODEL / "field"

Q_MAP = {
    "acceleration": {"name": "Acceleration", "units": "in/s²"},
    "displacement": {"name": "Displacement", "units": "in"},
        }

if __name__ == "__main__":

    if MODEL == "frame":
        if not MULTISUPPORT:
            input_labels = ['Channel 0 (X)', 'Channel 2 (Y)']
        output_nodes = [5, 5, 10, 10, 15, 15]
        output_dofs  = [1, 2, 1, 2, 1, 2]

    elif MODEL == "bridge":
        if not MULTISUPPORT:
            input_labels = ['Channel 1 (-X)', 'Channel 3 (Y)']
        output_nodes = [9, 3, 10]
        output_dofs  = [2, 2, 2]

    dof_map = {1: "X", 2: "Y", 3: "Z", 4: "RX", 5: "RY", 6: "RZ"}
    output_labels = [f"Node{n}{dof_map[d]}" for n,d in zip(output_nodes,output_dofs)]

    outputs = {"model": {}, "field": {}}
    quantities = ["displacement", "acceleration"]

    # Event ids are file names from saved field ground acceleration.
    event_files = sorted((FIELD_OUT_DIR / "acceleration" / "ground").glob("*.csv"))
    event_ids = [event.stem for event in event_files]
    for event_id in event_ids:
        print(f"Plotting Event {event_id}")

        try:
            inputs = np.loadtxt(
                FIELD_OUT_DIR / "acceleration" / "ground" / f"{event_id}.csv",
                delimiter=","
            )
            if inputs.ndim == 1:
                inputs = inputs[np.newaxis, :]

            outputs["model"]["displacement"] = np.loadtxt(
                MODEL_OUT_DIR / "displacement" / "structure" / f"{event_id}.csv",
                delimiter=","
            )
            outputs["model"]["acceleration"] = np.loadtxt(
                MODEL_OUT_DIR / "acceleration" / "structure" / f"{event_id}.csv",
                delimiter=","
            )
            outputs["field"]["displacement"] = np.loadtxt(
                FIELD_OUT_DIR / "displacement" / "structure" / f"{event_id}.csv",
                delimiter=","
            )
            outputs["field"]["acceleration"] = np.loadtxt(
                FIELD_OUT_DIR / "acceleration" / "structure" / f"{event_id}.csv",
                delimiter=","
            )
            for source in outputs:
                for q in quantities:
                    if outputs[source][q].ndim == 1:
                        outputs[source][q] = outputs[source][q][np.newaxis, :]

        except FileNotFoundError:
            if VERBOSE:
                print(f"No data for event {event_id}; skipping")
            continue

        with open(FIELD_OUT_DIR / "dt" / "ground" / f"{event_id}.txt", "r") as f:
            dt = float(f.read())

        t_in = np.arange(inputs.shape[1]) * dt
        t_out = np.arange(outputs["model"]["displacement"].shape[1]) * dt

        if WINDOWED_PLOT:
            bounds = intensity_bounds(outputs["model"]["displacement"][0], lb=0.001, ub=0.999)
            inputs = truncate_by_bounds(inputs, bounds)
            t_in = t_in[bounds[0]:bounds[1]]
            for source in outputs.keys():
                for q in quantities:
                    outputs[source][q] = truncate_by_bounds(outputs[source][q], bounds)
            t_out = t_out[bounds[0]:bounds[1]]


        # --------- input ---------
        plt.figure(figsize=(8,4))
        for ch in range(inputs.shape[0]):
            plt.plot(t_in, inputs[ch], alpha=0.7, label=input_labels[ch])
        plt.title(f"Event {event_id} Inputs")
        plt.xlabel("Time (s)")
        plt.ylabel("Input Acceleration (in/s²)")
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(FIELD_OUT_DIR / "acceleration" / "ground" / f"{event_id}.png", dpi=350)
        plt.close()

        fig = go.Figure()
        for ch in range(inputs.shape[0]):
            fig.add_scatter(x=t_in, y=inputs[ch], mode="lines", name=input_labels[ch])
        fig.update_layout(title=f"Event {event_id} Inputs",
                          xaxis_title="Time (s)",
                          yaxis_title="Input Acceleration (in/s²)",
                          width=800,height=300)
        fig.write_html(
            FIELD_OUT_DIR / "acceleration" / "ground" / f"{event_id}.html",
            include_plotlyjs="cdn"
        )

        # --------- output ---------
        source_dirs = {"model": MODEL_OUT_DIR, "field": FIELD_OUT_DIR}
        for source, source_dir in source_dirs.items():
            for q in quantities:
                plt.figure(figsize=(8,4))
                for ch in range(outputs[source][q].shape[0]):
                    plt.plot(t_out, outputs[source][q][ch], label=output_labels[ch])
                plt.title(f"Event {event_id} Outputs")
                plt.xlabel("Time (s)")
                plt.ylabel(f"Output {Q_MAP[q]['name']} ({Q_MAP[q]['units']})")
                plt.legend(loc='lower right', ncol=len(output_nodes))
                plt.tight_layout()
                plt.savefig(source_dir / q / "structure" / f"{event_id}.png", dpi=350)
                plt.close()

                fig = go.Figure()
                for ch in range(outputs[source][q].shape[0]):
                    fig.add_scatter(x=t_out, y=outputs[source][q][ch], mode="lines", name=output_labels[ch])
                fig.update_layout(title=f"Event {event_id} Outputs",
                                xaxis_title="Time (s)",
                                yaxis_title=f"Output {Q_MAP[q]['name']} ({Q_MAP[q]['units']})",
                                width=800,height=300)
                fig.write_html(
                    source_dir / q / "structure" / f"{event_id}.html",
                    include_plotlyjs="cdn"
                )
