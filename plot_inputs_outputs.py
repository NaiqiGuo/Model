import matplotlib.pyplot as plt
import numpy as np
from mdof.utilities.testing import intensity_bounds, truncate_by_bounds
import os
from pathlib import Path
import glob
import plotly.graph_objects as go
import utilities_visualization

# Analysis configuration
WINDOWED_PLOT = True
MODEL = "frame" # "frame", "bridge"
ELASTIC = False
MULTISUPPORT = False
VERBOSE = 1

# Main output directory
OUT_DIR = Path(f"{MODEL}")/("elastic" if ELASTIC else "inelastic")
#TODO CC: check 
FIELD_OUT_DIR = Path(MODEL) / "field"

if __name__ == "__main__":

    if MODEL == "frame":
        # TODO NG: Change this to require both output_nodes and output_dofs. Each output must
        # be separately and explicitly defined, e.g.
        # output_nodes = [5,5,10,10,15,15]
        # output_dofs = [1,2,1,2,1,2]
        if not MULTISUPPORT:
            input_labels = ['Channel 0 (X)', 'Channel 2 (Y)']
        output_nodes = [5, 5, 10, 10, 15, 15]
        output_dofs  = [1, 2, 1, 2, 1, 2]
        output_labels = ['Node 5, X', 'Node 5, Y', 'Node 10, X', 'Node 10, Y', 'Node 15, X', 'Node 15, Y']

    elif MODEL == "bridge":
        if not MULTISUPPORT:
            input_labels = ['Channel 1 (X)', 'Channel 3 (Y)']
        output_nodes = [9, 9, 3, 3, 10, 10]
        output_dofs  = [1, 2, 1, 2, 1, 2]
        output_labels = ['Node 9, X', 'Node 9, Y', 'Node 3, X', 'Node 3, Y', 'Node 10, X', 'Node 10, Y']

    # TODO NG: Fix this accordingly
    dof_map = {1: "X", 2: "Y", 3: "Z", 4: "RX", 5: "RY", 6: "RZ"}
    output_labels = [f"Node{n}{dof_map[d]}" for n, d in zip(output_nodes, output_dofs)]

        
    event_files = glob.glob(str(OUT_DIR/"[0-9]*"))
    event_ids = [Path(event).stem.replace("ce249Run", "") for event in event_files]
    for event_id in event_ids:
        print(f"Plotting Event {event_id}")
        event_dir = OUT_DIR/str(event_id)
        #TODO CC: check 
        field_event_dir = FIELD_OUT_DIR / str(event_id)

        try:
            inputs = np.loadtxt(event_dir/"inputs.csv")
            outputs_displ = np.loadtxt(event_dir/"outputs_displ.csv")
            outputs_accel = np.loadtxt(event_dir/"outputs_accel.csv")
            outputs_displ_field = np.loadtxt(field_event_dir/"outputs_displ_field.csv")
            outputs_accel_field = np.loadtxt(field_event_dir/"outputs_accel_field.csv")

        except FileNotFoundError:
            if VERBOSE:
                print(f"No data for event {event_id}; skipping")
            continue

        with open(event_dir/"dt.txt", "r") as f:
            dt = float(f.read())
        nt = inputs.shape[1]

        t_in = np.arange(inputs.shape[1]) * dt
        t_out = np.arange(outputs_displ.shape[1]) * dt

        if WINDOWED_PLOT:
            bounds = intensity_bounds(outputs_displ[0], lb=0.001, ub=0.999)
            inputs_trunc = truncate_by_bounds(inputs, bounds)
            t_in_trunc = t_in[bounds[0]:bounds[1]]
            outputs_displ_trunc = truncate_by_bounds(outputs_displ, bounds)
            outputs_accel_trunc = truncate_by_bounds(outputs_accel, bounds)
            outputs_displ_field_trunc = truncate_by_bounds(outputs_displ_field, bounds)
            outputs_accel_field_trunc = truncate_by_bounds(outputs_accel_field, bounds)
            t_out_trunc = t_out[bounds[0]:bounds[1]]
        else:
            inputs_trunc = inputs
            t_in_trunc = t_in
            outputs_displ_trunc = outputs_displ
            outputs_accel_trunc = outputs_accel
            outputs_displ_field_trunc = outputs_displ_field
            outputs_accel_field_trunc = outputs_accel_field
            t_out_trunc = t_out

        # --------- input ---------
        plt.figure(figsize=(8,4))
        for ch in range(inputs_trunc.shape[0]):
            plt.plot(t_in_trunc, inputs_trunc[ch], alpha=0.7, label=input_labels[ch])
        plt.title(f"Event {event_id} Inputs")
        plt.xlabel("Time (s)")
        plt.ylabel("Input Acceleration (in/s²)")
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(event_dir/"inputs.png", dpi=350)
        plt.close()

        fig = go.Figure()
        for ch in range(inputs_trunc.shape[0]):
            fig.add_scatter(x=t_in_trunc, y=inputs_trunc[ch], mode="lines", name=input_labels[ch])
        fig.update_layout(title=f"Event {event_id} Inputs",
                          xaxis_title="Time (s)",
                          yaxis_title="Input Acceleration (in/s²)",
                          width=800,height=300)
        fig.write_html(event_dir/"inputs.html", include_plotlyjs="cdn")

        # --------- output, displacement ---------
        plt.figure(figsize=(8,4))
        for ch in range(outputs_displ_trunc.shape[0]):
            plt.plot(t_out_trunc, outputs_displ_trunc[ch], alpha=1.0, label=output_labels[ch])
        plt.title(f"Event {event_id} Outputs")
        plt.xlabel("Time (s)")
        plt.ylabel("Output Displacement (in)")
        plt.legend(loc='lower right', ncol=len(output_nodes))
        plt.tight_layout()
        plt.savefig(event_dir/"outputs.png", dpi=350)
        plt.close()

        fig = go.Figure()
        for ch in range(outputs_displ_trunc.shape[0]):
            fig.add_scatter(x=t_out_trunc, y=outputs_displ_trunc[ch], mode="lines", name=output_labels[ch])
        fig.update_layout(title=f"Event {event_id} Outputs",
                          xaxis_title="Time (s)",
                          yaxis_title="Output Displacement (in)",
                          width=800,height=300)
        fig.write_html(event_dir/"outputs.html", include_plotlyjs="cdn")
        
        # --------- output, acceleration ---------
        plt.figure(figsize=(8,4))
        for ch in range(outputs_accel_trunc.shape[0]):
            plt.plot(t_out_trunc, outputs_accel_trunc[ch], alpha=1.0, label=output_labels[ch])
        plt.title(f"Event {event_id} Outputs")
        plt.xlabel("Time (s)")
        plt.ylabel("Output Acceleration (in/s²)")
        plt.legend(loc='lower right', ncol=len(output_nodes))
        plt.tight_layout()
        plt.savefig(event_dir/"outputs_acc.png", dpi=350)
        plt.close()

        fig = go.Figure()
        for ch in range(outputs_accel_trunc.shape[0]):
            fig.add_scatter(x=t_out_trunc, y=outputs_accel_trunc[ch], mode="lines", name=output_labels[ch])
        fig.update_layout(title=f"Event {event_id} Outputs",
                          xaxis_title="Time (s)",
                          yaxis_title="Output Acceleration (in/s²)",
                          width=800,height=300)
        fig.write_html(event_dir/"outputs_acc.html", include_plotlyjs="cdn")

        # --------- field output, displacement ---------
        plt.figure(figsize=(8,4))
        for ch in range(outputs_displ_field_trunc.shape[0]):
            label = output_labels[ch] if ch < len(output_labels) else f"Field {ch}"
            plt.plot(t_out_trunc, outputs_displ_field_trunc[ch], alpha=1.0, label=label)
        plt.title(f"Event {event_id} Field Outputs (Displacement)")
        plt.xlabel("Time (s)")
        plt.ylabel("Field Displacement (in)")
        plt.legend(loc='lower right', ncol=min(len(output_labels), 3))
        plt.tight_layout()
        plt.savefig(field_event_dir/"outputs_field_displ.png", dpi=350)
        plt.close()

        fig = go.Figure()
        for ch in range(outputs_displ_field_trunc.shape[0]):
            label = output_labels[ch] if ch < len(output_labels) else f"Field {ch}"
            fig.add_scatter(x=t_out_trunc, y=outputs_displ_field_trunc[ch], mode="lines", name=label)
        fig.update_layout(title=f"Event {event_id} Field Outputs (Displacement)",
                        xaxis_title="Time (s)",
                        yaxis_title="Field Displacement (in)",
                        width=800, height=300)
        fig.write_html(field_event_dir/"outputs_field_displ.html", include_plotlyjs="cdn")

        # --------- field output, acceleration ---------
        plt.figure(figsize=(8,4))
        for ch in range(outputs_accel_field_trunc.shape[0]):
            label = output_labels[ch] if ch < len(output_labels) else f"Field {ch}"
            plt.plot(t_out_trunc, outputs_accel_field_trunc[ch], alpha=1.0, label=label)
        plt.title(f"Event {event_id} Field Outputs (Acceleration)")
        plt.xlabel("Time (s)")
        plt.ylabel("Field Acceleration (in/s²)")
        plt.legend(loc='lower right', ncol=min(len(output_labels), 3))
        plt.tight_layout()
        plt.savefig(field_event_dir/"outputs_field_accel.png", dpi=350)
        plt.close()

        fig = go.Figure()
        for ch in range(outputs_accel_field_trunc.shape[0]):
            label = output_labels[ch] if ch < len(output_labels) else f"Field {ch}"
            fig.add_scatter(x=t_out_trunc, y=outputs_accel_field_trunc[ch], mode="lines", name=label)
        fig.update_layout(title=f"Event {event_id} Field Outputs (Acceleration)",
                        xaxis_title="Time (s)",
                        yaxis_title="Field Acceleration (in/s²)",
                        width=800, height=300)
        fig.write_html(field_event_dir/"outputs_field_accel.html", include_plotlyjs="cdn")
