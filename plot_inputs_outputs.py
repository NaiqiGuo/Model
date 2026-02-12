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
MODEL = "bridge" # "frame", "bridge"
ELASTIC = True
MULTISUPPORT = False

# Main output directory
OUT_DIR = Path(f"{MODEL}")/("elastic" if ELASTIC else "inelastic")

if __name__ == "__main__":

    if MODEL == "frame":
        if not MULTISUPPORT:
            input_labels = ['Channel 0 (X)', 'Channel 2 (Y)']
        output_nodes = [5,10,15]
        # output_labels = ['1X', '1Y', '2X', '2Y', '3X', '3Y']
        # output_labels = ['Floor 1, X', 'Floor 1, Y', 'Floor 2, X', 'Floor 2, Y', 'Floor 3, X', 'Floor 3, Y', ]
    elif MODEL == "bridge":
        if not MULTISUPPORT:
            input_labels = ['Channel 1 (X)', 'Channel 3 (Y)']
        output_nodes = [2,3,5]
        # output_labels = ['Deck, X', 'Deck, Y', 'Col 1, X', 'Col 1, Y', 'Col 2, X', 'Col 2, Y', ]
    output_labels = [f'Node{i}{dof}' for i in output_nodes for dof in ['X','Y']]
        
    event_files = glob.glob(str(OUT_DIR/"[0-9]*"))
    event_ids = [Path(event).stem.replace("ce249Run", "") for event in event_files]
    for event_id in event_ids:
        print(f"Plotting Event {event_id}")
        event_dir = OUT_DIR/str(event_id)

        inputs = np.loadtxt(event_dir/"inputs.csv")
        outputs_displ = np.loadtxt(event_dir/"outputs.csv")
        outputs_accel = np.loadtxt(event_dir/"outputs_acc.csv")
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
            t_out_trunc = t_out[bounds[0]:bounds[1]]
        else:
            inputs_trunc = inputs
            t_in_trunc = t_in
            outputs_displ_trunc = outputs_displ
            outputs_accel_trunc = outputs_accel
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
            plt.plot(t_out_trunc, outputs_displ_trunc[ch], alpha=0.7, label=output_labels[ch])
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

        # CHECK NG: Added acceleration output plots
        
        # --------- output, acceleration ---------
        plt.figure(figsize=(8,4))
        for ch in range(outputs_accel_trunc.shape[0]):
            plt.plot(t_out_trunc, outputs_accel_trunc[ch], alpha=0.7, label=output_labels[ch])
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