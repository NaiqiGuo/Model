import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from model_utils import (
get_inputs, create_model, analyze, get_outputs,
simulate, stabilize_discrete, intensity_bounds, truncate_by_bounds
)

num_events = 21
sys_names = ["srim", "n4sid", "det"]
num_algos = len(sys_names)
windowed_plot = True  # Set True for windowed L2 error, False for all samples

with open("events.pkl", "rb") as f:
    events = pickle.load(f)
print(f"Total events loaded: {len(events)}")

input_channels = [1, 3]
output_dir = "predictions"
os.makedirs(output_dir, exist_ok=True)

inputs, dt = get_inputs(0, events=events, input_channels=input_channels, scale=2.54)
nt = inputs.shape[1]
model = create_model(column="forceBeamColumn", girder="forceBeamColumn", inputx=inputs[0], inputy=inputs[1], dt=dt)
disp = analyze(model, output_nodes=[9, 14, 19], nt=nt, dt=dt)
outputs = get_outputs(disp)
outputs = outputs[:, 1:]
num_channels = outputs.shape[0]

# Initialize error matrix: shape [channels, algorithms, events]
error_matrix = np.zeros((num_channels, num_algos, num_events))

for event_id in range(1, num_events+1):
    inputs, dt = get_inputs(event_id-1, events=events, input_channels=input_channels, scale=2.54)
    nt = inputs.shape[1]
    model = create_model(
        column="forceBeamColumn",
        girder="forceBeamColumn",
        inputx=inputs[0],
        inputy=inputs[1],
        dt=dt
    )
    disp = analyze(model, output_nodes=[9, 14, 19], nt=nt, dt=dt)
    outputs = get_outputs(disp)
    outputs = outputs[:, 1:]
    time = np.arange(nt) * dt
    
    for sys_idx, sys_name in enumerate(sys_names):
        # Load identified system parameters for this event and algorithm
        with open(f"system_{sys_name}_{event_id:02d}.pkl", "rb") as f:
            A, B, C, D = pickle.load(f)
        # Stabilize A (modify here if you want to try LMI or radius clipping)
        A_stable = stabilize_discrete(A)
        # Optional: post-process A (choose only one)
        #A_stable = stabilize_with_lmi(A)                
        #A_stable = stabilize_by_radius_clipping(A)
        # Simulate predicted outputs
        pred = simulate((A_stable, B, C, D), inputs)
        
        # Windowed plotting and error calculation
        if windowed_plot:
            bounds = intensity_bounds(outputs[0], lb=0.01, ub=0.99)
            outputs_trunc = truncate_by_bounds(outputs, bounds)
            pred_trunc    = truncate_by_bounds(pred, bounds)
            time_trunc    = truncate_by_bounds(time, bounds)
        else:
            outputs_trunc = outputs
            pred_trunc    = pred
            time_trunc    = time

        # Compute L2 errors for each channel
        for ch in range(num_channels):
            num = np.linalg.norm(outputs_trunc[ch] - pred_trunc[ch])
            den = np.linalg.norm(outputs_trunc[ch])
            pred_norm = np.linalg.norm(pred_trunc[ch])
            ratio = num/den if den > 1e-10 else np.nan
            error_matrix[ch, sys_idx, event_id-1] = num
            print(f"Event {event_id} {sys_name.upper()} Ch {ch}: ‖pred‖={pred_norm:.3e}, ‖diff‖={num:.3e}, ‖true‖={den:.3e}, ratio={ratio:.3f}")

        # Plot one or more channels for qualitative comparison
        fig, axs = plt.subplots(1, 2, figsize=(12,4), constrained_layout=True)
        floors = [0,1,2]  # 1F, 2F, 3F

        # X方向
        ax = axs[0]
        for f in floors:
            ax.plot(time_trunc, outputs_trunc[2*f],  '--', label=f"True Fl{f+1} X")
            ax.plot(time_trunc, pred_trunc[2*f], label=f"{sys_name.upper()} Fl{f+1} X")
        ax.set(title=f"{sys_name.upper()} — X direction", xlabel="Time (s)", ylabel="Disp")
        ax.legend()

        # Y方向
        ax = axs[1]
        for f in floors:
            ax.plot(time_trunc, outputs_trunc[2*f+1], '--', label=f"True Fl{f+1} Y")
            ax.plot(time_trunc, pred_trunc[2*f+1], label=f"{sys_name.upper()} Fl{f+1} Y")
        ax.set(title=f"{sys_name.upper()} — Y direction", xlabel="Time (s)", ylabel="Disp")
        ax.legend()

        plt.suptitle(f"Event {event_id} {sys_name.upper()}")
        plt.savefig(os.path.join(output_dir, f"event_{event_id:02d}_{sys_name}.png"), dpi=300)
        plt.close(fig)

# Save the error matrix for later analysis
np.savez(os.path.join(output_dir, "error_matrix.npz"), error_matrix=error_matrix, sys_names=sys_names)
print("Error matrix saved to predictions/error_matrix.npz")


# Visualize error matrix as heatmap for each algorithm
os.makedirs("predictions", exist_ok=True)
channel_labels = ['1F X','1F Y','2F X','2F Y','3F X','3F Y']

fig, ax = plt.subplots(figsize=(12,6), constrained_layout=True)
im = ax.imshow(error_matrix,
            vmin=0, vmax=error_matrix.max(),
            aspect='auto',
            origin='lower',
            cmap='viridis')
cbar = fig.colorbar(im, ax=ax, extend='max')
cbar.set_label("Absolute error norm", fontsize=14)

ax.set_xlabel("Event index", fontsize=14)
ax.set_ylabel("Channel", fontsize=14)
ax.set_xticks(np.arange(num_events))
ax.set_xticklabels(np.arange(1, num_events+1), rotation=45, fontsize=12)
ax.set_yticks(np.arange(num_channels))
ax.set_yticklabels(channel_labels, fontsize=12)

ax.set_title("Error heatmap (windowed part)", fontsize=16) 
for ch in range(num_channels):
    for ev in range(num_events):
        val = error_matrix[ch, ev]
        if val <= 9999:
            color = 'black' if val > error_matrix.max()/2 else 'white'
            ax.text(
                ev, ch,
                f"{val:.2f}",
                ha='center', va='center',
                color=color,
                fontsize=6
            )

plt.savefig(os.path.join(output_dir, "error_heatmap.png"), dpi=300)
plt.close(fig)
    
print("All prediction plots and error heatmaps saved.")

