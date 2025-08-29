import pickle
import numpy as np
import matplotlib.pyplot as plt
from model_utils import simulate, stabilize_discrete, intensity_bounds, truncate_by_bounds

num_events = 21
sys_names = ["srim", "n4sid", "det", "okid"]
num_algos = len(sys_names)
windowed_plot = True  # Set True for windowed L2 error, False for all samples

# We'll assume number of channels is known (or can be obtained from one output file)
test_data = np.load("event_data/event_1.npz")
num_channels = test_data["outputs"].shape[0]

# Initialize error matrix: shape [channels, algorithms, events]
error_matrix = np.zeros((num_channels, num_algos, num_events))

for event_id in range(1, num_events+1):
    # Load event data (inputs, outputs, time)
    data = np.load(f"event_data/event_{event_id}.npz")
    inputs = data["inputs"]
    outputs = data["outputs"]
    time = data["time"]
    
    for sys_idx, sys_name in enumerate(sys_names):
        # Load identified system parameters for this event and algorithm
        with open(f"system_{sys_name}_{event_id:02d}.pkl", "rb") as f:
            A, B, C, D = pickle.load(f)
        # Stabilize A (modify here if you want to try LMI or radius clipping)
        A_stable = stabilize_discrete(A)
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
        plt.figure(figsize=(10, 6))
        for ch in range(num_channels):
            plt.subplot(num_channels, 1, ch+1)
            plt.plot(time_trunc, outputs_trunc[ch], label="True", color="black", linewidth=1.5)
            plt.plot(time_trunc, pred_trunc[ch], label=f"Pred ({sys_name.upper()})", alpha=0.7)
            plt.ylabel(f"Ch {ch+1}")
            if ch == 0:
                plt.title(f"Event {event_id}, {sys_name.upper()}: Prediction vs True")
            if ch == num_channels-1:
                plt.xlabel("Time (s)")
            plt.legend()
        plt.tight_layout()
        plt.savefig(f"predictions/event_{event_id:02d}_{sys_name}.png", dpi=200)
        plt.close()

# Save the error matrix for later analysis
np.savez("predictions/error_matrix.npz", error_matrix=error_matrix, sys_names=sys_names)
print("Error matrix saved to predictions/error_matrix.npz")

# Visualize error matrix as heatmap for each algorithm
import os
os.makedirs("predictions", exist_ok=True)
for sys_idx, sys_name in enumerate(sys_names):
    plt.figure(figsize=(10, 4))
    im = plt.imshow(error_matrix[:, sys_idx, :], aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(im, label="L2 error")
    plt.xlabel("Event index")
    plt.ylabel("Channel")
    plt.title(f"L2 error heatmap: {sys_name.upper()}")
    plt.xticks(np.arange(num_events), np.arange(1, num_events+1))
    plt.yticks(np.arange(num_channels), [f"Ch {ch+1}" for ch in range(num_channels)])
    plt.tight_layout()
    plt.savefig(f"predictions/error_heatmap_{sys_name}.png", dpi=200)
    plt.close()
print("All prediction plots and error heatmaps saved.")

