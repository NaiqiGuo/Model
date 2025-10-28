import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from model_utils import (
get_inputs, create_model, analyze, get_outputs, stabilize_with_lmi,stabilize_by_radius_clipping,
simulate, stabilize_discrete, intensity_bounds, truncate_by_bounds
)
from mdof.prediction import get_error_new
from mdof.utilities.testing import align_signals
import scienceplots
plt.style.use(["science"])

num_events = 21
sys_names = ["srim"] #, "det", "n4sid", "det"
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
#error_matrix = np.zeros((num_channels, num_algos, num_events))
error_matrix = np.full((num_channels, num_algos, num_events), np.nan)

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
    try:
        disp = analyze(model, output_nodes=[9, 14, 19], nt=nt, dt=dt)
    except RuntimeError:
        continue
    outputs = get_outputs(disp)
    outputs = outputs[:, 1:]
    time = np.arange(outputs.shape[1]) * dt
    
    param_dir = "event_outputs_ABCDk"
    for sys_idx, sys_name in enumerate(sys_names):
        # Load identified system parameters for this event and algorithm
        pkl_path = os.path.join(param_dir, f"system_{sys_name}_{event_id:02d}.pkl")
        if not os.path.exists(pkl_path):
            print(f"{pkl_path} pass")
            continue
        with open(pkl_path, "rb") as f:
            A, B, C, D = pickle.load(f)
        # Stabilize A (modify here if you want to try LMI or radius clipping)
        #A_stable = stabilize_discrete(A)
        # Optional: post-process A (choose only one)
        A_stable = stabilize_with_lmi(A)                
        #A_stable = stabilize_by_radius_clipping(A)
        # Simulate predicted outputs
        pred = simulate((A_stable, B, C, D), inputs)
        pred = pred[:, 1:]
        
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
        # for ch in range(num_channels):
        #     num = np.linalg.norm(outputs_trunc[ch] - pred_trunc[ch])
        #     den = np.linalg.norm(outputs_trunc[ch])
        #     pred_norm = np.linalg.norm(pred_trunc[ch])
        #     ratio = num/den if den > 1e-10 else np.nan
        #     error_matrix[ch, sys_idx, event_id-1] = num
        #     print(f"Event {event_id} {sys_name.upper()} Ch {ch}: ‖pred‖={pred_norm:.3e}, ‖diff‖={num:.3e}, ‖true‖={den:.3e}, ratio={ratio:.3f}")

        _, ytrue_aln, ypred_aln, time_aln = align_signals(
            outputs_trunc, pred_trunc, times=time_trunc, verbose=False, max_lag_allowed=None
        )
        for ch in range(num_channels):
            y_t = ytrue_aln[ch]
            y_p = ypred_aln[ch]
            try:
                val = get_error_new(
                    y_true=y_t,
                    y_pred=y_p,
                    normalized=True,
                    denominator_norm=2,
                    numerator_norm=2,
                    averaged=True,
                    averaging_mode='mean'
                )
            except ZeroDivisionError:
                val = np.nan
            error_matrix[ch, sys_idx, event_id - 1] = float(val)
            

        # Plot one or more channels for qualitative comparison
        fig, axs = plt.subplots(1, 2, figsize=(12,4), constrained_layout=True)
        floors = [0,1,2]  # 1F, 2F, 3F

        # X
        ax = axs[0]
        for f in floors:
            ax.plot(time_aln, ytrue_aln[2*f],   '--', label=f"True Fl{f+1} X")
            ax.plot(time_aln, ypred_aln[2*f],         label=f"{sys_name.upper()} Fl{f+1} X")
        ax.set(title=f"{sys_name.upper()} — X direction", xlabel="Time (s)", ylabel="Disp")
        ax.legend()

        # Y
        ax = axs[1]
        for f in floors:
            ax.plot(time_aln, ytrue_aln[2*f+1], '--', label=f"True Fl{f+1} Y")
            ax.plot(time_aln, ypred_aln[2*f+1],       label=f"{sys_name.upper()} Fl{f+1} Y")
        ax.set(title=f"{sys_name.upper()} — Y direction", xlabel="Time (s)", ylabel="Disp")
        ax.legend()

        plt.suptitle(f"Event {event_id} {sys_name.upper()} (aligned, windowed={windowed_plot})")
        plt.savefig(os.path.join(output_dir, f"event_{event_id:02d}_{sys_name}.png"), dpi=300)
        plt.close(fig)

# Save the error matrix for later analysis
np.savez(os.path.join(output_dir, "error_matrix.npz"), error_matrix=error_matrix, sys_names=sys_names)
print("Error matrix saved to predictions/error_matrix.npz")


# Visualize error matrix as heatmap for each algorithm
os.makedirs("predictions", exist_ok=True)
channel_labels = ['1F X','1F Y','2F X','2F Y','3F X','3F Y']
print("error_matrix shape:", error_matrix.shape)

algo_names = [name.upper() for name in sys_names]
for i, algo in enumerate(algo_names):
    fig, ax = plt.subplots(figsize=(12,6), constrained_layout=True)
    data = error_matrix[:, i, :]
    data_display = np.nan_to_num(data, nan=0.0)
    vmax_i = np.nanmax(error_matrix[:, i, :])
    im = ax.imshow(error_matrix[:, i, :],
                   vmin=0, vmax=vmax_i,
                   aspect='auto', origin='lower', cmap='viridis')
    cbar = fig.colorbar(im, ax=ax, extend='max')
    cbar.set_label("$\epsilon$:$L_2$ Error (mean) \n(normalized)", fontsize=14)
    ax.set_xlabel("Event index", fontsize=14)
    ax.set_ylabel("Channel", fontsize=14)
    ax.set_xticks(np.arange(num_events))
    ax.set_xticklabels(np.arange(1, num_events+1), rotation=45, fontsize=12)
    ax.set_yticks(np.arange(len(channel_labels)))
    ax.set_yticklabels(channel_labels, fontsize=12)
    ax.set_title(f"{algo} Error heatmap (windowed part)", fontsize=16)
    half = np.nanmax(data_display) / 2.0
    for ch in range(len(channel_labels)):
        for ev in range(num_events):
            val = data_display[ch, ev]
            color = 'black' if val > half else 'white'
            ax.text(ev, ch, f"{val:.2f}", ha='center', va='center',
                    color=color, fontsize=6)
    plt.savefig(os.path.join("predictions", f"error_heatmap_{algo}.png"), dpi=300)
    plt.close(fig)


# === Visualization for error_matrix ===
os.makedirs("predictions", exist_ok=True)
channel_labels = ['1F X','1F Y','2F X','2F Y','3F X','3F Y']
print("error_matrix shape:", error_matrix.shape)

algo_names = [name.upper() for name in sys_names]
DROP10 = False  # 改成 True 可跳过第10事件标签

# 统一色条范围（忽略 NaN）
vmax_global = np.nanmax(error_matrix)
print(f"Global color scale vmax = {vmax_global:.4f}")

for i, algo in enumerate(algo_names):
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)

    # 替换 NaN → 0（仅显示）
    data = np.nan_to_num(error_matrix[:, i, :], nan=0.0)

    # 绘制热图
    im = ax.imshow(
        data,
        vmin=0, vmax=vmax_global,   # 统一色条范围
        aspect='equal',             # 方块比例
        origin='lower',
        cmap='viridis'
    )

    # 色条设置
    cbar = fig.colorbar(im, ax=ax, extend='max', fraction=0.02, pad=0.04)
    cbar.set_label(
        "$\\epsilon$:$L_2$ Error (mean) \n(normalized)",
        fontsize=20, fontname='serif'
    )
    cbar.ax.tick_params(labelsize=15)

    # 坐标轴与字体
    ax.set_xlabel("Event index", fontsize=22)
    ax.set_ylabel("Channel", fontsize=22)
    ax.set_xticks(np.arange(num_events))
    ax.set_yticks(np.arange(len(channel_labels)))
    ax.set_yticklabels(channel_labels, fontsize=15)

    # X轴标签
    if DROP10:
        ax.set_xticklabels(np.delete(np.arange(1, num_events + 2), 9),
                           rotation=45, fontsize=15)
    else:
        ax.set_xticklabels(np.arange(1, num_events + 1),
                           rotation=45, fontsize=15)
    # Tick 样式
    ax.tick_params(axis='both', which='major',
                   direction='out', length=3, width=0.8,
                   top=False, right=False, pad=6)
    ax.tick_params(which='minor', direction='out', length=0)
    # 标题
    # ax.set_title(f"{algo} Error heatmap (windowed part)",
    #              fontsize=18, fontname='serif')
    if DROP10:
        fig.savefig(os.path.join("predictions", f"{algo}_heatmap_drop10.png"), dpi=300)
    else:
        fig.savefig(os.path.join("predictions", f"{algo}_heatmap.png"), dpi=300)

    plt.close(fig)

print("All error heatmaps for each algorithm have been saved in the predictions folder.")


