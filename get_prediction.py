import pickle
import numpy as np
import matplotlib.pyplot as plt
import os, glob
from pathlib import Path
from tqdm import tqdm

from mdof.simulate import simulate
from mdof.utilities.testing import intensity_bounds, truncate_by_bounds, align_signals
from mdof.validation import stabilize_discrete
from mdof.prediction import _get_error 
from scipy.signal import correlate, correlation_lags
import utilities_visualization
import plotly.graph_objects as go

# Analysis configuration
MODEL = "frame" # "frame", "bridge"
SID_METHOD = "srim"
elas_cases = ["elastic", "inelastic"]
WINDOWED = True # if true, truncates all signals before aligning, computing error, and plotting
VERBOSE = True # print extra feedback. 0 or False for no feedback; 1 or True for basic feedback; 2 for lots of feedback

# Output directories
OUT_DIR = Path(f"{MODEL}")
os.makedirs(OUT_DIR, exist_ok=True)
for elas in elas_cases:
    elas_dir  = OUT_DIR/elas
    os.makedirs(elas_dir, exist_ok=True)

if __name__ == "__main__":

    if MODEL == "frame":
        out_nodes = [5,10,15]
        # output_labels = ['1X', '1Y', '2X', '2Y', '3X', '3Y']
        out_labels = ['Floor 1, X', 'Floor 1, Y', 'Floor 2, X', 'Floor 2, Y', 'Floor 3, X', 'Floor 3, Y', ]
    elif MODEL == "bridge":
        out_nodes = [2,3,5]
        out_labels = ['Deck, X', 'Deck, Y', 'Col 1, X', 'Col 1, Y', 'Col 2, X', 'Col 2, Y', ]
    # out_labels = [f'Node{i}{dof}' for i in out_nodes for dof in ['X','Y']]


    for elas in elas_cases:
        if VERBOSE:
            print(f"\nComputing {elas} case.")

        n_events = len(glob.glob(str(OUT_DIR/elas/"[0-9]*")))
        errors = np.full((n_events,len(out_labels)), np.nan)
        for event_id in tqdm(range(1, n_events+1)):
            # Load inputs and true outputs
            event_dir = OUT_DIR/elas/str(event_id)
            inputs = np.loadtxt(event_dir/"inputs.csv")
            out_true = np.loadtxt(event_dir/"outputs.csv")
            with open(event_dir/"dt.txt", "r") as f:
                dt = float(f.read())
            nt = inputs.shape[1]
            time = np.arange(nt) * dt

            # Load identified system
            with open(event_dir/f"system_{SID_METHOD}.pkl", "rb") as f:
                A,B,C,D = pickle.load(f)
            # Stabilize identified system
            A_stable = stabilize_discrete(A)

            # Predict outputs
            out_pred = simulate((A_stable,B,C,D), inputs)

            # Window signals
            if WINDOWED:
                bounds = intensity_bounds(out_true[0], lb=0.01, ub=0.99)
                inputs_trunc = truncate_by_bounds(inputs, bounds)
                out_true_trunc = truncate_by_bounds(out_true, bounds)
                out_pred_trunc = truncate_by_bounds(out_pred, bounds)
                time_trunc     = truncate_by_bounds(time, bounds)
            else:
                inputs_trunc = inputs
                out_true_trunc = out_true
                out_pred_trunc = out_pred
                time_trunc     = time

            # Align signals
            max_lag_allowed_sec = 1.0

            if VERBOSE==2:
                print(f">>> Aligning signals for Event {event_id:02d}.")

            out_true_aln_stacked = []
            out_pred_aln_stacked = []
            for i, output_label in enumerate(out_labels):
                s1 = out_true_trunc[i]
                s2 = out_pred_trunc[i]
                lag, out_true_aln, out_pred_aln, _ = align_signals(s1, s2, time_trunc, 
                                                            verbose=False, 
                                                            max_lag_allowed=max_lag_allowed_sec)
                out_true_aln_stacked.append(out_true_aln)
                out_pred_aln_stacked.append(out_pred_aln)

                if VERBOSE==2:
                    if lag == 0:
                            print(f">>>>>> {output_label}: no shift (already aligned).")
                    else:
                        lag_sec = lag*dt
                        direction = "ypred originally lagged ytrue" if lag > 0 else "ypred originally led ytrue"
                        print(f">>>>>> {output_label}: shifted {lag:+d} samples ({lag_sec:+.4f} s) → {direction}")
                    
                
                nt = np.min([nt, len(out_true_aln)])
            
            # Prediction directory
            pred_dir  = event_dir/SID_METHOD
            os.makedirs(pred_dir, exist_ok=True)

            # Save processed input, true output, predicted output, and time
            input_array = np.array([input_series[:nt] for input_series in inputs_trunc])
            np.savetxt(pred_dir/"inputs_processed.csv", input_array)
            out_true_aln_array = np.array([out_true_aln[:nt] for out_true_aln in out_true_aln_stacked])
            np.savetxt(pred_dir/"outputs_true_processed.csv", out_true_aln_array)
            out_pred_aln_array = np.array([out_pred_aln[:nt] for out_pred_aln in out_pred_aln_stacked])
            np.savetxt(pred_dir/"outputs_pred_processed.csv", out_pred_aln_array)
            time_aln = time_trunc[:nt]
            np.savetxt(pred_dir/"time_processed.csv", time_aln)
            
            # Compute errors
            for i,output_label in enumerate(out_labels):
                out_true = out_true_aln_array[i]
                out_pred = out_pred_aln_array[i]
                errors[event_id-1,i] = (_get_error(
                    ytrue = out_true,
                    ypred = out_pred,
                    numerator_norm = 2,
                    denominator_norm = 2,
                    numerator_averaged = True,
                    denominator_averaged = True
                ))
            np.savetxt(pred_dir/"errors.csv", np.array(errors))

            # Plot true vs predicted output timeseries
            fig_plt, axs = plt.subplots(int(len(out_labels)/2), 2,
                                        figsize=(14,len(out_labels)),
                                        sharex=True,
                                        constrained_layout=True) # matplotlib
            fig_go = [go.Figure(), go.Figure()] # plotly
            dirs = ['X','Y']
            for j in range(2):
                colors_go = iter(["blue","darkorange","green"])
                for i,out_label in enumerate(out_labels):
                    if dirs[j] in out_label:
                        r = i//2
                        color = next(colors_go)
                        axs[r,j].plot(time_aln, out_true_aln_array[i], color="black", linestyle='-',  label=f"True") 
                        axs[r,j].plot(time_aln, out_pred_aln_array[i], color="red", linestyle='--', label=f"Pred") 
                        axs[r,j].set_ylabel(out_label)
                        axs[r,j].legend()
                        fig_go[j].add_scatter(x=time_aln, y=out_true_aln_array[i],
                                              mode="lines", line=dict(color=color),
                                              name=f"True {out_label}")
                        fig_go[j].add_scatter(x=time_aln, y=out_pred_aln_array[i],
                                              mode="lines", line=dict(color=color, dash="dash"),
                                              name=f"Pred {out_label}")
                axs[r,j].set_xlabel(f"Time (s)")
                fig_go[j].update_layout(
                    title=f"Event {event_id} Prediction, {dirs[j]} direction",
                    xaxis_title="Time (s)",
                    yaxis_title="Displacement (in)",
                    legend=dict(orientation="h", yanchor="bottom", y=0.0, xanchor="left", x=0,
                                font=dict(size=18)),
                )
                fig_go[j].update_xaxes(rangeslider=dict(visible=True))
                fig_go[j].write_html(pred_dir/f"prediction_{dirs[j]}.html", include_plotlyjs="cdn")
            fig_plt.align_ylabels()
            fig_plt.suptitle(f"Event {event_id} Displacement Response (in)")
            fig_plt.savefig(pred_dir/"prediction.png", dpi=350)
            plt.close(fig_plt)

        # Heatmap non-square with numbers
        heatmap_dir = OUT_DIR/elas/SID_METHOD
        os.makedirs(heatmap_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(12,6), constrained_layout=True)
        heatmap_data = np.nan_to_num(errors.T, nan=0.0)
        if MODEL == "frame":
            vmax = 1.0
        elif MODEL == "bridge":
            vmax = np.max(heatmap_data)
        im = ax.imshow(
            heatmap_data,
            vmin=0, vmax=vmax,
            aspect='auto',
            origin='lower',
            cmap='viridis'
        )
        cbar = fig.colorbar(im, ax=ax, extend='max')
        cbar.set_label("$\\epsilon$: Normalized $L_2$ Error", fontsize=14)
        ax.set_xlabel("Event", fontsize=14)
        ax.set_xticks(np.arange(n_events))
        ax.set_xticklabels(np.arange(1, n_events+1), rotation=45, fontsize=12)
        ax.set_yticks(np.arange(len(out_labels)))
        ax.set_yticklabels(out_labels, fontsize=12)
        half_vmax = np.nanmax(heatmap_data)/2.0
        for ev in range(n_events):
            for i in range(len(out_labels)):
                val = heatmap_data[i,ev]
                color = 'black' if val > half_vmax else 'white'
                ax.text(ev, i, f"{val:.2f}", ha='center', va='center', color=color, fontsize=6)
        fig.savefig(heatmap_dir/f"heatmap.png", dpi=400)
        plt.close(fig)

        # Heatmap square with no numbers
        fig, ax = plt.subplots(figsize=(12,6), constrained_layout=True)
        im = ax.imshow(
            heatmap_data,
            vmin=0, vmax=vmax,
            aspect='equal',
            origin='lower',
            cmap='viridis'
        )
        cbar = fig.colorbar(im, ax=ax, extend='max', fraction=0.02, pad=0.04)
        cbar.set_label("$\\epsilon$: Normalized $L_2$ Error", fontsize=20)
        cbar.ax.tick_params(labelsize=15)
        ax.set_xlabel("Event", fontsize=22)
        ax.set_xticks(np.arange(n_events))
        ax.set_xticklabels(np.arange(1, n_events+1), rotation=45, fontsize=15)
        ax.set_yticks(np.arange(len(out_labels)))
        ax.set_yticklabels(out_labels, fontsize=15)
        fig.savefig(heatmap_dir/f"heatmap_square.png", dpi=400)
        plt.close(fig)

