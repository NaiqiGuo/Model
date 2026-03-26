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
import utilities_visualization
import plotly.graph_objects as go

# Analysis configuration
MODEL = "bridge" # "frame", "bridge"
SID_METHOD = "srim"
elas_cases = ["elastic", "inelastic"]  #"elastic",
OUTPUT_QUANTITY = "displacement"  # "displacement" or "acceleration"
WINDOWED = False # if true, truncates all signals before aligning, computing error, and plotting
VERBOSE = True # print extra feedback. 0 or False for no feedback; 1 or True for basic feedback; 2 for lots of feedback

Q_MAP = {
    "acceleration": {"name": "Acceleration", "units": "in/s²"},
    "displacement": {"name": "Displacement", "units": "in"},
        }

# I/O directories
MODELING_DIR = Path("Modeling")
SYSTEM_ID_DIR = Path("System ID")
SID_MODEL_DIR = SYSTEM_ID_DIR / MODEL
os.makedirs(SID_MODEL_DIR, exist_ok=True)


def modeling_path(model: str, source: str, quantity: str, location: str, event_id: int | str):
    return MODELING_DIR / model / source / quantity / location / f"{event_id}.csv"


def modeling_dt_path(model: str, event_id: int | str, location: str = "ground"):
    return MODELING_DIR / model / "field" / "dt" / location / f"{event_id}.csv"


def sid_training_dir(model: str, source: str, quantity: str, location: str):
    return SYSTEM_ID_DIR / model / source / quantity / "System ID Training Data" / location


def sid_results_dir(model: str, source: str, quantity: str, result_name: str):
    return SYSTEM_ID_DIR / model / source / quantity / "System ID Results" / result_name

if __name__ == "__main__":

    if MODEL == "frame":
        out_nodes = [5,10,15]
        # output_labels = ['1X', '1Y', '2X', '2Y', '3X', '3Y']
        out_labels = ['Floor 1, X', 'Floor 1, Y', 'Floor 2, X', 'Floor 2, Y', 'Floor 3, X', 'Floor 3, Y', ]
    elif MODEL == "bridge":
        out_nodes = [9, 3, 10]
        out_labels = ['West Deck Interface, Y', 'Column 1 Top, Y', 'East Deck Interface, Y']


    for elas in elas_cases:
        if VERBOSE:
            print(f"\nComputing {elas} case.")

        if MODEL == "frame":
            event_ids = list(range(226, 248))  # 226..247 (226, 248)
        else:
            n_events = len(glob.glob(str(MODELING_DIR / MODEL / elas / OUTPUT_QUANTITY / "structure" / "[0-9]*.csv")))
            event_ids = list(range(1, n_events+1))

        n_events = len(event_ids)
        errors = np.full((n_events, len(out_labels)), np.nan)

        for k, event_id in enumerate(tqdm(event_ids)):
            # Load true input (ground acceleration) and true output from Modeling/.
            input_path = modeling_path(MODEL, "field", "acceleration", "ground", event_id)
            out_true_path = modeling_path(MODEL, elas, OUTPUT_QUANTITY, "structure", event_id)
            dt_path = modeling_dt_path(MODEL, event_id, "ground")
            if not input_path.exists() or not out_true_path.exists() or not dt_path.exists():
                if VERBOSE:
                    print(f"skip event {event_id}: missing modeling file(s)")
                continue
            inputs = np.loadtxt(input_path)
            out_true = np.loadtxt(out_true_path)
            with open(dt_path, "r") as f:
                dt = float(f.read().strip())
            nt = inputs.shape[1]
            time = np.arange(nt) * dt
            

            # Load identified system from README System ID path.
            sys_path = sid_results_dir(MODEL, elas, OUTPUT_QUANTITY, "system realization") / f"{event_id}.pkl"
            if not sys_path.exists():
                if VERBOSE:
                    print(f"skip event {event_id}: missing system realization {sys_path}")
                continue
            with open(sys_path, "rb") as f:
                A,B,C,D = pickle.load(f)
            # Stabilize identified system
            A_stable = stabilize_discrete(A)

            # Predict outputs
            out_pred = simulate((A_stable,B,C,D), inputs)

            # Window signals
            if WINDOWED:
                sig = out_true[0].copy()
                bounds = intensity_bounds(sig, lb=0.01, ub=0.99)
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

            # Save README-compliant System ID Training Data.
            train_ground_dir = sid_training_dir(MODEL, elas, OUTPUT_QUANTITY, "ground")
            train_struct_dir = sid_training_dir(MODEL, elas, OUTPUT_QUANTITY, "structure")
            train_time_dir = sid_training_dir(MODEL, elas, OUTPUT_QUANTITY, "time")
            train_dt_dir = sid_training_dir(MODEL, elas, OUTPUT_QUANTITY, "dt")
            for d in [train_ground_dir, train_struct_dir, train_time_dir, train_dt_dir]:
                d.mkdir(parents=True, exist_ok=True)

            input_array = np.array([input_series[:nt] for input_series in inputs_trunc])
            out_true_aln_array = np.array([out_true_aln[:nt] for out_true_aln in out_true_aln_stacked])
            out_pred_aln_array = np.array([out_pred_aln[:nt] for out_pred_aln in out_pred_aln_stacked])
            time_aln = time_trunc[:nt]
            np.savetxt(train_ground_dir / f"{event_id}.csv", input_array, delimiter=",")
            np.savetxt(train_struct_dir / f"{event_id}.csv", out_true_aln_array, delimiter=",")
            np.savetxt(train_time_dir / f"{event_id}.csv", time_aln, delimiter=",")
            np.savetxt(train_dt_dir / f"{event_id}.csv", np.array([dt]))

            # Save README-compliant System ID Results.
            sid_pred_dir = sid_results_dir(MODEL, elas, OUTPUT_QUANTITY, "prediction")
            sid_pred_dir.mkdir(parents=True, exist_ok=True)
            np.savetxt(sid_pred_dir / f"{event_id}.csv", out_pred_aln_array, delimiter=",")

            sid_time_dir = sid_results_dir(MODEL, elas, OUTPUT_QUANTITY, "time")
            sid_time_dir.mkdir(parents=True, exist_ok=True)
            np.savetxt(sid_time_dir / f"{event_id}.csv", time_aln, delimiter=",")

            # Optional event-level artifacts for debugging.
            pred_dir = sid_pred_dir / SID_METHOD / str(event_id)
            pred_dir.mkdir(parents=True, exist_ok=True)
            np.savetxt(pred_dir/"inputs_processed.csv", input_array)
            np.savetxt(pred_dir/"outputs_true_processed.csv", out_true_aln_array)
            np.savetxt(pred_dir/"outputs_pred_processed.csv", out_pred_aln_array)
            np.savetxt(pred_dir/"time_processed.csv", time_aln)
            
            # Compute errors
            for i,output_label in enumerate(out_labels):
                out_true = out_true_aln_array[i]
                out_pred = out_pred_aln_array[i]
                errors[k,i] = (_get_error(
                    ytrue = out_true,
                    ypred = out_pred,
                    numerator_norm = 2,
                    denominator_norm = 2,
                    numerator_averaged = True,
                    denominator_averaged = True
                ))
            np.savetxt(pred_dir/"errors.csv", np.array(errors))

            sid_err_dir = sid_results_dir(MODEL, elas, OUTPUT_QUANTITY, "prediction error")
            sid_err_dir.mkdir(parents=True, exist_ok=True)
            np.savetxt(sid_err_dir / f"{event_id}.csv", errors[k], delimiter=",")

            # Plot true vs predicted output timeseries
            dirs = ['X','Y']
            n_rows = max(1, max(sum(d in label for label in out_labels) for d in dirs))
            fig_plt, axs = plt.subplots(n_rows, 2,
                                        figsize=(14,len(out_labels)),
                                        sharex=True,
                                        constrained_layout=True) # matplotlib
            axs = np.atleast_2d(axs)
            fig_go = [go.Figure(), go.Figure()] # plotly
            for j in range(2):
                colors_go = iter(["blue","darkorange","green"])
                row_idx = 0
                for i,out_label in enumerate(out_labels):
                    if dirs[j] in out_label:
                        color = next(colors_go)
                        axs[row_idx,j].plot(time_aln, out_true_aln_array[i], color="black", linestyle='-',  label=f"True") 
                        axs[row_idx,j].plot(time_aln, out_pred_aln_array[i], color="red", linestyle='--', label=f"Pred") 
                        axs[row_idx,j].set_ylabel(out_label)
                        axs[row_idx,j].legend()
                        fig_go[j].add_scatter(x=time_aln, y=out_true_aln_array[i],
                                              mode="lines", line=dict(color=color),
                                              name=f"True {out_label}")
                        fig_go[j].add_scatter(x=time_aln, y=out_pred_aln_array[i],
                                              mode="lines", line=dict(color=color, dash="dash"),
                                              name=f"Pred {out_label}")
                        row_idx += 1
                for r in range(n_rows):
                    axs[r,j].set_xlabel("Time (s)")
                    if r >= row_idx:
                        axs[r,j].set_visible(False)
                fig_go[j].update_layout(
                    title=f"Event {event_id} Prediction, {dirs[j]} direction",
                    xaxis_title="Time (s)",
                    yaxis_title=f"{Q_MAP[OUTPUT_QUANTITY]['name']} ({Q_MAP[OUTPUT_QUANTITY]['units']})",
                    legend=dict(orientation="h", yanchor="bottom", y=0.0, xanchor="left", x=0,
                                font=dict(size=18)),
                )
                fig_go[j].update_xaxes(rangeslider=dict(visible=True))
                fig_go[j].write_html(pred_dir/f"prediction_{dirs[j]}.html", include_plotlyjs="cdn")
            fig_plt.align_ylabels()
            fig_plt.suptitle(f"Event {event_id} {Q_MAP[OUTPUT_QUANTITY]['name']} ({Q_MAP[OUTPUT_QUANTITY]['units']})")
            fig_plt.savefig(pred_dir/"prediction.png", dpi=350)
            plt.close(fig_plt)

        # Heatmap non-square with numbers
        heatmap_dir = SYSTEM_ID_DIR / MODEL / elas / OUTPUT_QUANTITY / "System ID Results"
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
        ax.set_xticklabels(event_ids, rotation=45, fontsize=12)
        ax.set_yticks(np.arange(len(out_labels)))
        ax.set_yticklabels(out_labels, fontsize=12)
        half_vmax = np.nanmax(heatmap_data)/2.0
        for ev in range(n_events):
            for i in range(len(out_labels)):
                val = heatmap_data[i,ev]
                color = 'black' if val > half_vmax else 'white'
                ax.text(ev, i, f"{val:.2f}", ha='center', va='center', color=color, fontsize=6)
        fig.savefig(heatmap_dir / f"heatmap_{SID_METHOD}.png", dpi=400)
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
        #ax.set_xticklabels(np.arange(1, n_events+1), rotation=45, fontsize=15)
        ax.set_xticklabels(event_ids, rotation=45, fontsize=15)
        ax.set_yticks(np.arange(len(out_labels)))
        ax.set_yticklabels(out_labels, fontsize=15)
        fig.savefig(heatmap_dir / f"heatmap_square_{SID_METHOD}.png", dpi=400)
        plt.close(fig)
