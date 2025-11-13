import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

from model_utils import (
    get_inputs, create_frame_model, analyze, get_outputs,
    stabilize_with_lmi, stabilize_by_radius_clipping,
    simulate, stabilize_discrete, intensity_bounds, truncate_by_bounds, record_strain_step,
    plot_q4_max_strain, get_natural_periods, plot_deltaT_across_events
)
from mdof.utilities.testing import align_signals
import scienceplots
plt.style.use(["science"])
from scipy.signal import correlate, correlation_lags
from mdof.prediction import _get_error 

try:
    import plotly.graph_objects as go
    PLOTLY_OK = True
except Exception as e:
    print("Plotly not available, skip interactive plots:", e)
    PLOTLY_OK = False


num_events = 21
sys_names = ["srim"]
num_algos = len(sys_names)
windowed_plot = True

with open("events.pkl", "rb") as f:
    events = pickle.load(f)
print(f"Total events loaded: {len(events)}")

input_channels = [1, 3]
output_dir = "predictions_framemodel"
os.makedirs(output_dir, exist_ok=True)

inputs, dt = get_inputs(0, events=events, input_channels=input_channels, scale=2.54)
nt = inputs.shape[1]
model = create_frame_model(
    column="elasticBeamColumn",
    girder="elasticBeamColumn",
    inputx=inputs[0],
    inputy=inputs[1],
    dt=dt
)
disp = analyze(model, output_nodes=[5, 10, 15], nt=nt, dt=dt, step_callback=record_strain_step)
outputs = get_outputs(disp)
outputs = outputs[:, 1:]
num_channels = outputs.shape[0]

# Initialize error matrix: shape [channels, algorithms, events]
#error_matrix = np.zeros((num_channels, num_algos, num_events))
error_matrix = np.full((num_channels, num_algos, num_events), np.nan)

DO_Q5 = True
DO_INELASTIC = True  
dT_mode1_inel = []   # Collect the first mode’s ΔT/T (%) and plot it across events
ev_ids_inel  = []

for event_id in range(1, num_events+1):
    inputs, dt = get_inputs(event_id-1, events=events, input_channels=input_channels, scale=2.54)
    nt = inputs.shape[1]
    
    model = create_frame_model(
        column="elasticBeamColumn",
        girder="elasticBeamColumn",
        inputx=inputs[0],
        inputy=inputs[1],
        dt=dt
    )

    
    #q5
    T_before_el = get_natural_periods(model, nmodes=3) if DO_Q5 else None  

    try:
        disp = analyze(model, output_nodes=[5, 10, 15], nt=nt, dt=dt, step_callback=record_strain_step)
    except RuntimeError:
        continue

    T_after_el = get_natural_periods(model, nmodes=3) if DO_Q5 else None 

    if DO_Q5 and T_before_el is not None and T_after_el is not None:
        dT_pct_el = (T_after_el - T_before_el) / T_before_el * 100.0
        print(f"[Q5][Elastic] Event {event_id:02d} ΔT/T (%):", np.array2string(dT_pct_el, precision=3))
        if np.nanmax(np.abs(dT_pct_el)) > 1.0:
            print(f"[WARN] Elastic model period changed >1% in Event {event_id:02d}. Check stiffness/damping setup.")

    

    if DO_INELASTIC:
        # build inelastic model
        model_inel = create_frame_model(
            column="forceBeamColumn",
            girder="elasticBeamColumn",
            inputx=inputs[0],
            inputy=inputs[1],
            dt=dt
        )
        # periods before earthquake
        T0_inel = get_natural_periods(model_inel, nmodes=3) if DO_Q5 else None

        try:
            disp_inel = analyze(model_inel, output_nodes=[5, 10, 15], nt=nt, dt=dt, step_callback=record_strain_step)
        except RuntimeError:
            pass
        else:
            # periods after earthquake
            T1_inel = get_natural_periods(model_inel, nmodes=3) if DO_Q5 else None

            if DO_Q5 and T0_inel is not None and T1_inel is not None:
                dT_pct = (T1_inel - T0_inel) / T0_inel * 100.0
                print(f"[Q5][Inelastic] Event {event_id:02d} ΔT/T (%):", np.array2string(dT_pct, precision=3))
                dT_mode1_inel.append(float(dT_pct[0]))
                ev_ids_inel.append(event_id)
        


    outputs = get_outputs(disp)
    outputs = outputs[:, 1:]
    time = np.arange(outputs.shape[1]) * dt


    plotly_dir = os.path.join(output_dir, "plotly_html")
    os.makedirs(plotly_dir, exist_ok=True)

    have_aligned = False     
    last_sys_name = None
    
    param_dir = "event_outputs_ABCD_frame_model"
    for sys_idx, sys_name in enumerate(sys_names):
        # Load identified system parameters for this event and algorithm
        pkl_path = os.path.join(param_dir, f"system_{sys_name}_{event_id:02d}.pkl")
        if not os.path.exists(pkl_path):
            print(f"{pkl_path} pass")
            continue
        with open(pkl_path, "rb") as f:
            A, B, C, D = pickle.load(f)
        # Stabilize A (modify here if you want to try LMI or radius clipping)
        A_stable = stabilize_discrete(A)
        # Optional: post-process A (choose only one)
        #A_stable = stabilize_with_lmi(A)                
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

        max_lag_samp_global, ytrue_aln, ypred_aln, time_aln = align_signals(
            outputs_trunc, pred_trunc, times=time_trunc, verbose=False, max_lag_allowed=None
        )

        have_aligned = True
        last_sys_name = sys_name


        lags_samp = np.empty(num_channels, dtype=int)
        for ch in range(num_channels):
            s1 = outputs_trunc[ch]
            s2 = pred_trunc[ch]
            corr = correlate(s1, s2, mode='full')
            lags = correlation_lags(len(s1), len(s2), mode='full')
            lags_samp[ch] = int(lags[np.argmax(corr)])

        max_lag_allowed_sec = 1.0
        for ch in range(num_channels):
            max_lag_samp = int(lags_samp[ch])
            lag_sec = max_lag_samp * dt
            ch_name = f"Ch{ch}" 
            if abs(lag_sec) > max_lag_allowed_sec:
                print(f"[ALIGN] E{event_id:02d} {ch_name}: LARGE LAG {lag_sec:+.3f}s "
                    f"({max_lag_samp:+d} samples) — exceeds ±{max_lag_allowed_sec:.1f}s limit. NO ALIGN APPLIED.")
            elif max_lag_samp == 0:
                print(f"[ALIGN] E{event_id:02d} {ch_name}: no shift (already aligned).")
            else:
                direction = "ypred originally lagged ytrue" if max_lag_samp > 0 else "ypred originally led ytrue"
                print(f"[ALIGN] E{event_id:02d} {ch_name}: shifted {max_lag_samp:+d} samples "
                    f"({lag_sec:+.4f} s) → {direction}")
                
        
        
        
                
        for ch in range(num_channels):
            y_t = ytrue_aln[ch]
            y_p = ypred_aln[ch]
            try:
                val = _get_error(
                ytrue=y_t,
                ypred=y_p,
                numerator_norm=2,
                denominator_norm= 2,
                numerator_averaged=True,
                denominator_averaged=True
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

    
    # q4 plot the maximum strain time history — elastic case
    if PLOTLY_OK:
        sr_el = model.meta["strain_record"]
        plot_q4_max_strain(
            sr_el, model,
            title=f"Event {event_id:02d} — Max Strain (Elastic)",
            html_base=os.path.join(plotly_dir, f"event_{event_id:02d}_strain_elastic"),
            PLOTLY_OK=PLOTLY_OK
        )

    # q4 plot the maximum strain time history — inelastic case
    if DO_INELASTIC and 'model_inel' in locals() and hasattr(model_inel, 'meta'):
        sr_inel = model_inel.meta.get("strain_record", None)
        if sr_inel:
            plot_q4_max_strain(
                sr_inel, model_inel,
                title=f"Event {event_id:02d} — Max Strain (Inelastic)",
                html_base=os.path.join(plotly_dir, f"event_{event_id:02d}_strain_inelastic"),
                PLOTLY_OK=PLOTLY_OK
            )

# after run all events
if DO_Q5 and DO_INELASTIC and dT_mode1_inel:
    plot_deltaT_across_events(  
        ev_ids_inel,
        dT_mode1_inel,
        plotly_dir,
        title="ΔT/T of Mode-1 per Event (Inelastic)"
    )

    # === Plotly interactive time series (drop-in) ===
    

    if PLOTLY_OK and have_aligned:
        floors = [0, 1, 2]  # 1F, 2F, 3F
        algo_tag = (last_sys_name or "model").upper()

        # --- X direction ---
        figx = go.Figure()
        for f in floors:
            ch_true = 2 * f
            ch_pred = 2 * f
            figx.add_scatter(
                x=time_aln, y=ytrue_aln[ch_true],
                mode="lines", name=f"True Fl{f+1} X", line=dict(dash="dash")
            )
            figx.add_scatter(
                x=time_aln, y=ypred_aln[ch_pred],
                mode="lines", name=f"{algo_tag} Fl{f+1} X"
            )
        figx.update_layout(
            title=f"Event {event_id} {algo_tag} — X direction (interactive)",
            xaxis_title="Time (s)",
            yaxis_title="Disp",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        figx.update_xaxes(rangeslider=dict(visible=True))
        html_x = os.path.join(plotly_dir, f"event_{event_id:02d}_{algo_tag}_X.html")
        figx.write_html(html_x, include_plotlyjs="cdn")

        # --- Y direction ---
        figy = go.Figure()
        for f in floors:
            ch_true = 2 * f + 1
            ch_pred = 2 * f + 1
            figy.add_scatter(
                x=time_aln, y=ytrue_aln[ch_true],
                mode="lines", name=f"True Fl{f+1} Y", line=dict(dash="dash")
            )
            figy.add_scatter(
                x=time_aln, y=ypred_aln[ch_pred],
                mode="lines", name=f"{algo_tag} Fl{f+1} Y"
            )
        figy.update_layout(
            title=f"Event {event_id} {algo_tag} — Y direction (interactive)",
            xaxis_title="Time (s)",
            yaxis_title="Disp",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        figy.update_xaxes(rangeslider=dict(visible=True))
        html_y = os.path.join(plotly_dir, f"event_{event_id:02d}_{algo_tag}_Y.html")
        figy.write_html(html_y, include_plotlyjs="cdn")

        print(f"[PLOTLY] Saved: {html_x} and {html_y}")
    else:
        if PLOTLY_OK and not have_aligned:
            print(f"[PLOTLY] E{event_id:02d}: skip (no ABCD/aligned series for this event).")


# Save the error matrix for later analysis
np.savez(os.path.join(output_dir, "error_matrix.npz"), error_matrix=error_matrix, sys_names=sys_names)
print(f"Error matrix saved to {os.path.join(output_dir, 'error_matrix.npz')}")



# Visualize error matrix as heatmap for each algorithm
os.makedirs(output_dir, exist_ok=True)
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
    #ax.set_title(f"{algo} Error heatmap (windowed part)", fontsize=16)
    half = np.nanmax(data_display) / 2.0
    for ch in range(len(channel_labels)):
        for ev in range(num_events):
            val = data_display[ch, ev]
            color = 'black' if val > half else 'white'
            ax.text(ev, ch, f"{val:.2f}", ha='center', va='center',
                    color=color, fontsize=6)
    plt.savefig(os.path.join(output_dir, f"error_heatmap_{algo}.png"), dpi=300)
    plt.close(fig)

    


# === Visualization for error_matrix ===
os.makedirs(output_dir, exist_ok=True)
channel_labels = ['1F X','1F Y','2F X','2F Y','3F X','3F Y']
print("error_matrix shape:", error_matrix.shape)

algo_names = [name.upper() for name in sys_names]
DROP10 = False  

# Unified colorbar range (ignore NaN)
vmax_global = np.nanmax(error_matrix)
print(f"Global color scale vmax = {vmax_global:.4f}")

for i, algo in enumerate(algo_names):
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)

    # Replace NaN with 0 (for visualization only)
    data = np.nan_to_num(error_matrix[:, i, :], nan=0.0)

    # Plot heatmap
    im = ax.imshow(
        data,
        #vmin=0, vmax=vmax_global, 
        vmin=0, vmax=1.2,  
        aspect='equal',             
        origin='lower',
        cmap='viridis'
    )

    # Colorbar settings
    cbar = fig.colorbar(im, 
                        ax=ax, 
                        extend='max', 
                        fraction=0.02, 
                        pad=0.04
                        )
    cbar.set_label(
        "$\\epsilon$:$L_2$ Error (mean) \n(normalized)",
        fontsize=20, fontname='serif'
    )
    cbar.ax.tick_params(labelsize=15)

    # Axis labels and font
    ax.set_xlabel("Event index", fontsize=22)
    ax.set_ylabel("Channel", fontsize=22)
    ax.set_xticks(np.arange(num_events))
    ax.set_yticks(np.arange(len(channel_labels)))
    ax.set_yticklabels(channel_labels, fontsize=15)

    # X-axis labels
    if DROP10:
        ax.set_xticklabels(np.delete(np.arange(1, num_events + 2), 9),
                           rotation=45, fontsize=15)
    else:
        ax.set_xticklabels(np.arange(1, num_events + 1),
                           rotation=45, fontsize=15)

    # Tick style
    ax.tick_params(axis='both', which='major',
                   direction='out', length=3, width=0.8,
                   top=False, right=False, pad=6)
    ax.tick_params(which='minor', direction='out', length=0)

    # Title (optional)
    # ax.set_title(f"{algo} Error heatmap (windowed part)",
    #              fontsize=18, fontname='serif')

    # Save figure
    if DROP10:
        fig.savefig(os.path.join(output_dir, f"{algo}_heatmap_drop10.png"), dpi=300)
    else:
        fig.savefig(os.path.join(output_dir, f"{algo}_heatmap.png"), dpi=300)

    plt.close(fig)

print(f"All error heatmaps for each algorithm have been saved in: {output_dir}")

