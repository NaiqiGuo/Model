import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from scipy.signal import find_peaks, detrend

from utilities_experimental import (
    get_inputs, create_frame_model, get_natural_periods
)

# -----------------------------
# Configuration
# -----------------------------
EVENT_ID = 20               
INPUT_CHANNELS = [1, 3]    # The two input channels you used before
SCALE = 2.54               # Scale factor used in get_inputs
OUTDIR = os.path.join("predictions_framemodel", "q6_event04")
os.makedirs(OUTDIR, exist_ok=True)

# -----------------------------
# Utilities: PSD / FFT and Peak Extraction
# -----------------------------
def compute_psd_welch(x, fs, nperseg=None, noverlap=None):
    """
    Compute the Power Spectral Density (PSD) using Welch’s method.
    Returns frequencies f (Hz) and PSD values Pxx.
    """
    if nperseg is None:
        # Empirical rule: use 2^nextpow2 for sufficient resolution
        n = int(2 ** np.floor(np.log2(len(x))))
        nperseg = max(256, min(n, 8192))
    if noverlap is None:
        noverlap = nperseg // 2

    f, Pxx = signal.welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap,
                          detrend='linear', return_onesided=True)
    return f, Pxx

def compute_fft_power(x, fs):
    """
    Simple FFT power spectrum (one-sided). 
    Returns frequencies f (Hz) and |X|².
    """
    n = len(x)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, d=1.0/fs)
    P = (np.abs(X)**2) / n
    return f, P

def dominant_periods_from_spectrum(f, spec, k=3, fmin=0.05, fmax=None, prominence=0.0):
    """
    Extract the top k dominant peaks from the spectrum (f, spec) 
    and convert them to periods (s). 
    Filters out very low frequencies (near DC) and high-frequency noise.
    """
    if fmax is None:
        fmax = np.max(f)

    mask = (f >= fmin) & (f <= fmax)
    f_sel = f[mask]
    s_sel = spec[mask]

    if len(f_sel) < 5:
        return []

    peaks, props = find_peaks(s_sel, prominence=prominence)
    if len(peaks) == 0:
        return []

    # Sort peaks by intensity and select top k
    idx = np.argsort(s_sel[peaks])[::-1][:k]
    dom_f = f_sel[peaks][idx]
    dom_P = 1.0 / dom_f

    # Sort by frequency (optional)
    order = np.argsort(dom_f)
    dom_f = dom_f[order]
    dom_P = dom_P[order]
    return list(zip(dom_f, dom_P))  # (Hz, s)

def print_compare(dom_pairs, Tn, tag):
    """
    Print comparison between dominant input periods and model modal periods.
    """
    print(f"\n[{tag}] Dominant frequencies/periods (from low to high):")
    for i, (ff, pp) in enumerate(dom_pairs, 1):
        print(f"  Peak {i}: f = {ff:.3f} Hz  ->  T = {pp:.3f} s")

    print(f"[{tag}] Model natural periods (first 3 modes):",
          ", ".join([f"T{i+1}={Tn[i]:.3f}s" for i in range(min(3, len(Tn)))]))

    # Nearest matching between dominant and modal periods
    for i, (ff, pp) in enumerate(dom_pairs, 1):
        diffs = np.abs(Tn - pp)
        j = int(np.argmin(diffs))
        print(f"    - Peak {i} (T≈{pp:.3f}s) closest to T{j+1}={Tn[j]:.3f}s "
              f"(Δ={diffs[j]:.3f}s, {100*diffs[j]/Tn[j]:.1f}%)")

def plot_psd_with_markers(f, Pxx, dom_pairs, Tn, title, outfile):
    """
    Plot PSD (log scale) with midpoint annotations for:
      - strongest dominant period
      - closest modal period
    X-axis: Period (s)
    """
    # frequency → period
    mask = f > 0.0
    T = np.zeros_like(f)
    T[mask] = 1.0 / f[mask]

    plt.figure(figsize=(10, 5))
    plt.semilogy(T[mask], Pxx[mask], label="PSD (Welch)")

    ymax = np.max(Pxx[mask])
    y_mid1 = ymax * 0.40  
    y_mid2 = ymax * 0.20 

    # --- Only draw 1 dominant peak ---
    if dom_pairs:
        main_f, main_T = dom_pairs[0]

        # Dominant peak line
        plt.axvline(main_T, linestyle="--", color="blue", linewidth=1)

        # Annotation in the middle
        plt.text(
            main_T,
            y_mid1,
            f"T={main_T:.2f}s",
            rotation=90,
            va='center',
            ha='center',
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=2)
        )

        # --- Closest modal period ---
        diffs = np.abs(Tn - main_T)
        j = int(np.argmin(diffs))
        Ti = Tn[j]

        plt.axvline(Ti, color="black", linewidth=1.2)

        plt.text(
            Ti,
            y_mid2,
            f"T{j+1}={Ti:.2f}s",
            rotation=90,
            va='center',
            ha='center',
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=2)
        )

    plt.xlabel("Period (s)")
    plt.ylabel("PSD")
    plt.title(title)
    plt.grid(True, which='both', ls=':')
    plt.legend()

    plt.gca().invert_xaxis()  
    plt.xlim(2.0, 0.1)

    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()





# -----------------------------
# Main Routine
# -----------------------------
if __name__ == "__main__":
    with open("events.pkl", "rb") as f:
        events = pickle.load(f)
    print(f"Total events loaded: {len(events)}")

    # Load inputs for Event 4
    inputs, dt = get_inputs(EVENT_ID-1, events=events, input_channels=INPUT_CHANNELS, scale=SCALE)
    fs = 1.0 / dt
    print(f"Event {EVENT_ID}: dt = {dt:.5f}s, fs = {fs:.3f} Hz, length = {inputs.shape[1]} samples")

    # Create an elastic model to obtain modal properties (consistent with main workflow)
    model_el = create_frame_model(
        column="elasticBeamColumn",
        girder="elasticBeamColumn",
        inputx=inputs[0],
        inputy=inputs[1],
        dt=dt
    )
    Tn_el = get_natural_periods(model_el, nmodes=3)

    results_rows = []
    for idx, ch in enumerate(INPUT_CHANNELS):
        x = inputs[idx].astype(float)

        # Preprocessing: remove mean/trend to avoid low-frequency bias
        x = x - np.mean(x)
        x = detrend(x, type='linear')

        # —— PSD (Welch) —— #
        f_psd, Pxx = compute_psd_welch(x, fs=fs)

        # Optionally, switch to FFT
        # f_psd, Pxx = compute_fft_power(x, fs=fs)

        # Extract dominant peaks (ignore f<0.05Hz, i.e., T>20s)
        dom = dominant_periods_from_spectrum(f_psd, Pxx, k=3,
                                             fmin=0.05, fmax=fs/2*0.95, prominence=0.0)

        # Print comparison
        tag = f"Channel {idx} (input index {ch})"
        print_compare(dom, Tn_el, tag)

        # Plot
        out_png = os.path.join(OUTDIR, f"event{EVENT_ID:02d}_ch{idx}_psd.png")
        plot_psd_with_markers(f_psd, Pxx, dom, Tn_el,
                              title=f"Event {EVENT_ID} PSD — {tag}",
                              outfile=out_png)

        # Save to CSV
        for rank, (ff, pp) in enumerate(dom, 1):
            results_rows.append([EVENT_ID, idx, ch, rank, ff, pp])

    # Write CSV
    if results_rows:
        csv_path = os.path.join(OUTDIR, f"event{EVENT_ID:02d}_dominant_periods.csv")
        with open(csv_path, "w") as f:
            f.write("event_id,channel_idx,input_id,peak_rank,freq_hz,period_s\n")
            for row in results_rows:
                f.write(",".join(str(v) for v in row) + "\n")
        print(f"[Q6] Dominant period results saved to: {csv_path}")

    print(f"[Q6] Output figures directory: {OUTDIR}")
