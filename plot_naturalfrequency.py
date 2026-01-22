import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_natural_frequencies(csv_path, title, output_png):
    """
    Load a natural frequency CSV file (already in Hz),
    and plot before/after values for all modes versus event ID.
    """
    # Load CSV data
    data = np.genfromtxt(csv_path, delimiter=",", names=True)

    # Event IDs
    event_ids = data["event_id"]

    # Detect frequency columns
    cols = data.dtype.names
    before_cols = [c for c in cols if "_before" in c]
    after_cols  = [c for c in cols if "_after" in c]

    before_cols.sort()
    after_cols.sort()

    n_modes = len(before_cols)

    print(f"Processing file: {csv_path}")
    print(f"Detected modes: {n_modes}")
    print("Before columns:", before_cols)
    print("After columns:",  after_cols)

    if n_modes == 0:
        print("Warning: no mode columns detected. Check the column names in the CSV.")
        return

    # Plot
    plt.figure(figsize=(10, 6))

    for k in range(n_modes):

        f_before = data[before_cols[k]]   # already in Hz
        f_after  = data[after_cols[k]]    # already in Hz

        # Plot "before"
        plt.plot(
            event_ids,
            f_before,
            "o-",
            label=f"Mode {k+1} before"
        )

        # Plot "after"
        plt.plot(
            event_ids,
            f_after,
            "s--",
            label=f"Mode {k+1} after"
        )

    plt.xlabel("Event ID")
    plt.ylabel("Natural Frequency [Hz]")   # correct unit
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()

    print(f"Saved: {output_png}")

def _read_modes123(csv_path: Path) -> np.ndarray:
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing file: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
        cols = [c.lower().strip() for c in df.columns]
        if "mode1" in cols and "mode2" in cols and "mode3" in cols:
            m1 = float(df.iloc[0, cols.index("mode1")])
            m2 = float(df.iloc[0, cols.index("mode2")])
            m3 = float(df.iloc[0, cols.index("mode3")])
            return np.array([m1, m2, m3], dtype=float)
    except Exception:
        pass

    raw = np.genfromtxt(csv_path, delimiter=",", dtype=float)
    raw = np.atleast_1d(raw).flatten()
    raw = raw[~np.isnan(raw)]
    if raw.size < 3:
        raise ValueError(f"Cannot parse 3 modes from: {csv_path}. Got {raw.size} values.")
    return raw[:3].astype(float)

def export_pre_post_summary(data_dir: Path, out_csv: str,
                            pre_name: str = "pre_eq_natural_frequencies.csv",
                            post_name: str = "post_eq_natural_frequencies.csv"):
    
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"DATA_DIR not found: {data_dir}")

    ev_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    ev_dirs.sort(key=lambda d: int(d.name))
    if not ev_dirs:
        raise RuntimeError(f"No numeric event folders found under: {data_dir}")

    rows = []
    for d in ev_dirs:
        ev = int(d.name)
        pre_path = d / pre_name
        post_path = d / post_name

        pre = _read_modes123(pre_path)
        post = _read_modes123(post_path)

        rows.append({
            "event_id": ev,
            "mode1_before": pre[0],
            "mode2_before": pre[1],
            "mode3_before": pre[2],
            "mode1_after": post[0],
            "mode2_after": post[1],
            "mode3_after": post[2],
        })

    df = pd.DataFrame(rows).sort_values("event_id")
    df.to_csv(out_csv, index=False)
    print(f"Saved summary CSV: {out_csv}")


DATA_DIR_INELASTIC = Path("frame/inelastic")   
DATA_DIR_ELASTIC = Path("frame/elastic")  
export_pre_post_summary(DATA_DIR_INELASTIC, "natural_frequencies_inelastic.csv")
export_pre_post_summary(DATA_DIR_ELASTIC, "natural_frequencies_elastic.csv")    



# Call the function for elastic and inelastic

plot_natural_frequencies(
    csv_path="natural_frequencies_elastic.csv",
    title="Natural Frequencies Before/After Earthquake (Elastic Events)",
    output_png="natural_frequencies_elastic.png"
)

plot_natural_frequencies(
    csv_path="natural_frequencies_inelastic.csv",
    title="Natural Frequencies Before/After Earthquake (Inelastic Events)",
    output_png="natural_frequencies_inelastic.png"
)
