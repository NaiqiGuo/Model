import numpy as np
import matplotlib.pyplot as plt

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

        f_before = data[before_cols[k]]   # in Hz
        f_after  = data[after_cols[k]]    # in Hz

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
    plt.ylabel("Natural Frequency [Hz]")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()

    print(f"Saved: {output_png}")


# BRIDGE

plot_natural_frequencies(
    csv_path="natural_frequencies_bridge_elastic.csv",
    title="Bridge Model - Natural Frequencies Before/After (Elastic)",
    output_png="bridge_natural_frequencies_elastic.png"
)

plot_natural_frequencies(
    csv_path="natural_frequencies_bridge_inelastic.csv",
    title="Bridge Model - Natural Frequencies Before/After (Inelastic)",
    output_png="bridge_natural_frequencies_inelastic.png"
)
