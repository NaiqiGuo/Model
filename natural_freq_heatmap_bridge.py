import numpy as np
import matplotlib.pyplot as plt


# Load Δf/f for bridge model
def load_df_over_f(csv_path):
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    cols = data.dtype.names

    before_cols = [c for c in cols if "before_s" in c]
    after_cols  = [c for c in cols if "after_s"  in c]
    before_cols.sort()
    after_cols.sort()

    # shape = (N_events, N_modes)
    f_before = np.column_stack([data[c] for c in before_cols])
    f_after  = np.column_stack([data[c] for c in after_cols])

    # Δf/f = (f_before - f_after) / f_before
    df_over_f = (f_before - f_after) / f_before

    return data["event_id"], df_over_f


# PLOT FOR BRIDGE ELASTIC VERSION 

event_el, df_el = load_df_over_f("natural_frequencies_bridge_elastic.csv")
mode1_el = df_el[:, 0]     # mode 1

plt.figure(figsize=(12, 2.5))
plt.imshow(mode1_el[np.newaxis, :],
           aspect="auto",
           cmap="Blues",
           vmin=-0.3, vmax=0.3)
plt.colorbar(label="Δf/f_before")
plt.xticks(range(len(event_el)), event_el, rotation=90)
plt.yticks([0], ["Mode 1"])
plt.title("Bridge Elastic Model – Mode 1 Δf/f")
plt.tight_layout()
plt.savefig("bridge_heatmap_elastic_mode1.png", dpi=300)
plt.close()

print("Saved: bridge_heatmap_elastic_mode1.png")


#  PLOT FOR BRIDGE INELASTIC VERSION 

event_in, df_in = load_df_over_f("natural_frequencies_bridge_inelastic.csv")
mode1_in = df_in[:, 0]

plt.figure(figsize=(12, 2.5))
plt.imshow(mode1_in[np.newaxis, :],
           aspect="auto",
           cmap="Blues",
           vmin=-0.3, vmax=0.3)
plt.colorbar(label="Δf/f_before")
plt.xticks(range(len(event_in)), event_in, rotation=90)
plt.yticks([0], ["Mode 1"])
plt.title("Bridge Inelastic Model – Mode 1 Δf/f")
plt.tight_layout()
plt.savefig("bridge_heatmap_inelastic_mode1.png", dpi=300)
plt.close()

print("Saved: bridge_heatmap_inelastic_mode1.png")
