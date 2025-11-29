import numpy as np
import matplotlib.pyplot as plt


# ΔT/T 
def load_dT_over_T(csv_path):
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    cols = data.dtype.names

    before_cols = [c for c in cols if "before_s" in c]
    after_cols  = [c for c in cols if "after_s"  in c]
    before_cols.sort()
    after_cols.sort()

    # N_events × M_modes
    f_before = np.column_stack([data[c] for c in before_cols])
    f_after  = np.column_stack([data[c] for c in after_cols])

    # # ΔT/T
    # dT_over_T = (T_after - T_before) / T_before

    # return data["event_id"], dT_over_T

    # Δf/f
    df_over_f = (f_after - f_before) / f_before

    return data["event_id"], df_over_f

# elastic
event_el, dT_el = load_dT_over_T("natural_frequencies_elastic.csv")
mode1_el = dT_el[:, 0]     # modeshape1（0）

plt.figure(figsize=(12, 2.5))
plt.imshow(mode1_el[np.newaxis, :], 
           aspect="auto",
           cmap="Blues_r",
           vmin=-0.3, vmax=0.3)        
plt.colorbar(label="Δf/f_before")
plt.xticks(range(len(event_el)), event_el, rotation=90)
plt.yticks([0], ["Mode 1"])
plt.title("Elastic Model – Mode 1 Natural Freq Change")
plt.tight_layout()
plt.savefig("heatmap_elastic_mode1.png", dpi=300)
plt.close()
print("Saved: heatmap_elastic_mode1.png")

# inelastic
event_in, dT_in = load_dT_over_T("natural_frequencies_inelastic.csv")
mode1_in = dT_in[:, 0]

plt.figure(figsize=(12, 2.5))
plt.imshow(mode1_in[np.newaxis, :], 
           aspect="auto",
           cmap="Blues_r",
           vmin=-0.3, vmax=0.3)
plt.colorbar(label="Δf/f_before")
plt.xticks(range(len(event_in)), event_in, rotation=90)
plt.yticks([0], ["Mode 1"])
plt.title("Inelastic Model – Mode 1 Natural Freq Change")
plt.tight_layout()
plt.savefig("heatmap_inelastic_mode1.png", dpi=300)
plt.close()
print("Saved: heatmap_inelastic_mode1.png")
