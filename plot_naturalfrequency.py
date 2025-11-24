import numpy as np
import matplotlib.pyplot as plt

csv_path = "natural_frequencies.csv"

# 读 CSV
data = np.genfromtxt(csv_path, delimiter=",", names=True)

event_ids = data["event_id"]

# 自动识别列
cols = data.dtype.names
before_cols = [c for c in cols if "_before_Hz" in c]
after_cols  = [c for c in cols if "_after_Hz" in c]
n_modes = len(before_cols)

# 画一张图
plt.figure(figsize=(10,6))

for k in range(n_modes):
    f_before = data[before_cols[k]]
    f_after  = data[after_cols[k]]

    # Before
    plt.plot(event_ids, f_before, "o-",
             label=f"Mode {k+1} before")

    # After
    plt.plot(event_ids, f_after, "s--",
             label=f"Mode {k+1} after")

plt.xlabel("Event ID")
plt.ylabel("Natural Frequency [Hz]")
plt.title("Natural Frequencies Before/After Each Earthquake")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("natural_frequencies_plot.png", dpi=300)

print("Saved: natural_frequencies_plot.png")
