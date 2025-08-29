import pickle
import numpy as np
from model_utils import (
    create_model, get_inputs, get_true_modes_xara, analyze, get_outputs,
    stabilize_with_lmi, stabilize_by_radius_clipping, normalize_Psi,
    condense_eigvecs, phi_output, mac_matrix, stabilize_discrete
)

# Load event data
with open("events.pkl", "rb") as f:
    events = pickle.load(f)
print(f"Loaded {len(events)} events.")

# Select a representative event (e.g., the first event)
input_channels = [1, 3]
inputs, dt = get_inputs(0, events=events, input_channels=input_channels, scale=2.54)
nt = inputs.shape[1]

# Construct the true (reference) structural model
model = create_model(
    column="forceBeamColumn",
    girder="forceBeamColumn",
    inputx=np.zeros(nt),
    inputy=np.zeros(nt),
    dt=dt
)

# Extract true mode shapes and natural frequencies
true_freqs, Phi_true = get_true_modes_xara(model, floor_nodes=(9,14,19), dofs=(1,2), n=3)
print("True frequencies (Hz):", np.round(true_freqs, 3))
for i, phi in enumerate(Phi_true.T):
    print(f"True mode {i+1} shape: {np.round(phi, 4)}")

# Save true modal data
with open("modes_model.pkl", "wb") as f:
    pickle.dump((true_freqs, Phi_true), f)
print("Saved modes_model.pkl")

# Normalize true mode shapes for MAC computation
Phi_true_norm = Phi_true / (np.linalg.norm(Phi_true, axis=0, keepdims=True) + 1e-12)

num_events = 21
algos = ["srim", "n4sid", "det", "okid"]

# Loop through all events and extract mode shapes for each identification algorithm
for event_id in range(1, num_events+1):
    print(f"\n===== Event {event_id} =====")
    for algo in algos:
        with open(f"system_{algo}_{event_id:02d}.pkl", "rb") as f:
            A, B, C, D = pickle.load(f)
        A_stable = stabilize_discrete(A)
        Phi, eigvals = phi_output(A_stable, C)
        mac = mac_matrix(Phi_true_norm, Phi)
        print(f"\n{algo.upper()} MAC:\n", np.round(mac, 3))
        for mode_idx in range(Phi.shape[1]):
            print(f"Event {event_id} Mode {mode_idx+1}:")
            print("  True   :", np.round(Phi_true[:, mode_idx], 4))
            print(f"  {algo.upper()}:", np.round(Phi[:, mode_idx], 4))
        
        with open(f"modes_{algo}_{event_id:02d}.pkl", "wb") as ff:
            pickle.dump((eigvals, Phi), ff)
