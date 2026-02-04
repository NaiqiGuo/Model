import pickle
import numpy as np
from utilities_experimental import (
    create_model, get_inputs, get_true_modes_xara, analyze, get_node_displacements,
    stabilize_with_lmi, stabilize_by_radius_clipping, normalize_Psi,
     phi_output, mac_matrix, stabilize_discrete
)
import csv
import os

output_dir = "event_modes_outputs"
ABCD_dir = "event_outputs_ABCD_frame_model_elastic" 
os.makedirs(output_dir, exist_ok=True)

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
    column="elasticBeamColumn",
    girder="elasticBeamColumn",
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
with open(os.path.join(output_dir, "modes_model.pkl"), "wb") as f:
    pickle.dump((true_freqs, Phi_true), f)
print("Saved modes_model.pkl")

# Normalize true mode shapes for MAC computation
Phi_true_norm = Phi_true / (np.linalg.norm(Phi_true, axis=0, keepdims=True) + 1e-12)

num_events = 21
algos = ["srim", "n4sid", "det"] #, "okid"

# Loop through all events and extract mode shapes for each identification algorithm
for event_id in range(1, num_events+1):
    print(f"\n===== Event {event_id} =====")
    method_modes = {}
    method_macs = {}
    method_freqs = {}
    for algo in algos:
        output_dir = "event_modes_outputs"
        pkl_path = os.path.join(ABCD_dir, f"system_{algo}_{event_id:02d}.pkl")
        if not os.path.exists(pkl_path):
            print(f"{pkl_path} pass")
            continue
        with open(pkl_path, "rb") as f:
            A, B, C, D = pickle.load(f)
        A_stable = stabilize_discrete(A)
        # Optional: post-process A (choose only one)
        #A_stable = stabilize_with_lmi(A)                
        #A_stable = stabilize_by_radius_clipping(A)
        eigvals_all, _ = np.linalg.eig(np.asarray(A_stable, dtype=complex))
        print(f"\nAll eigenvalues of {algo.upper()} for event {event_id}:")
        print(eigvals_all) #print(np.round(eigvals_all, 6))
        Phi, eigvals = phi_output(A_stable, C)
        freqs = np.abs(np.angle(eigvals)) / (2 * np.pi * dt)
        sort_idx = np.argsort(freqs)
        freqs = freqs[sort_idx]   
        eigvals = eigvals[sort_idx]
        Phi = Phi[:, sort_idx]
        mac = mac_matrix(Phi_true_norm, Phi)
        method_modes[algo] = Phi
        method_macs[algo] = mac
        method_freqs[algo] = freqs
        print(f"\n{algo.upper()} MAC:\n", np.round(mac, 3))
        for mode_idx in range(Phi_true.shape[1]):
            print(f"Event {event_id} Mode {mode_idx+1}:")
            print("  True   :", np.round(Phi_true[:, mode_idx], 4))
            print(f"  {algo.upper()}:", np.round(Phi[:, mode_idx], 4))
        
        with open(os.path.join(output_dir, f"modes_{algo}_{event_id:02d}.pkl"), "wb") as ff:
            pickle.dump((eigvals, Phi), ff)

    filename = os.path.join(output_dir, f"event_modes_{event_id:02d}.csv")
    with open(filename, "w", newline='') as f:
        writer = csv.writer(f)
        #True freqs
        writer.writerow(["True Natural Frequencies"])
        writer.writerow([""] + list(true_freqs))
        # True mode shapes
        writer.writerow([f"Event {event_id} True Mode Shapes"])
        for i in range(Phi_true.shape[1]):
            writer.writerow([f"True Mode {i+1}"] + list(Phi_true[:, i]))
        # mode shapes
        for algo in algos:
            if algo not in method_modes:
                print(f"{algo} no modeshape,pass event {event_id}")
                continue
            writer.writerow([f"{algo.upper()} Mode Shapes"])
            Phi = method_modes[algo]
            for i in range(Phi.shape[1]):
                writer.writerow([f"{algo.upper()} Mode {i+1}"] + list(Phi[:, i]))
            freqs = method_freqs[algo]
            writer.writerow([f"{algo.upper()} Natural Frequencies"])
            writer.writerow([""] + list(freqs))

        # MAC
        for algo in algos:
            if algo not in method_modes:
                print(f"{algo} no modeshape, pass event {event_id}")
                continue
            writer.writerow([f"{algo.upper()} MAC vs True"])
            MAC = method_macs[algo]
            for row in MAC:
                writer.writerow([""] + list(row))
        writer.writerow([])  
    print(f"Saved {filename}")

        
