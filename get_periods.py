import pickle
import numpy as np
import csv
import os
from model_utils import create_model, get_true_modes_xara, get_inputs, get_outputs
from model_utils import stabilize_discrete, periods_from_A, analyze, phi_output

# get true frequency
with open("events.pkl", "rb") as f:
    events = pickle.load(f)
input_channels = [1, 3]
inputs, dt = get_inputs(0, events=events, input_channels=input_channels, scale=2.54)
nt = inputs.shape[1]

model = create_model(column="forceBeamColumn", girder="forceBeamColumn", inputx=inputs[0], inputy=inputs[1], dt=dt)
disp = analyze(model, output_nodes=[9, 14, 19], nt=nt, dt=dt)
outputs = get_outputs(disp)
outputs = outputs[:, 1:]

freqs_true, _ = get_true_modes_xara(model, floor_nodes=(9,14,19), dofs=(1,2), n=3)
periods_true = 1 / freqs_true
print("True structure periods (s):", np.round(periods_true, 4))

num_events = 21
sys_names = ["srim", "n4sid", "det"]

max_num_modes = len(periods_true)
with open("system_periods_all.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["event", "method"] + [f"period{i+1}" for i in range(max_num_modes)]) 
    writer.writerow([0, "true"] + list(periods_true))


for event_id in range(1, num_events+1):
    print(f"\n===== Event {event_id} =====")
    for sys_name in sys_names:
        output_dir = "event_outputs_ABCDk"
        pkl_path = os.path.join(output_dir, f"system_{sys_name}_{event_id:02d}.pkl")
        if not os.path.exists(pkl_path):
            print(f"{pkl_path} pass")
            continue
        with open(pkl_path, "rb") as f:
            A, B, C, D = pickle.load(f)
        # Optional: post-process A (choose only one)
        #A_stable = stabilize_with_lmi(A)                
        #A_stable = stabilize_by_radius_clipping(A)     
        A_stable = stabilize_discrete(A)                
        Phi, eigvals = phi_output(A_stable, C)
        freqs = np.abs(np.angle(eigvals)) / (2 * np.pi * dt)
        sort_idx = np.argsort(freqs)
        Phi = Phi[:, sort_idx]
        eigvals = eigvals[sort_idx]
        periods = 1 / freqs[sort_idx]
        print(f"{sys_name.upper()} system periods count: {len(periods)}")
        print(f"{sys_name.upper()} system periods (s): {np.round(periods, 4)}")

        with open("system_periods_all.csv", "a", newline="") as ff:
            writer = csv.writer(ff)
            row = [event_id, sys_name] + list(periods) #all: list(periods) periods[:3]
            writer.writerow(row)
