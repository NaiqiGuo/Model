import pickle
import numpy as np
from model_utils import create_model, get_true_modes_xara
from model_utils import stabilize_discrete, periods_from_A

# get true frequency
with open("events.pkl", "rb") as f:
    events = pickle.load(f)
input_channels = [1, 3]
inputs, dt = events[0]["accel"][input_channels], events[0]["dt"] 

nt = inputs.shape[1] if inputs.ndim == 2 else len(inputs[0])

model = create_model(
    column="forceBeamColumn",
    girder="forceBeamColumn",
    inputx=np.zeros(nt),
    inputy=np.zeros(nt),
    dt=dt
)
freqs_true, _ = get_true_modes_xara(model, floor_nodes=(9,14,19), dofs=(1,2), n=3)
periods_true = 1 / freqs_true
print("True structure periods (s):", np.round(periods_true, 4))

num_events = 21
sys_names = ["srim", "n4sid", "det", "okid"]

for event_id in range(1, num_events+1):
    print(f"\n===== Event {event_id} =====")
    for sys_name in sys_names:
        with open(f"system_{sys_name}_{event_id:02d}.pkl", "rb") as f:
            A, B, C, D = pickle.load(f)
        # Optional: post-process A (choose only one)
        #A_stable = stabilize_with_lmi(A)                
        #A_stable = stabilize_by_radius_clipping(A)     
        A_stable = stabilize_discrete(A)                

        npz = np.load(f"event_data/event_{event_id}.npz")
        dt = npz["time"][1] - npz["time"][0]

        periods = periods_from_A(A_stable, dt)
        print(f"{sys_name.upper()} system periods (s): {np.round(periods, 4)}")
