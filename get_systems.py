from pathlib import Path
import quakeio
import pickle
import csv
import numpy as np
import os
from mdof import sysid
from mdof.validation import stabilize_discrete
from mdof.utilities.config import Config
from model_utils import( create_model, get_inputs, analyze, get_outputs, stabilize_with_lmi,
                         stabilize_by_radius_clipping, save_all_methods_to_csv, create_frame_model)


# Load events
LOAD_EVENTS = False
if LOAD_EVENTS:
    events = sorted([
        print(file) or quakeio.read(file, exclusions=["*filter*"])
        for file in list(Path(f"/Users/guonaiqi/Documents/GitHub/mdof_studies/uploads/CE89324/").glob("????????*.[zZ][iI][pP]"))
    ], key=lambda event: abs(event["peak_accel"]))
    with open("events.pkl","wb") as f:
        pickle.dump(events,f)
else:
    with open("events.pkl","rb") as f:
        events = pickle.load(f)

print(f"Total events loaded: {len(events)}")

# Choose inputs channels [x,y]
input_channels = [1,3]

output_dir = "event_outputs_ABCD_frame_model"
os.makedirs(output_dir, exist_ok=True)

# selected_indices = [19, 20, 21]  # 20、21、22（from 0）
# for i in selected_indices:
#     event = events[i]

for i, event in enumerate(events):
    inputs, dt = get_inputs(i, events=events, input_channels=input_channels, scale=2.54)
    print(f"\nevent {i+1} inputs shape: {inputs.shape}, dt = {dt}")
    # model = create_model(column="forceBeamColumn",
    #                             girder="forceBeamColumn",
    #                             inputx=inputs[0],
    #                             inputy=inputs[1],
    #                             dt=dt)
    
    model = create_frame_model(column="elasticBeamColumn",
                                girder="elasticBeamColumn",
                                inputx=inputs[0],
                                inputy=inputs[1],
                                dt=dt)
    
    nt = inputs.shape[1]
    try:
        disp = analyze(model, output_nodes = [5, 10, 15], nt=nt, dt=dt)
    except RuntimeError:
        continue
    outputs = get_outputs(disp)
    outputs = outputs[:, 1:] 
    assert inputs.shape[1] == outputs.shape[1], "inputs and outputs have different length of time samples."
    time = np.arange(nt) * dt

    n = 6
    options = Config(
        m           = 500,
        horizon     = 190,
        nc          = 190,
        order       = 2*n,
        period_band = (0.1,0.6),
        damping     = 0.06,
        pseudo      = True,
        outlook     = 190,
        threads     = 8,
        chunk       = 200,
        i           = 250,
        j           = 4400
    )
    # i           = 250,
    # j           = 4400

    # ---- SRIM ----
    system_srim = sysid(inputs, outputs, method='srim', **options)
    A_s, B_s, C_s, D_s, *rest = system_srim
    system_srim = (A_s, B_s, C_s, D_s)
    with open(os.path.join(output_dir, f"system_srim_{i+1:02d}.pkl"), "wb") as f:
        pickle.dump(system_srim, f)
    print(f"Saved system_srim_{i+1:02d}.pkl")

    # ---- N4SID ----
    # system_n4sid = sysid(inputs, outputs, method='n4sid', **options)
    # A_n, B_n, C_n, D_n, *rest = system_n4sid
    # system_n4sid = (A_n, B_n, C_n, D_n)
    # with open(os.path.join(output_dir, f"system_n4sid_{i+1:02d}.pkl"), "wb") as f:
    #     pickle.dump(system_n4sid, f)
    # print(f"Saved system_n4sid_{i+1:02d}.pkl")

    # ---- DETERMINISTIC ----
    # system_det = sysid(inputs, outputs, method='deterministic', **options)
    # A_d, B_d, C_d, D_d, *rest = system_det
    # system_det = (A_d, B_d, C_d, D_d)
    # with open(os.path.join(output_dir, f"system_det_{i+1:02d}.pkl"), "wb") as f:
    #     pickle.dump(system_det, f)
    # print(f"Saved system_det_{i+1:02d}.pkl")

    # ---- OKID ----
    # system_okid = sysid(inputs, outputs, method='okid-era', **options)
    # A_o, B_o, C_o, D_o, *rest = system_okid
    # system_okid = (A_o, B_o, C_o, D_o)
    # with open(f"system_okid_{i+1:02d}.pkl", "wb") as f:
    #     pickle.dump(system_okid, f)
    # print(f"Saved system_okid_{i+1:02d}.pkl")

    methods_dict = {
    'srim': (A_s, B_s, C_s, D_s),
    #'n4sid': (A_n, B_n, C_n, D_n),
    #'det': (A_d, B_d, C_d, D_d),
    #'okid': (A_o, B_o, C_o, D_o)
    }
    save_all_methods_to_csv(i, methods_dict)


