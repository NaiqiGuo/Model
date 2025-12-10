from pathlib import Path
import quakeio
import pickle
import csv
import numpy as np
import os
from mdof import sysid
from mdof.validation import stabilize_discrete
from mdof.utilities.config import Config
from model_utils import (
    create_model, get_inputs, analyze, get_outputs, stabilize_with_lmi,
    stabilize_by_radius_clipping, save_all_methods_to_csv,
    save_event_disp, save_event_strain_stress,
    create_painter_bridge_model,     # make sure this is defined in model_utils
)

# Global switch for bridge model
ELASTIC = True   # True = elastic bridge, False = inelastic bridge

# --------------------------------------------------------
# Load events
# --------------------------------------------------------
LOAD_EVENTS = False
if LOAD_EVENTS:
    events = sorted([
        print(file) or quakeio.read(file, exclusions=["*filter*"])
        # for file in list(Path("/Users/guonaiqi/Documents/GitHub/mdof_studies/uploads/CE89324/").glob("????????*.[zZ][iI][pP]"))
        for file in list(Path("../uploads/CE89324/").glob("????????*.[zZ][iI][pP]"))
    ], key=lambda event: abs(event["peak_accel"]))
    with open("events_bridge.pkl", "wb") as f:
        pickle.dump(events, f)
else:
    with open("events.pkl", "rb") as f:
        # if you prefer, you can also create a separate events_bridge.pkl
        events = pickle.load(f)

print(f"Total events loaded: {len(events)}")

START_EVENT_ID = 19
END_EVENT_ID   = 22

# Input channels [x, y]
input_channels = [1, 3]

# --------------------------------------------------------
# Output directories, specific to bridge model
# --------------------------------------------------------
if ELASTIC:
    output_dir = "event_outputs_ABCD_bridge_model_elastic"
else:
    output_dir = "event_outputs_ABCD_bridge_model_inelastic"

os.makedirs(output_dir, exist_ok=True)

# --------------------------------------------------------
# Main loop over events
# --------------------------------------------------------
for i, event in enumerate(events):
    event_id = i + 1   

    if event_id < START_EVENT_ID or event_id > END_EVENT_ID:
        continue

    # inputs has shape (2, nt) for [x, y] components
    inputs, dt = get_inputs(
        i,
        events=events,
        input_channels=input_channels,
        scale=2.54
    )
    print(f"\nBridge model, event {i+1} inputs shape: {inputs.shape}, dt = {dt}")

    # ------------------------------------------------
    # Build Painter Street bridge model
    # ------------------------------------------------
    model = create_painter_bridge_model(
        elastic=ELASTIC,
        inputx=inputs[0],
        inputy=inputs[1],
        dt=dt
    )

    nt = inputs.shape[1]

    # Choose output nodes on the deck.
    # Here we use nodes 2, 3, 5.
    # You can adjust this list to match the sensor layout if needed.
    output_nodes = [2, 3, 5]

    
    # Time history analysis
    try:
        disp, stresses, strains = analyze(
            model,
            output_nodes=output_nodes,
            nt=nt,
            dt=dt,
            output_elements=[2,3],
            yFiber=8.0,
            zFiber=0.0
        )
    except RuntimeError as e:
        print(f"Error for event {i+1}:")
        print(e)
        continue

    outputs = get_outputs(disp)
    outputs = outputs[:, 1:]   # drop time row
    assert inputs.shape[1] == outputs.shape[1], \
        "inputs and outputs have different length of time samples."
    time = np.arange(nt) * dt

    # Save displacement and strain/stress for this event
    event_id = i + 1

    if ELASTIC:
        disp_dir = "event_disp_bridge_elastic"
        ss_dir   = "event_strain_stress_bridge_elastic"
    else:
        disp_dir = "event_disp_bridge_inelastic"
        ss_dir   = "event_strain_stress_bridge_inelastic"

    os.makedirs(disp_dir, exist_ok=True)
    os.makedirs(ss_dir, exist_ok=True)

    save_event_disp(event_id, disp, dt, out_dir=disp_dir)
    save_event_strain_stress(event_id, stresses, strains, dt, out_dir=ss_dir)


    # System identification options
    # n = 6
    # options = Config(
    #     m           = 500,
    #     horizon     = 190,
    #     nc          = 190,
    #     order       = 2*n,
    #     period_band = (0.1, 0.6),
    #     damping     = 0.06,
    #     pseudo      = True,
    #     outlook     = 190,
    #     threads     = 8,
    #     chunk       = 200,
    #     i           = 250,
    #     j           = 4400
    # )

    n = 4  
    options = Config(
        m           = 300,       
        horizon     = 100,       
        nc          = 100,
        order       = 2*n,       
        period_band = (0.1, 1.0),
        damping     = 0.05,
        pseudo      = True,
        outlook     = 100,
        threads     = 4,       
        chunk       = 200,
        i           = 250,
        j           = 3500    
    )

    # ---- SRIM ----
    system_srim = sysid(inputs, outputs, method='srim', **options)
    A_s, B_s, C_s, D_s, *rest = system_srim
    system_srim = (A_s, B_s, C_s, D_s)

    with open(os.path.join(output_dir, f"system_srim_bridge_{event_id:02d}.pkl"), "wb") as f:
        pickle.dump(system_srim, f)
    print(f"Saved system_srim_bridge_{event_id:02d}.pkl")

    methods_dict = {
        'srim': (A_s, B_s, C_s, D_s),
        # you can enable other methods here if needed
    }
    # save methods to CSV, with event index i and method dict
    save_all_methods_to_csv(event_id - 1, methods_dict)
