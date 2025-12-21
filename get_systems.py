from pathlib import Path
import quakeio
import pickle
import numpy as np
import os
from mdof import sysid
from mdof.utilities.config import Config
from model_utils import( get_inputs, analyze, get_outputs, save_all_methods_to_csv,
                         create_frame_model, create_bridge_model,
                         apply_load_frame_model, apply_load_bridge_model,
                         write_freq_csv,
                         save_event_disp, save_event_strain_stress
                         )

ELASTIC = False
MODEL = "frame"
SID_METHOD = 'srim'

# Load events
LOAD_EVENTS = False
if LOAD_EVENTS:
    events = sorted([
        print(file) or quakeio.read(file, exclusions=["*filter*"])
        # for file in list(Path(f"/Users/guonaiqi/Documents/GitHub/mdof_studies/uploads/CE89324/").glob("????????*.[zZ][iI][pP]"))
        for file in list(Path(f"../uploads/CE89324/").glob("????????*.[zZ][iI][pP]"))
    ], key=lambda event: abs(event["peak_accel"]))
    with open("events.pkl","wb") as f:
        pickle.dump(events,f)
else:
    with open("events.pkl","rb") as f:
        events = pickle.load(f)

print(f"Total events loaded: {len(events)}")

# Choose inputs channels [x,y]
input_channels = [1,3]

output_dir = f"event_outputs_ABCD_{MODEL}_model"
if ELASTIC:
    output_dir += "_elastic"
os.makedirs(output_dir, exist_ok=True)


for i, event in enumerate(events):
    inputs, dt = get_inputs(i, events=events, input_channels=input_channels, scale=2.54)
    print(f"\nevent {i+1} inputs shape: {inputs.shape}, dt = {dt}")

    if MODEL == 'frame':
        model = create_frame_model(elastic=ELASTIC)
        model = apply_load_frame_model(model,
                                inputx=inputs[0],
                                inputy=inputs[1],
                                dt=dt)
    elif MODEL == 'bridge':
        model = create_bridge_model(elastic=ELASTIC)
        model = apply_load_bridge_model(model,
                                inputx=inputs[0],
                                inputy=inputs[1],
                                dt=dt)
    
    nt = inputs.shape[1]
    try:
        disp, stresses, strains, freqs_before, freqs_after = analyze(model,
                                                                nt=nt,
                                                                dt=dt,
                                                                output_nodes=[5, 10, 15],
                                                                output_elements=[1, 5, 9],
                                                                yFiber=9.0,
                                                                zFiber=0.0
                                                            )
    except RuntimeError as e:
        print(f"Error encounted when analyzing event {i}:")
        print(e)
        continue
    outputs = get_outputs(disp)
    outputs = outputs[:,1:] 
    assert inputs.shape[1] == outputs.shape[1], "inputs and outputs have different length of time samples."
    time = np.arange(nt) * dt

    event_id = i+1

    # Save frequencies
    write_freq_csv(event_id,
                   freqs_before,
                   freqs_after,
                   freq_csv_path="natural_frequencies.csv")

    if ELASTIC:
        disp_dir = "event_disp_elastic"
        ss_dir   = "event_strain_stress_elastic"
    else:
        disp_dir = "event_disp_inelastic"
        ss_dir   = "event_strain_stress_inelastic"

    os.makedirs(disp_dir, exist_ok=True)
    os.makedirs(ss_dir, exist_ok=True)

    # Save displacements, strains, and stresses
    save_event_disp(event_id, disp, dt, out_dir=disp_dir)
    save_event_strain_stress(event_id, stresses, strains, dt, out_dir=ss_dir)

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
    

    # System ID
    system = sysid(inputs, outputs, method=SID_METHOD, **options)
    A,B,C,D, *rest = system
    system = A,B,C,D
    with open(os.path.join(output_dir, f"system_{SID_METHOD}_{i+1:02d}.pkl"), "wb") as f:
        pickle.dump(system, f)
    print(f"Saved system_{SID_METHOD}_{i+1:02d}.pkl")

    methods_dict = {
    SID_METHOD: (A,B,C,D),
    }
    
    save_all_methods_to_csv(i, methods_dict)


