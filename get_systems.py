from pathlib import Path
import os
import pickle
import numpy as np
import quakeio
from mdof import sysid
from mdof.utilities.config import Config
from model_utils import( get_inputs, get_outputs,
                         create_frame_model, create_bridge_model,
                         apply_load_frame_model, apply_load_bridge_model,
                         analyze,
                         save_displacements, save_strain_stress
                         )


# Analysis configuration
SID_METHOD = 'srim'
MODEL = "frame" # "frame", "bridge"
ELASTIC = False


# Output directories
OUT_DIR = Path(f"{MODEL}")/{"elastic" if ELASTIC else "inelastic"}
os.makedirs(OUT_DIR, exist_ok=True)

SYSTEMS_DIR = OUT_DIR/f"systems_{SID_METHOD}"
os.makedirs(SYSTEMS_DIR, exist_ok=True)

NF_DIR   = OUT_DIR/f"natural_frequencies"
os.makedirs(NF_DIR, exist_ok=True)

DISP_DIR = OUT_DIR/f"displacements"
os.makedirs(DISP_DIR, exist_ok=True)

SS_DIR   = OUT_DIR/f"strain_stress"
os.makedirs(SS_DIR, exist_ok=True)


# Load events
LOAD_EVENTS = False
if LOAD_EVENTS:
    events = sorted([
        print(file) or quakeio.read(file, exclusions=["*filter*"])
        for file in list(Path(f"../uploads/CE89324/").glob("????????*.[zZ][iI][pP]"))
    ], key=lambda event: abs(event["peak_accel"]))
    with open("events.pkl","wb") as f:
        pickle.dump(events,f)
else:
    with open("events.pkl","rb") as f:
        events = pickle.load(f)
print(f"Total events loaded: {len(events)}")


# Perform model analysis and system identification and record responses
if MODEL == "frame" or MODEL == "bridge":
    input_channels = [1,3] # x,y

for i, event in enumerate(events):

    event_id = i+1

    # System identification and model inputs
    inputs, dt = get_inputs(i, events=events, input_channels=input_channels, scale=2.54)
    nt = inputs.shape[1]
    print(f"\nevent {i+1} inputs shape: {inputs.shape}, dt = {dt}")

    # Finite element model
    if MODEL == 'frame':
        model = create_frame_model(elastic=ELASTIC)
        model = apply_load_frame_model(model,
                                inputx=inputs[0],
                                inputy=inputs[1],
                                dt=dt)
        output_nodes = [5,10,15]
        output_elements = [1,5,9]
    elif MODEL == 'bridge':
        model = create_bridge_model(elastic=ELASTIC)
        model = apply_load_bridge_model(model,
                                inputx=inputs[0],
                                inputy=inputs[1],
                                dt=dt)
        output_nodes = [2,3,5]
        output_elements = [2,3]
    try:
        disp, stresses, strains, freqs_before, freqs_after = analyze(model,
                                                                nt=nt,
                                                                dt=dt,
                                                                output_nodes=output_nodes,
                                                                output_elements=output_elements,
                                                                yFiber=9.0,
                                                                zFiber=0.0
                                                            )
    except RuntimeError as e:
        print(f"Error encounted when analyzing event {i}:")
        print(e)
        continue

    # Save frequencies, displacements, strains, and stresses
    np.savetxt(NF_DIR/f"pre_eq_{event_id:02d}.csv", freqs_before)
    np.savetxt(NF_DIR/f"post_eq_{event_id:02d}.csv", freqs_after)
    save_displacements(disp, dt, filename=DISP_DIR/f"{event_id:02d}.csv")
    save_strain_stress(stresses, strains, dt, filename=SS_DIR/f"{event_id:02d}.csv")


    # System identification outputs
    outputs = get_outputs(disp)
    outputs = outputs[:,1:] 
    assert inputs.shape[1] == outputs.shape[1], "inputs and outputs have different length of time samples."
    time = np.arange(nt) * dt

    # Save system identification inputs and outputs
    np.savetxt(SYSTEMS_DIR/f"inputs_{event_id:02d}.csv", inputs)
    np.savetxt(SYSTEMS_DIR/f"outputs_{event_id:02d}.csv", outputs)

    # Perform system identification and save systems
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
    system_full = sysid(inputs, outputs, method=SID_METHOD, **options)
    A,B,C,D, *rest = system_full
    system = (A,B,C,D)
    system_file = f"system_{event_id:02d}.pkl"
    with open(SYSTEMS_DIR/system_file, "wb") as f:
        pickle.dump(system, f)
    print(f"Saved {str(system_file)}")