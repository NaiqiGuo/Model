from pathlib import Path
import os
import pickle
import numpy as np
import quakeio
from mdof import sysid
from mdof.utilities.config import Config
from model_utils import( get_inputs, get_node_displacements,
                         create_frame_model, create_bridge_model,
                         apply_load_frame_model, apply_load_bridge_model,
                         analyze,
                         save_displacements, save_strain_stress
                         )

# Analysis configuration
SID_METHOD = 'srim'
MODEL = "bridge" # "frame", "bridge"
ELASTIC = True
LOAD_EVENTS = False

# Main output directory
OUT_DIR = Path(f"{MODEL}")/("elastic" if ELASTIC else "inelastic")
os.makedirs(OUT_DIR, exist_ok=True)



if __name__ == "__main__":

    # Load events
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

        event_id = str(i+1)

        # Result output directory
        event_dir = OUT_DIR/event_id
        os.makedirs(event_dir, exist_ok=True)

        # Model and system identification inputs (acceleration, in/s²)
        inputs, dt = get_inputs(i,
                                events=events,
                                input_channels=input_channels,
                                scale=1/2.54 # cm to inches
                                )
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
            # # TODO: model-specific y and z fibers for stress-strain measurements
            yFiber = 7.5
            zFiber = 0.0
        elif MODEL == 'bridge':
            model = create_bridge_model(elastic=ELASTIC)
            model = apply_load_bridge_model(model,
                                    inputx=inputs[0],
                                    inputy=inputs[1],
                                    dt=dt)
            output_nodes = [2,3,5]
            output_elements = [2,3]
            # # TODO: model-specific y and z fibers for stress-strain measurements
            yFiber = 22.5
            zFiber = 0.0
        try:
            disp, stresses, strains, freqs_before, freqs_after = analyze(model,
                                                                    nt=nt,
                                                                    dt=dt,
                                                                    output_nodes=output_nodes,
                                                                    output_elements=output_elements,
                                                                    yFiber=yFiber,
                                                                    zFiber=zFiber
                                                                )
        except RuntimeError as e:
            print(f"Error encounted when analyzing event {i}:")
            print(e)
            continue

        # Save frequencies, displacements, strains, and stresses
        np.savetxt(event_dir/"pre_eq_natural_frequencies.csv", freqs_before)
        np.savetxt(event_dir/"post_eq_natural_frequencies.csv", freqs_after)
        save_displacements(disp, dt, filename=event_dir/"displacements.csv")
        save_strain_stress(stresses, strains, dt, filename=event_dir/"strain_stress.csv")


        # System identification outputs (displacement, inches)
        outputs = get_node_displacements(disp, nodes=output_nodes, dt=dt)[:,1:]
        print(f"{outputs.shape=}")
        assert inputs.shape[1] == outputs.shape[1], (
            "system identification inputs and outputs have different length of time samples.")
        time = np.arange(nt) * dt

        # Save dt, time, and system identification inputs and outputs
        with open(event_dir/"dt.txt", "w") as f:
            f.write(str(dt))
        np.savetxt(event_dir/"time.csv", time)
        np.savetxt(event_dir/"inputs.csv", inputs)
        np.savetxt(event_dir/"outputs.csv", outputs)

        # Perform system identification and save systems
        n = 3
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
        with open(event_dir/f"system_{SID_METHOD}.pkl", "wb") as f:
            pickle.dump(system, f)