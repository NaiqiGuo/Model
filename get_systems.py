from pathlib import Path
import os, glob
from get_249_data import get_249_data
import pickle
import numpy as np
import quakeio
from mdof import sysid
from mdof.utilities.config import Config
from model_utils import( get_inputs, get_node_displacements,
                         create_frame_model, create_bridge_model,
                         apply_load_frame_model, apply_load_bridge_model,
                         analyze,
                         save_displacements, save_strain_stress, apply_gravity_static, apply_load_bridge_model_multi_support
                         )

# Analysis configuration
SID_METHOD = 'srim'
MODEL = "frame" # "frame", "bridge"
ELASTIC = True
LOAD_EVENTS = False

# Main output directory
OUT_DIR = Path(f"{MODEL}")/("elastic" if ELASTIC else "inelastic")
os.makedirs(OUT_DIR, exist_ok=True)



if __name__ == "__main__":

    # Load events
    if MODEL == "frame":
        files_249 = sorted(glob.glob("uploads/CE249_2024_Lab4data/ce249Run*.txt"))
        print(f"Total CE249 txt files loaded: {len(files_249)}")
    else:
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
        input_channels = [1,3] # x,y [1,3,15,17,18,20] [1,3] 

    if MODEL == "frame":
        iterator = enumerate(files_249)
    else:
        iterator = enumerate(events)

    for i, item in iterator:

        if MODEL == "frame":
            # item is file path like .../ce249Run244.txt
            event_id = Path(item).stem.replace("ce249Run", "")  # "244"
        else:
            event_id = str(i+1)

        # Result output directory
        event_dir = OUT_DIR/event_id
        os.makedirs(event_dir, exist_ok=True)

        # Model and system identification inputs (acceleration, in/s²)
        if MODEL == "frame":
            array, sensor_names, sensor_units, time_raw, dt = get_249_data(item)

            # 纯数据第1列 -> x, 第3列 -> y
            inputs = np.vstack([array[0, :], array[2, :]])  # (2, nt)

            print("CE249 file:", item)
            print("x:", sensor_names[0], sensor_units[0])
            print("y:", sensor_names[2], sensor_units[2])

        else:
            inputs, dt = get_inputs(i,
                                    events=events,
                                    input_channels=input_channels,
                                    scale=1/2.54 # cm to inches  1/2.54
                                    )
            
        print("requested input_channels:", input_channels)
        print("loaded inputs.shape:", inputs.shape)
        nt = inputs.shape[1]
        print(f"\nevent {i+1} inputs shape: {inputs.shape}, dt = {dt}")

        # Finite element model
        if MODEL == 'frame':
            output_nodes = [5,10,15]
            model = create_frame_model(elastic=ELASTIC)
            model = apply_load_frame_model(model,
                                    inputx=inputs[0],
                                    inputy=inputs[1],
                                    dt=dt)
            apply_gravity_static(
                model,
                output_nodes=[5,10,15],
                fixed_nodes=[1,2,3,4],
            )
            output_elements = [1,5,9]
            # model-specific y and z fibers for stress-strain measurements
            yFiber = 7.5
            zFiber = 0.0
        elif MODEL == 'bridge':
            output_nodes = [3,5]
            model = create_bridge_model(elastic=ELASTIC)
            apply_gravity_static(
                model,
                output_nodes=[3,5],
                fixed_nodes=[0,1,4,6], #[0,1,4,6,11,12,13,14] [0,1,4,6]
            )
            
            # #for  mutiple excitation
            # node_channel_map = {
            #     0: (15, 17),
            #     6: (1,  3),
            #     4: (1,  3),
            #     1: (18, 20),
            # }
            # model = apply_load_bridge_model_multi_support(
            #     model,
            #     inputs=inputs,
            #     dt=dt,
            #     node_channel_map=node_channel_map,
            #     input_channels=input_channels,
            # )

            #for uniform excitation
            model = apply_load_bridge_model(model,
                                    inputx=inputs[0],
                                    inputy=inputs[1],
                                    dt=dt)
            
            output_elements = [3]
            # model-specific y and z fibers for stress-strain measurements
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
        # n = 3
        # options = Config(
        #     m           = 500,
        #     horizon     = 190,
        #     nc          = 190,
        #     order       = 2*n,
        #     period_band = (0.1,0.6),
        #     damping     = 0.06,
        #     pseudo      = True,
        #     outlook     = 190,
        #     threads     = 8,
        #     chunk       = 200,
        #     i           = 250,
        #     j           = 4400
        # )

        n = 2  
        options = Config(
            m           = 120,       
            horizon     = 25,       
            nc          = 25,
            order       = 3,       
            period_band = (0.15, 0.8),
            damping     = 0.05,
            pseudo      = True,
            outlook     = 25,
            threads     = 4,       
            chunk       = 200,
            i           = 250,
            j           = 3500    
        )
        system_full = sysid(inputs, outputs, method=SID_METHOD, **options)
        A,B,C,D, *rest = system_full
        system = (A,B,C,D)
        with open(event_dir/f"system_{SID_METHOD}.pkl", "wb") as f:
            pickle.dump(system, f)