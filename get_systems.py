from pathlib import Path
import os, glob
from get_249_data import get_249_data
import pickle
import numpy as np
import quakeio
from mdof import sysid
from mdof.utilities.config import Config
from utilities import (
    get_inputs,
    get_node_displacements,
    create_frame_model,
    create_bridge_model,
    apply_load_frame_model,
    analyze,
    )
from utilities_experimental import(
    apply_load_bridge_model, # TODO CC: first pass clean
    apply_load_bridge_model_multi_support, # TODO CC+NG: after clean apply_load_bridge_model, merge
    apply_gravity_static, # TODO CC+NG: clean this; clarify wipe analysis commands
    save_displacements, # TODO CC: verify and move to utilities
    save_strain_stress, # TODO CC: verify and move to utilities
    )

# Analysis configuration
SID_METHOD = 'srim'
MODEL = "bridge" # "frame", "bridge"
MULTISUPPORT = False
ELASTIC = True
LOAD_EVENTS = False
VERBOSE = 1 # False means print nothing;
            # True or 1 means print some helper messages;
            # 2 means print many messages

# Main output directory
OUT_DIR = Path(f"{MODEL}")/("elastic" if ELASTIC else "inelastic")
os.makedirs(OUT_DIR, exist_ok=True)



if __name__ == "__main__":
    # Print analysis configuration
    if VERBOSE:
        print(f"{MODEL=}")
        print(f"{ELASTIC=}")

    # Load events
    if MODEL == "frame":
        # events are a list of filepaths to txt
        events = sorted(glob.glob("uploads/CE249_2024_Lab4data/ce249Run*.txt"))
    if MODEL == "bridge":
        # events are a list of quakeio objects
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
    if VERBOSE:
        print(f"Total events loaded: {len(events)}")


    # Perform model analysis and system identification and record responses

    if MODEL == "frame":
        if not MULTISUPPORT:
            # Rows in data array parsed from txt file
            input_channels = [0,2] # x, y
    elif MODEL == "bridge":
        if not MULTISUPPORT:
            # Labeled channel numbers from quakeio object
            input_channels = [1,3] # x, y
        else:
            # Labeled channel numbers from quakeio object
            input_channels = [1,3,15,17,18,20] # ordered arbitrarily
            # Nodes for excitation, order corresponds to input_channels
            input_nodes = [] # TODO: fill in input nodes
            # DOFs for excitation, order corresponds to input_channels
            input_dofs = []  # TODO: fill in input dofs

    for i,event in enumerate(events):
        if MODEL == "frame":
            # filepaths are like .../ce249Run244.txt
            event_id = Path(event).stem.replace("ce249Run", "")  # "244"
        elif MODEL == "bridge":
            event_id = str(i+1)
        if VERBOSE:
            print(f"Event: {event}; Event ID: {event_id}")

        # Result output directory
        event_dir = OUT_DIR/event_id
        os.makedirs(event_dir, exist_ok=True)

        # Model and system identification inputs (acceleration, in/s²)
        if MODEL == "frame":
            array, sensor_names, sensor_units, time_raw, dt = get_249_data(event)

            inputs = array[input_channels]

            if VERBOSE >= 2:
                print(
                    f"frame sensor x: \n"
                        f"\tName = {sensor_names[input_channels[0]]}\n"
                        f"\tUnits = {sensor_units[input_channels[0]]}"
                    f"frame sensor y: \n"
                        f"\tName = {sensor_names[input_channels[1]]}\n"
                        f"\tUnits = {sensor_units[input_channels[1]]}"
                    )
                
        elif MODEL == "bridge":
            inputs, dt = get_inputs(i,
                                    events=events,
                                    input_channels=input_channels,
                                    scale=1/2.54 # cm to inches  1/2.54
                                    )
        
        # For uniform excitation, inputs shape should be (2, nt)
        # For multi-support excitation, inputs shape should be (len(input_channels), nt)
        nin,nt = inputs
        assert nin==len(input_channels) if MULTISUPPORT else nin==2
        if VERBOSE >= 2:
            print("Requested input channels:", input_channels)
            print(f"Event {event_id} time series length: {nt}, Time step dt = {dt}")


        # Finite element model
        if MODEL == 'frame':
            output_nodes = [5,10,15]
            output_elements = [1,5,9]
            yFiber = 7.5
            zFiber = 0.0

            model = create_frame_model(elastic=ELASTIC,
                                       multisupport=MULTISUPPORT,
                                       verbose=VERBOSE)

            model = apply_load_frame_model(model,
                                    inputx=inputs[0],
                                    inputy=inputs[1],
                                    dt=dt)

        elif MODEL == 'bridge':
            output_nodes = [3,5]
            model = create_bridge_model(elastic=ELASTIC,
                                        multisupport=MULTISUPPORT,
                                        separate_deck_ends=True,
                                        verbose=VERBOSE
                                        )
            
            if MULTISUPPORT:
                node_channel_map = {
                    0: (15, 17),
                    6: (1,  3),
                    4: (1,  3),
                    1: (18, 20),
                }
                model = apply_load_bridge_model_multi_support(
                    model,
                    inputs=inputs,
                    dt=dt,
                    node_channel_map=node_channel_map,
                    input_channels=input_channels,
                )
            else:
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
            print(f"Error encounted when analyzing event {event_id}:")
            print(e)
            continue


        if True:
            # Save frequencies, displacements, strains, and stresses
            np.savetxt(event_dir/"pre_eq_natural_frequencies.csv", freqs_before)
            np.savetxt(event_dir/"post_eq_natural_frequencies.csv", freqs_after)
            save_displacements(disp, dt, filename=event_dir/"displacements.csv")
            save_strain_stress(stresses, strains, dt, filename=event_dir/"strain_stress.csv")


            # System identification outputs (displacement, inches)
            outputs = get_node_displacements(disp, nodes=output_nodes, dt=dt)[:,1:]

            print("post-gravity nodeDisp(3) =", model.nodeDisp(3))
            print("post-gravity nodeDisp(5) =", model.nodeDisp(5))


            # peak + simple corr checks (assume outputs are [X1,Y1,X2,Y2,...])
            a0 = inputs[0] - np.mean(inputs[0])
            a1 = inputs[1] - np.mean(inputs[1])
            # print every output channel stats (no labels, just index)
            for ch in range(outputs.shape[0]):
                y = outputs[ch]
                imax = int(np.argmax(np.abs(y)))
                print(f"out[{ch:02d}]  std={np.std(y):.3e}  peakAbs={y[imax]:+.3e} @ t={imax*dt:.3f}s")
            # split X/Y pairs if possible
            if outputs.shape[0] % 2 == 0:
                x_outputs = outputs[0::2]
                y_outputs = outputs[1::2]
                mx = np.mean(x_outputs, axis=0) - np.mean(np.mean(x_outputs, axis=0))
                my = np.mean(y_outputs, axis=0) - np.mean(np.mean(y_outputs, axis=0))
                denom = (np.linalg.norm(a0) * np.linalg.norm(mx))
                c_x_mx = (a0 @ mx) / denom if denom > 0 else np.nan
                denom = (np.linalg.norm(a0) * np.linalg.norm(my))
                c_x_my = (a0 @ my) / denom if denom > 0 else np.nan
                denom = (np.linalg.norm(a1) * np.linalg.norm(mx))
                c_y_mx = (a1 @ mx) / denom if denom > 0 else np.nan
                denom = (np.linalg.norm(a1) * np.linalg.norm(my))
                c_y_my = (a1 @ my) / denom if denom > 0 else np.nan

                print("corr(inputX, mean X outputs) =", c_x_mx)
                print("corr(inputX, mean Y outputs) =", c_x_my)
                print("corr(inputY, mean X outputs) =", c_y_mx)
                print("corr(inputY, mean Y outputs) =", c_y_my)
            else:
                print(f"WARNING: outputs has odd channels ({outputs.shape[0]}), cannot split X/Y pairs cleanly.")



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

            n = 3
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