from pathlib import Path
import os, glob
from get_249_data import get_249_data, scale_249_units
import numpy as np
import quakeio
from mdof import sysid
from mdof.utilities.config import Config
import xara.units.iks as units
from utilities import (
    get_inputs,
    get_node_displacements,
    create_frame,
    apply_load_frame,
    analyze,
    )
from utilities_experimental import(
    create_bridge, # TODO CC: update this
    apply_load_bridge, # TODO CC: first pass clean
    apply_load_bridge_multi_support, # TODO CC+NG: after clean apply_load_bridge, absorb
    apply_gravity_static, # TODO CC: verify and move to utilities
    save_displacements, # TODO CC: verify and move to utilities
    save_strain_stress, # TODO CC: verify and move to utilities
    )

# Analysis configuration
SID_METHOD = 'srim'
MODEL = "frame" # "frame", "bridge"
MULTISUPPORT = False
ELASTIC = True
LOAD_EVENTS = False

# Verbosity
# False means print nothing;
# True or 1 means print progress messages only;
# 2 means print progress and validation messages
VERBOSE = 2

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
            input_nodes = [] # TODO NG: fill in input nodes.
            # DOFs for excitation, order corresponds to input_channels
            input_dofs = []  # TODO NG: fill in input dofs

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

            input_units = sensor_units[input_channels[0]]
            assert all(sensor_units[i] == input_units for i in input_channels)
            inputs = array[input_channels]*scale_249_units(units=input_units)

            if VERBOSE >= 2:
                print(
                    f"frame sensor x: \n"
                        f"\tName = {sensor_names[input_channels[0]]}\n"
                        f"\tUnits = {sensor_units[input_channels[0]]}\n"
                    f"frame sensor y: \n"
                        f"\tName = {sensor_names[input_channels[1]]}\n"
                        f"\tUnits = {sensor_units[input_channels[1]]}"
                    )
                
        elif MODEL == "bridge":
            input_units = units.cmps2
            scale = 1/2.54
            inputs, dt = get_inputs(i,
                                    events=events,
                                    input_channels=input_channels,
                                    scale=scale
                                    )
        
        # For uniform excitation, inputs shape should be (2, nt)
        # For multi-support excitation, inputs shape should be (len(input_channels), nt)
        nin,nt = inputs.shape
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

            model = create_frame(elastic=ELASTIC,
                                       multisupport=MULTISUPPORT,
                                       verbose=VERBOSE)

            model = apply_load_frame(model,
                                    inputx=inputs[0],
                                    inputy=inputs[1],
                                    dt=dt)
            
            # TODO CC: verify
            model = apply_gravity_static(
                model,
                output_nodes=output_nodes,
                fixed_nodes=[1,2,3,4],  
                tol=1e-8,
                max_iter=50,
                n_steps=10,
            )

        elif MODEL == 'bridge':
            output_nodes = [3,5]
            output_elements = [3]
            yFiber = 22.5
            zFiber = 0.0

            model = create_bridge(elastic=ELASTIC,
                                        multisupport=MULTISUPPORT,
                                        separate_deck_ends=True,
                                        verbose=VERBOSE
                                        )
            
            if not MULTISUPPORT:
                model = apply_load_bridge(model,
                                        inputx=inputs[0],
                                        inputy=inputs[1],
                                        dt=dt,
                                        # multisupport=MULTISUPPORT,
                                        # input_nodes=input_nodes,
                                        # input_channels=input_channels
                                        )
                
            elif False:
                # TODO NG: After clean apply_load_bridge,
                # absorb into apply_load_bridge.
                # Supersede with input_nodes and input_dofs
                node_channel_map = { 
                    0: (15, 17),
                    6: (1,  3),
                    4: (1,  3),
                    1: (18, 20),
                }
                model = apply_load_bridge_multi_support(
                    model,
                    inputs=inputs,
                    dt=dt,
                    node_channel_map=node_channel_map,
                    input_channels=input_channels,
                )

        try:
            disp, stresses, strains, freqs_before, freqs_after = analyze(model,
                                                                    nt=nt,
                                                                    dt=dt,
                                                                    output_nodes=output_nodes,
                                                                    output_elements=output_elements,
                                                                    yFiber=yFiber,
                                                                    zFiber=zFiber,
                                                                    verbose=VERBOSE
                                                                )

        except RuntimeError as e:
            print(f"Error encounted when analyzing event {event_id}:")
            print(e)
            continue


        # Save frequencies, displacements, strains, and stresses
        np.savetxt(event_dir/"pre_eq_natural_frequencies.csv", freqs_before)
        np.savetxt(event_dir/"post_eq_natural_frequencies.csv", freqs_after)
        save_displacements(disp, dt, filename=event_dir/"displacements.csv")
        save_strain_stress(stresses, strains, dt, filename=event_dir/"strain_stress.csv")

        # System identification outputs (displacement, inches)
        outputs = get_node_displacements(disp, nodes=output_nodes, dt=dt)[:,1:]


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