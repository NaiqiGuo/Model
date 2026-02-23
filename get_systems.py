from pathlib import Path
import os
import glob
import pickle
from get_249_data import get_249_data, scale_249_units
import numpy as np
import quakeio
from mdof import sysid
from mdof.utilities.config import Config
import xara.units.iks as units
import pickle
from utilities import (
    get_measurements,
    get_node_outputs,
    create_frame,
    apply_load_frame,
    analyze,
)

from models.painter import create_bridge

from utilities_experimental import(
    apply_load_bridge, # TODO CC: first pass clean
    apply_load_bridge_multi_support, # TODO CC+NG: after clean apply_load_bridge, absorb
    save_displacements, # TODO CC: verify and move to utilities
    save_strain_stress, # TODO CC: verify and move to utilities
)

# Analysis configuration
SID_METHOD = 'srim'
MODEL = "bridge" # "frame", "bridge"
MULTISUPPORT = False
ELASTIC = True
LOAD_EVENTS = False

# Verbosity
# False means print nothing;
# True or 1 means print progress messages only;
# 2 means print progress and validation messages
VERBOSE = 1

# Main output directory
OUT_DIR = Path(f"{MODEL}")/("elastic" if ELASTIC else "inelastic")
os.makedirs(OUT_DIR, exist_ok=True)

# TODO CC: check
FIELD_OUT_DIR = Path(MODEL) / "field"
os.makedirs(FIELD_OUT_DIR, exist_ok=True)

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
                for file in list(Path(f"uploads/CE89324/").glob("????????*.[zZ][iI][pP]"))
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
            # TODO CC: check channel mapping/order for frame in-field accel/displ outputs
            output_channels_accel = [3, 4, 6, 7, 9, 10] # A2X_1_W, A2Y, A3X_2_W, A3Y, A4X_3_W, A4Y
            output_channels_displ = [21, 22, 23, 24, 25, 26] #WP1_1stFloor_N, WP2_1stFloor_S, WP3_2ndFloor_N, WP4_2ndFloor_S, WP5_3rdFloor_N, WP6_3rdFloor_S 
    elif MODEL == "bridge":
        if not MULTISUPPORT:
            # Labeled channel numbers from quakeio object
            input_channels = [1,3] # x, y
            # TODO CC: check channel mapping/order for bridge in-field outputs
            output_channels = [9, 7, 4]
        else:
            # Labeled channel numbers from quakeio object
            input_channels = [1,3,15,17,18,20] # ordered arbitrarily
            # Nodes for excitation, order corresponds to input_channels
            input_nodes = [6, 6, 0, 0, 1, 1] # TODO CC: check support-node mapping for each input channel
            # DOFs for excitation, order corresponds to input_channels
            input_dofs = [1, 2, 1, 2, 1, 2]  # TODO CC: check DOF mapping 1 is X, 2 is Y

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

        # TODO CC: check
        field_event_dir = FIELD_OUT_DIR / event_id
        os.makedirs(field_event_dir, exist_ok=True)

        # Model and system identification inputs (acceleration, in/s²)
        if MODEL == "frame":
            array, sensor_names, sensor_units, time_raw, dt = get_249_data(event)

            input_units = sensor_units[input_channels[0]]
            assert all(sensor_units[i] == input_units for i in input_channels)
            inputs = array[input_channels]*scale_249_units(units=input_units)
            # TODO CC: check frame in-field unit consistency (displ in, accel in/s^2)
            displ_units = sensor_units[output_channels_displ[0]]
            assert all(sensor_units[i] == displ_units for i in output_channels_displ)
            outputs_displ_field = array[output_channels_displ] 
            output_units = sensor_units[output_channels_accel[0]]
            assert all(sensor_units[i] == output_units for i in output_channels_accel)
            outputs_accel_field = array[output_channels_accel] * scale_249_units(units=output_units)


            if VERBOSE >= 2:
                print(
                    f"frame sensor x: \n"
                        f"\tName = {sensor_names[input_channels[0]]}\n"
                        f"\tUnits = {sensor_units[input_channels[0]]}\n"
                    f"frame sensor y: \n"
                        f"\tName = {sensor_names[input_channels[1]]}\n"
                        f"\tUnits = {sensor_units[input_channels[1]]}"
                    )
                print("accel channels:")
                for ch in output_channels_accel:
                    print(ch, sensor_names[ch], sensor_units[ch])

                print("displ channels:")
                for ch in output_channels_displ:
                    print(ch, sensor_names[ch], sensor_units[ch])
                
        elif MODEL == "bridge":
            input_units = units.cmps2

            try:
                # TODO CC: check get_measurement
                # to allow intepretation as something to 
                # get any in-field measurement from a quakeio record.
                # rename all the variables to remove reference to an "input"
                # return a dictionary instead of an array
                # the dictionary keys are channel numbers, the values
                # are timeseries
                # i put get_measurement in utilities_experimental. You can move to utilities after checking
                #i changed to unit.cmps2
                # bridge input accel
                input_scale = input_units 

                measurements, dt = get_measurements(
                    i,
                    events=events,
                    channels=input_channels,
                    scale=input_scale,
                )
                inputs = np.vstack([measurements[ch] for ch in input_channels])

                # TODO CC: check bridge in-field unit conversion (cm -> in, cm/s^2 -> in/s^2)
                accel_measurements, _ = get_measurements(
                        i, events=events, channels=output_channels, scale=input_scale, response="accel"
                    )
                outputs_accel_field = np.vstack([accel_measurements[ch] for ch in output_channels])
                displ_measurements, _ = get_measurements(
                    i, events=events, channels=output_channels, scale=input_scale, response="displ"
                )
                outputs_displ_field = np.vstack([displ_measurements[ch] for ch in output_channels])

            except:
                print(f"Error getting inputs for event {event_id}. Skipping event.")
                continue
        
        # For uniform excitation, inputs shape should be (2, nt)
        # For multi-support excitation, inputs shape should be (len(input_channels), nt)
        nin,nt = inputs.shape
        assert nin==len(input_channels) if MULTISUPPORT else nin==2
        if VERBOSE >= 2:
            print("Requested input channels:", input_channels)
            print(f"Event {event_id} time series length: {nt}, Time step dt = {dt}")



        # Finite element model
        if MODEL == 'frame':
            output_nodes = [5, 5, 10, 10, 15, 15] # TODO CC: check node order 
            output_dofs = [1, 2, 1, 2, 1, 2] # TODO CC: check dof order 
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
            

        elif MODEL == 'bridge':
            output_nodes = [9, 9, 3, 3, 10, 10] # TODO CC: check node order 
            output_dofs = [1, 2, 1, 2, 1, 2] # TODO CC: check dof order 
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
                # TODO CC: After clean apply_load_bridge,
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
            displ, accel, stresses, strains, freqs_before, freqs_after = analyze(model,
                                                                    nt=nt,
                                                                    dt=dt,
                                                                    output_nodes=output_nodes,
                                                                    output_elements=output_elements,
                                                                    yFiber=yFiber,
                                                                    zFiber=zFiber,
                                                                    verbose=VERBOSE
                                                                )

        except RuntimeError as e:
            print(f"Error encountered when analyzing event {event_id}:")
            print(e)
            continue


        # Save frequencies, displacements, strains, and stresses
        np.savetxt(event_dir/"pre_eq_natural_frequencies.csv", freqs_before)
        np.savetxt(event_dir/"post_eq_natural_frequencies.csv", freqs_after)
        save_displacements(displ, dt, filename=event_dir/"displacements.csv") # TODO CC: Check if we use this file anywhere
        save_strain_stress(stresses, strains, dt, filename=event_dir/"strain_stress.csv")

        # System identification outputs 
        # TODO CC: verify node/dof  matches in-field channel ordering
        # Displacement outputs (inches)
        outputs_displ = get_node_outputs(displ, nodes=output_nodes, dofs=output_dofs)[:, 1:] # [1:] is because extra first timestep is recorded during analysis
        # Acceleration outputs (inches/second/second)
        outputs_accel = get_node_outputs(accel, nodes=output_nodes, dofs=output_dofs)[:, 1:]


        assert inputs.shape[1] == outputs_displ.shape[1], (
            "system identification inputs and outputs have different length of time samples.")
        time = np.arange(nt) * dt

        # Save dt, time, and system identification inputs and outputs
        with open(event_dir/"dt.txt", "w") as f:
            f.write(str(dt))
        np.savetxt(event_dir/"time.csv", time)
        np.savetxt(event_dir/"inputs.csv", inputs)
        np.savetxt(event_dir/"outputs_displ.csv", outputs_displ)
        np.savetxt(event_dir/"outputs_accel.csv", outputs_accel)
        np.savetxt(field_event_dir / "outputs_displ_field.csv", outputs_displ_field)
        np.savetxt(field_event_dir / "outputs_accel_field.csv", outputs_accel_field)
        
        with open(field_event_dir / "dt.txt", "w") as f:
            f.write(str(dt))

        if False: # TODO CC: Debug
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


            system_full = sysid(inputs, outputs_displ, method=SID_METHOD, **options)
            A,B,C,D, *rest = system_full
            system = (A,B,C,D)
            with open(event_dir/f"system_{SID_METHOD}.pkl", "wb") as f:
                pickle.dump(system, f)
