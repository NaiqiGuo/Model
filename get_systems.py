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
    triangulate_wirepot
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

    # Set input channels, input dofs, output channels, and output dofs
    if MODEL == "frame":
        if not MULTISUPPORT:
            # Rows in data array parsed from txt file
            input_channels = [0, 2] # x, y
            input_dofs = [1, 2]
        output_channels_accel = [3, 4, 6, 7, 9, 10] # A2X_1_W, A2Y, A3X_2_W, A3Y, A4X_3_W, A4Y
        output_channels_displ = [21, 22, 23, 24, 25, 26] # WP1_1stFloor_N, WP2_1stFloor_S, WP3_2ndFloor_N, WP4_2ndFloor_S, WP5_3rdFloor_N, WP6_3rdFloor_S 
        output_dofs = [1, 2, 1, 2, 1, 2]

        wirepot_ref_226 = None
        ref_event = "uploads/CE249_2024_Lab4data/ce249Run226.txt"
        array_ref, sensor_names_ref, sensor_units_ref, time_raw_ref, dt_ref = get_249_data(ref_event)
        wirepot_ref_226 = np.vstack([array_ref[ch] * scale_249_units(units=sensor_units_ref[ch])
            for ch in output_channels_displ])
        
    elif MODEL == "bridge":
            # `input_channels` are labeled channel numbers from quakeio
            # object, parsed from CESMD.
            # See https://www.strongmotioncenter.org/NCESMD/photos/CGS/lllayouts/ll89324.pdf
            # Note that X = East, Y = North, and Z = Up in FE model
            # `input_dofs` are FEM DOFs for excitation, order corresponds
            # to input_channels.
            # X=1, 2=Y, 3=Z; Negative values indicate flipped coordinates.
            # If coordinates are flipped, the sensor time series are
            # sign-flipped when retrieved.
        if not MULTISUPPORT:
            input_channels = [1, 3]
            input_dofs = [-1, 2] # CHECK NG: I had to add some logic for the X direction (it is opposite from the channel direction)
        else:
            # `input_nodes` are FE model nodes for excitation; order
            # corresponds to `input_channels`
            input_channels = [1, 3, 15, 17, 18, 20]
            input_nodes = [4, 4, 1, 1, 0, 0] # CHECK NG: should be node 4 instead of node 6, and nodes 1 and 0 were flipped
            input_dofs = [-1, 2, -1, 2, -1, 2, -1, 2]  # CHECK NG: I had to add some logic for the X direction (it is opposite from the channel direction)
        # CHECK NG: order was opposite; nodes 4 and 9 were flipped. (X goes from west to east)
        output_channels = [4, 7, 9]
        # CHECK NG: moved output dofs here. Note: Y direction only.
        output_dofs = [2, 2, 2]

    for i,event in enumerate(events):
        if MODEL == "frame":
            # filepaths are like .../ce249Run244.txt
            event_id = Path(event).stem.replace("ce249Run", "")  # "244"
        elif MODEL == "bridge":
            event_id = str(i+1)
        if VERBOSE:
            print(f"\nEvent: {event}; Event ID: {event_id}")

        # Result output directory
        event_dir = OUT_DIR/event_id
        os.makedirs(event_dir, exist_ok=True)

        # TODO CC: check
        field_event_dir = FIELD_OUT_DIR / event_id
        os.makedirs(field_event_dir, exist_ok=True)

        # Measurements from the field.
        # Input acceleration (in/s²) is used as model and system identification input 
        # Output displacement (in) and acceleration (in/s²) are used to compare
        # with FE model outputs and system identification outputs. 
        if MODEL == "frame":     
            array, sensor_names, sensor_units, time_raw, dt = get_249_data(event)

            # Check NG: I added in the logic for flipping the sign of the sensor time series
            # here (not needed for the frame, but included for consistency).
            # Also, I consolidated the units into this computation.
            inputs = np.vstack([np.sign(dof)*array[ch]*scale_249_units(units=sensor_units[ch])
                                for ch,dof in zip(input_channels,input_dofs)])
            
            outputs_displ_field = np.vstack([array[ch]*scale_249_units(units=sensor_units[ch])
                                             for ch in output_channels_displ])
            # TODO CC check: 
            # Created a function, triangulate_wirepot that computes 2D triangulation 
            # to obtain X & Y displacements.
            # Note that in Lab 4 you were given ReadDAQ_2Dtriangulation.m and getTriXY.m
            outputs_displ_field = triangulate_wirepot(outputs_displ_field, wirepot_ref=wirepot_ref_226)

            outputs_accel_field = np.vstack([np.sign(dof)*array[ch]*scale_249_units(units=sensor_units[ch])
                                             for ch,dof in zip(output_channels_accel,output_dofs)])

            if VERBOSE >= 2:
                if not MULTISUPPORT:
                    print(
                        f"input sensor x: \n"
                            f"\tName = {sensor_names[input_channels[0]]}\n"
                            f"\tUnits = {sensor_units[input_channels[0]]}\n"
                        f"input sensor y: \n"
                            f"\tName = {sensor_names[input_channels[1]]}\n"
                            f"\tUnits = {sensor_units[input_channels[1]]}"
                        )
                print("output accel channels:")
                for ch in output_channels_accel:
                    print(ch, sensor_names[ch], sensor_units[ch])
                print("output displ channels:")
                for ch in output_channels_displ:
                    print(ch, sensor_names[ch], sensor_units[ch])
                
        elif MODEL == "bridge":
            measurement_units_accel = units.cmps2
            measurement_units_displ = units.cm

            try:
                # Read in-field measurements. Scale by units and flip sign where needed.

                # CHECK NG: limit calls to get_measurements,
                # because the parsing done inside it in extract_channels is slow.
                measurements_accel, dt = get_measurements(
                    i, events=events, channels=[*input_channels, *output_channels],
                    scale=measurement_units_accel, response="accel")
                measurements_displ, dt = get_measurements(
                    i, events=events, channels=[*input_channels, *output_channels],
                    scale=measurement_units_displ, response="displ")

                # CHECK NG: This is I where incorporated the logic for flipping the sign of the sensors
                inputs =                np.vstack([np.sign(dof)*measurements_accel[ch]
                                                   for ch,dof in zip(input_channels,input_dofs)])

                outputs_accel_field =   np.vstack([np.sign(dof)*measurements_accel[ch]
                                                   for ch,dof in zip(output_channels,output_dofs)])
                
                outputs_displ_field =   np.vstack([np.sign(dof)*measurements_displ[ch] 
                                                   for ch,dof in zip(output_channels,output_dofs)])

            except:
                print(f"Error getting measurements for event {event_id}. Skipping event.")
                continue
        
        # Verify inputs; shape should be (len(input_channels), nt)
        nin,nt = inputs.shape
        assert nin==len(input_channels)
        if VERBOSE >= 2:
            print("Requested input channels:", input_channels)
            print(f"Event {event_id} time series length: {nt}, Time step dt = {dt}")



        # Finite element model
        if MODEL == 'frame':
            output_nodes = [5, 5, 10, 10, 15, 15]
            output_elements = [1, 5, 9]
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
            output_nodes = [9, 3, 10] # CHECK NG: We are only using Y direction, so no repeat
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

        # FE model outputs, used as true outputs in system identification 
        # Displacement outputs (inches)
        # Note, slice [1:] is because extra first timestep is recorded during analysis
        outputs_displ = get_node_outputs(displ, nodes=output_nodes, dofs=output_dofs)[:, 1:]
        # Acceleration outputs (inches/second/second)
        outputs_accel = get_node_outputs(accel, nodes=output_nodes, dofs=output_dofs)[:, 1:]


        assert inputs.shape[1] == outputs_displ.shape[1], (
            "system identification training inputs and outputs have different length of time samples.")
        time = np.arange(nt) * dt

        # Save dt, time, FE model inputs and outputs, and in-field inputs and outputs
        with open(event_dir/"dt.txt", "w") as f:
            f.write(str(dt))
        np.savetxt(event_dir/"time.csv", time)
        np.savetxt(event_dir/"inputs.csv", inputs)
        np.savetxt(event_dir/"outputs_displ.csv", outputs_displ)
        np.savetxt(event_dir/"outputs_accel.csv", outputs_accel)
        np.savetxt(field_event_dir/"outputs_displ.csv", outputs_displ_field)
        np.savetxt(field_event_dir/"outputs_accel.csv", outputs_accel_field)


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
