"""
Get input and output data from field and model event responses.
Performs finite element analysis.
"""

import argparse
from pathlib import Path
import os
import glob
import pickle
from get_249_data import get_249_data, scale_249_units
import numpy as np
import quakeio
import xara.units.iks as units
import pickle
from utilities import (
    get_measurements,
    get_node_outputs,
    create_and_save_csv,
)

from models.painter import create_bridge
from models.frame import create_frame, apply_load_frame
from models.analysis import analyze

from utilities_experimental import(
    apply_load_bridge, # TODO CC: first pass clean
    apply_load_bridge_multi_support, # TODO CC+NG: after clean apply_load_bridge, absorb
    save_strain_stress, # TODO CC: verify and move to utilities
    save_force_deformation,
    triangulate_wirepot
)




def load_events(structure, upload_dir, from_scratch, verbose=False):
    if structure == "frame":
        # events are a list of filepaths to txt
        frame_event_pattern = str(upload_dir / "CE249_2024_Lab4data" / "ce249Run*.txt")
        events = sorted(glob.glob(frame_event_pattern))
    elif structure == "bridge":
        # events are a list of quakeio objects
        bridge_upload_dir = upload_dir / "CE89324"
        if from_scratch:
            if verbose:
                events = sorted([
                    print(file) or quakeio.read(file, exclusions=["*filter*"])
                    for file in list(bridge_upload_dir.glob("????????*.[zZ][iI][pP]"))
                ], key=lambda event: abs(event["peak_accel"]))
            else:
                events = sorted([
                    quakeio.read(file, exclusions=["*filter*"])
                    for file in list(bridge_upload_dir.glob("????????*.[zZ][iI][pP]"))
                ], key=lambda event: abs(event["peak_accel"]))
            with open("events.pkl","wb") as f:
                pickle.dump(events,f)
        else:
            with open("events.pkl","rb") as f:
                events = pickle.load(f)
    if verbose:
        print(f"Total events loaded: {len(events)}")

    return events


def set_channels_dofs(structure, multisupport):

    channels_dofs = {
        "input": {
            "field": {
                "channels": {},
                "dofs": {},
            },
            "model": {
                "channels": {},
                "dofs": {},
                "nodes": {},
            },
        },
        "output": {
            "channels": {},
            "dofs": {},
        },
    }

    if structure == "frame":
        if not multisupport:
            # Rows in data array parsed from txt file
            for source in ["field","model"]:
                channels_dofs["input"][source]["channels"] = {
                    "accel": [0, 2], # x, y
                    "displ": [34, 35], # x, y
                }
                channels_dofs["input"][source]["dofs"] = [1, 2]
        else:
            raise ValueError("Multisupport is not applicable for frame structure.")
        channels_dofs["output"]["channels"] = {
            "accel": [3, 4, 6, 7, 9, 10], # A2X_1_W, A2Y, A3X_2_W, A3Y, A4X_3_W, A4Y
            "displ": [21, 22, 23, 24, 25, 26], # WP1_1stFloor_N, WP2_1stFloor_S, WP3_2ndFloor_N, WP4_2ndFloor_S, WP5_3rdFloor_N, WP6_3rdFloor_S
        }
        channels_dofs["output"]["dofs"] = [1, 2, 1, 2, 1, 2]
        
    elif structure == "bridge":
            # input channels are labeled channel numbers from quakeio
            # object, parsed from CESMD.
            # See https://www.strongmotioncenter.org/NCESMD/photos/CGS/lllayouts/ll89324.pdf
            # Note that X = East, Y = North, and Z = Up in FE model
            # input dofs are FEM DOFs for excitation, order corresponds
            # to input_channels.
            # X=1, 2=Y, 3=Z; Negative values indicate flipped coordinates.
            # If coordinates are flipped, the sensor time series are
            # sign-flipped when retrieved.
        if not multisupport:
            # Field system identification uses the three transverse ground
            # sensors. The FE models keep the original two uniform-excitation
            # inputs (longitudinal channel 1 and transverse channel 3).
            channels_dofs["input"]["field"]["channels"] = {
                "accel": [3, 17, 20], # transverse ground sensors
                "displ": [3, 17, 20], # transverse ground sensors
            }
            channels_dofs["input"]["field"]["dofs"] = [2, 2, 2]
            channels_dofs["input"]["model"]["channels"] = {
                "accel": [1, 3], # longitudinal and transverse ground sensors
                "displ": [1, 3], # longitudinal and transverse ground sensors
            }
            channels_dofs["input"]["model"]["dofs"] = [-1, 2]
        else:
            # input nodes are FE model nodes for excitation; order
            # corresponds to input channels
            for source in ["field","model"]:
                channels_dofs["input"][source]["channels"] = {
                    "accel": [1, 3, 15, 17, 18, 20], # longitudinal and transverse ground sensors
                    "displ": [1, 3, 15, 17, 18, 20], # longitudinal and transverse ground sensors
                }
                channels_dofs["input"][source]["dofs"] = [-1, 2, -1, 2, -1, 2]
            channels_dofs["input"]["model"]["nodes"] = [4, 4, 1, 1, 0, 0]
        channels_dofs["output"]["channels"] = {
            "accel": [4, 7, 9], # A2, A3, A4
            "displ": [4, 7, 9], # WP2, WP3, WP4
        }
        channels_dofs["output"]["dofs"] = [2, 2, 2]

    return channels_dofs


def get_event_data(structure,
                     event,
                     event_idx,
                     event_id,
                     events,
                     channels_dofs,
                     inputs,
                     outputs,
                     verbose=False):

    if structure == "frame":     
        array, sensor_names, sensor_units, time_raw, inputs["field"]["dt"] = get_249_data(event)

        inputs["field"]["displacement"] = np.vstack([np.sign(dof)*array[ch]*scale_249_units(units=sensor_units[ch])
                                                        for ch,dof in zip(channels_dofs["input"]["field"]["channels"]["displ"],
                                                                          channels_dofs["input"]["field"]["dofs"])])
        inputs["field"]["acceleration"] = np.vstack([np.sign(dof)*array[ch]*scale_249_units(units=sensor_units[ch])
                                                        for ch,dof in zip(channels_dofs["input"]["field"]["channels"]["accel"],
                                                                          channels_dofs["input"]["field"]["dofs"])])
        
        outputs["field"]["displacement"] = np.vstack([array[ch]*scale_249_units(units=sensor_units[ch])
                                            for ch in channels_dofs["output"]["channels"]["displ"]])
        outputs["field"]["displacement"] = triangulate_wirepot(outputs["field"]["displacement"])

        outputs["field"]["acceleration"] = np.vstack([np.sign(dof)*array[ch]*scale_249_units(units=sensor_units[ch])
                                            for ch,dof in zip(channels_dofs["output"]["channels"]["accel"],
                                                              channels_dofs["output"]["dofs"])])

        # Frame field and FE cases use the same ground inputs.
        inputs["model"] = {
            "dt": inputs["field"]["dt"],
            "displacement": inputs["field"]["displacement"].copy(),
            "acceleration": inputs["field"]["acceleration"].copy(),
        }

        if verbose >= 2:
            print("Frame input accel channels, names, and units:")
            for ch in channels_dofs["input"]["field"]["channels"]["accel"]:
                print(ch, sensor_names[ch], sensor_units[ch])
            print("Frame input displ channels, names, and units:")
            for ch in channels_dofs["input"]["field"]["channels"]["displ"]:
                print(ch, sensor_names[ch], sensor_units[ch])
            print("Frame output accel channels, names, and units:")
            for ch in channels_dofs["output"]["channels"]["accel"]:
                print(ch, sensor_names[ch], sensor_units[ch])
            print("Frame output displ channels, names, and units:")
            for ch in channels_dofs["output"]["channels"]["displ"]:
                print(ch, sensor_names[ch], sensor_units[ch])
            
    elif structure == "bridge":
        measurement_units_accel = units.cmps2
        measurement_units_displ = units.cm

        # Read in-field measurements. Scale by units and flip sign where needed.

        accel_channels = list(dict.fromkeys([
            *channels_dofs["input"]["field"]["channels"]["accel"],
            *channels_dofs["input"]["model"]["channels"]["accel"],
            *channels_dofs["output"]["channels"]["accel"]
        ]))
        displ_channels = list(dict.fromkeys([
            *channels_dofs["input"]["field"]["channels"]["displ"],
            *channels_dofs["input"]["model"]["channels"]["displ"],
            *channels_dofs["output"]["channels"]["displ"]
        ]))
        measurements_accel, inputs["field"]["dt"] = get_measurements(
            event_idx, events=events, channels=accel_channels,
            scale=measurement_units_accel, response="accel")
        measurements_displ, _  = get_measurements(
            event_idx, events=events, channels=displ_channels,
            scale=measurement_units_displ, response="displ")

        inputs["field"]["acceleration"] =  np.vstack([np.sign(dof)*measurements_accel[ch]
                                            for ch,dof in zip(channels_dofs["input"]["field"]["channels"]["accel"],
                                                              channels_dofs["input"]["field"]["dofs"])])
        
        inputs["field"]["displacement"] =  np.vstack([np.sign(dof)*measurements_displ[ch]
                                            for ch,dof in zip(channels_dofs["input"]["field"]["channels"]["displ"],
                                                              channels_dofs["input"]["field"]["dofs"])])

        inputs["model"] = {
            "dt": inputs["field"]["dt"],
            "acceleration": np.vstack([
                np.sign(dof) * measurements_accel[ch]
                for ch, dof in zip(channels_dofs["input"]["model"]["channels"]["accel"],
                                   channels_dofs["input"]["model"]["dofs"])
            ]),
            "displacement": np.vstack([
                np.sign(dof) * measurements_displ[ch]
                for ch, dof in zip(channels_dofs["input"]["model"]["channels"]["displ"],
                                   channels_dofs["input"]["model"]["dofs"])
            ]),
        }

        outputs["field"]["acceleration"] = np.vstack([np.sign(dof)*measurements_accel[ch]
                                            for ch,dof in zip(channels_dofs["output"]["channels"]["accel"],
                                                              channels_dofs["output"]["dofs"])])
        
        outputs["field"]["displacement"] = np.vstack([np.sign(dof)*measurements_displ[ch] 
                                                for ch,dof in zip(channels_dofs["output"]["channels"]["displ"],
                                                                  channels_dofs["output"]["dofs"])])

        if verbose >= 2:
            print("Bridge input accel channels:", channels_dofs["input"]["field"]["channels"]["accel"])
            print("Bridge input displ channels:", channels_dofs["input"]["field"]["channels"]["displ"])
            print("Bridge output accel channels:", channels_dofs["output"]["channels"]["accel"])
            print("Bridge output displ channels:", channels_dofs["output"]["channels"]["displ"])

    # Verify inputs; shape should be (len(input_channels), nt)
    nin,nt = inputs["field"]["acceleration"].shape
    assert nin==len(channels_dofs["input"]["field"]["channels"]["accel"]), (
        "Number of rows in input acceleration array does not match number of input channels."
    )
    if verbose >= 2:
        print(f"Event {event_id} time series length: {nt} samples, Time step (dt) = {inputs['field']['dt']}")

    return inputs, outputs, nt


def save_field_data(inputs, outputs, nt, event_id, rewrite=True):
    inputs["field"]["time"] = np.arange(nt) * inputs["field"]["dt"]

    # Save measured ground inputs and measured structural responses.
    for location, quantities in (
        ("ground", inputs["field"]),
        ("structure", outputs["field"]),
    ):
        for q_name, q in quantities.items():
            create_and_save_csv(
                path=FIELD_OUT_DIR / q_name / location / f"{event_id}.csv",
                array=q,
                rewrite=rewrite,
            )

    return inputs, outputs
        

def run_finite_element(event_id,
                       structure,
                       inputs,
                       nt,
                       output_element,
                       output_response,
                       frame_zerolength,
                       frame_coupons,
                       elastic,
                       multisupport,
                       verbose=False):

    if structure == 'frame':
        output_nodes = [5, 5, 10, 10, 15, 15]
        output_elements = [output_element]
        yFiber = 2
        zFiber = 0.0
        response_mode = "material" if output_response == "force_deformation" else "fiber"
        fiber_response_dof = None
        material_deformation_dof = 2 if output_response == "force_deformation" else None
        material_force_dof = 2 if output_response == "force_deformation" else None

        if frame_zerolength not in {"element", "section"}:
            raise ValueError(
                f"Unsupported FRAME_ZEROLENGTH={frame_zerolength!r}; "
                "expected 'element' or 'section'."
            )
        model = create_frame(elastic=elastic,
                                    multisupport=multisupport,
                                    verbose=verbose,
                                    material='steel',
                                    coupons=frame_coupons,
                                    zerolength=frame_zerolength)

        model = apply_load_frame(model,
                                    inputx=inputs["model"]["acceleration"][0],
                                    inputy=inputs["model"]["acceleration"][1],
                                    dt=inputs["model"]["dt"])
        

    elif structure == 'bridge':
        output_nodes = [9, 3, 10] 
        output_elements = [output_element]
        yFiber = 22.5 
        zFiber = 0.0
        response_mode = "material" if output_response == "force_deformation" else "fiber"
        fiber_response_dof = None
        material_deformation_dof = 2 if output_response == "force_deformation" else None
        material_force_dof = 8 if output_response == "force_deformation" else None

        model = create_bridge(elastic=elastic,
                                    multisupport=multisupport,
                                    separate_deck_ends=True,
                                    verbose=verbose
                                    )
        
        if not MULTISUPPORT:
            model = apply_load_bridge(model,
                                    inputx=inputs["model"]["acceleration"][0],
                                    inputy=inputs["model"]["acceleration"][1],
                                    dt=inputs["model"]["dt"],
                                    # multisupport=multisupport,
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
                inputs=inputs["field"]["acceleration"],
                dt=inputs["field"]["dt"],
                node_channel_map=node_channel_map,
                input_channels=input_channels_accel,
            )

    try:
        displ, accel, response_x, response_y, freqs_before, freqs_after = analyze(model,
                                                                nt=nt,
                                                                dt=inputs["field"]["dt"],
                                                                output_nodes=output_nodes,
                                                                output_elements=output_elements,
                                                                yFiber=yFiber,
                                                                zFiber=zFiber,
                                                                response_mode=response_mode,
                                                                fiber_response_dof=fiber_response_dof,
                                                                material_deformation_dof=material_deformation_dof,
                                                                material_force_dof=material_force_dof,
                                                                n_modes=5,
                                                                verbose=verbose
                                                            )

    except RuntimeError as e:
        if verbose:
            print(f"Error encountered when analyzing event {event_id}:")
            print(e)
        return None

    return model, displ, accel, response_x, response_y, freqs_before, freqs_after, output_nodes


def save_model_outputs(
    elastic,
    event_id,
    channels_dofs,
    inputs,
    outputs,
    nt,
    displ,
    accel,
    response_x,
    response_y,
    freqs_before,
    freqs_after,
    output_response,
    rewrite=True,
    verbose=False):
    """
    Save all model outputs, including:
    frequencies, displacements, and element-response pairs
    model 
    """

    source = "elastic" if elastic else "inelastic"

    for quantity,label in zip(
                            [freqs_before,freqs_after],
                            ["frequency_pre_eq","frequency_post_eq"]):
        create_and_save_csv(
            path=MODEL_OUT_DIR / label  / "structure" / f"{event_id}.csv",
            array=quantity,
            rewrite=rewrite
            )

    if output_response == "force_deformation":
        fd_path = MODEL_OUT_DIR / "force_deformation" / "structure" / f"{event_id}.csv"
        fd_path.parent.mkdir(parents=True, exist_ok=True)
        save_force_deformation(response_y, response_x, inputs["field"]["dt"], filename=fd_path)
    else:
        ss_path = MODEL_OUT_DIR / "strain_stress" / "structure" / f"{event_id}.csv"
        ss_path.parent.mkdir(parents=True, exist_ok=True)
        save_strain_stress(response_y, response_x, inputs["field"]["dt"], filename=ss_path)

    # FE model outputs, used as true outputs in system identification 
    # Note, slice [1:] is because extra first timestep is recorded during analysis
    # Displacement outputs (inches)
    outputs["model"]["displacement"] = get_node_outputs(displ, nodes=output_nodes, dofs=channels_dofs["output"]["dofs"])[:, 1:]
    # Acceleration outputs (inches/second/second)
    outputs["model"]["acceleration"] = get_node_outputs(accel, nodes=output_nodes, dofs=channels_dofs["output"]["dofs"])[:, 1:]

    assert inputs["model"]["acceleration"].shape[1] == outputs["model"]["displacement"].shape[1], (
        "system identification training inputs and outputs have different length of time samples.")
    inputs["model"]["time"] = np.arange(nt) * inputs["model"]["dt"]

    if verbose >= 2:
        for qdict,qdict_name in zip([inputs,outputs],["inputs","outputs"]):
            print(qdict_name, "saved:")
            for source,quantities in qdict.items():
                print(source, list(quantities.keys()))

    # Use create_and_save_csv to save csvs, with argument rewrite
    for location,quantities in zip(["ground","structure"],[inputs["model"],outputs["model"]]):
        for q_name,q in quantities.items():
            create_and_save_csv(
                path = MODEL_OUT_DIR / q_name / location / f"{event_id}.csv",
                array = q,
                rewrite=rewrite
            )

    return inputs, outputs


def parse_data_args():
    parser = argparse.ArgumentParser(description="Get input and output data from field and model event responses.")
    parser.add_argument("--structure", type=str, default="bridge", choices=["frame", "bridge"], help="Structure type: 'frame' or 'bridge'.")
    parser.add_argument("--multisupport", action="store_true", help="Use multisupport excitation for bridge structure.")
    parser.add_argument("--elastic", action="store_true", help="Use elastic model; otherwise inelastic.")
    parser.add_argument("--field_only", action="store_true", help="Save measured field data without creating or analyzing the FE model.")
    parser.add_argument("--from_scratch", action="store_true", help="Load events from scratch instead of using cached events.pkl.")
    parser.add_argument("--no_frame_coupons", action="store_false", dest="frame_coupons", help="Disable coupons in frame model.")
    parser.add_argument("--frame_zerolength", type=str, default="section", choices=["element", "section"], help="Zerolength element type for frame model: 'element' or 'section'.")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level: 0 (silent), 1 (progress), 2 (progress + validation).")
    return parser.parse_args()



if __name__ == "__main__":

    # Read custom analysis configuration parameters from command line arguments
    args = parse_data_args()
    STRUCTURE = args.structure
    MULTISUPPORT = args.multisupport
    ELASTIC = args.elastic
    FIELD_ONLY = args.field_only
    FROM_SCRATCH = args.from_scratch
    FRAME_COUPONS = args.frame_coupons
    FRAME_ZEROLENGTH = args.frame_zerolength
    VERBOSE = args.verbose

    # Fixed analysis configuration parameters
    UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
    SID_METHOD = 'srim'
    # Save measured field data without creating or analyzing the FE model.
    FRAME_OUTPUT_ELEMENT = int(os.environ.get("FRAME_OUTPUT_ELEMENT", "102"))
    FRAME_OUTPUT_RESPONSE = "force_deformation" \
                                if FRAME_COUPONS and FRAME_ZEROLENGTH=="element" \
                                and FRAME_OUTPUT_ELEMENT in [str(i) for i in np.arange(101,117)]+[str(i) for i in np.arange(201,217)] \
                                else "stress_strain"
    BRIDGE_OUTPUT_ELEMENT = int(os.environ.get("BRIDGE_OUTPUT_ELEMENT", "107"))
    BRIDGE_OUTPUT_RESPONSE = "force_deformation" if BRIDGE_OUTPUT_ELEMENT in ["107","108","109","110"] else "stress_strain"

    # Main output directory
    BASE_DIR = Path("Modeling")
    MODEL_OUT_DIR = BASE_DIR / STRUCTURE / ("elastic" if ELASTIC else "inelastic")
    os.makedirs(MODEL_OUT_DIR, exist_ok=True)
    FIELD_OUT_DIR = BASE_DIR / STRUCTURE / "field"
    os.makedirs(FIELD_OUT_DIR, exist_ok=True)

    # Print analysis configuration
    if VERBOSE:
        print(f"{STRUCTURE=}")
        print(f"{ELASTIC=}")

    # Load events
    events = load_events(structure=STRUCTURE,
                         upload_dir=UPLOAD_DIR,
                         from_scratch=FROM_SCRATCH,
                         verbose=VERBOSE)

    # Perform model analysis and system identification and record responses

    # Set input channels, input dofs, output channels, and output dofs
    channels_dofs = set_channels_dofs(
        structure=STRUCTURE,
        multisupport=MULTISUPPORT
    )
    
    for event_idx,event in enumerate(events):
        # Set the event ID
        if STRUCTURE == "frame":
            # filepaths are like .../ce249Run244.txt
            event_id = Path(event).stem.replace("ce249Run", "")  # "244"
        elif STRUCTURE == "bridge":
            event_id = str(event_idx+1)
        if VERBOSE:
            print(f"\nEvent: {event}; Event ID: {event_id}")

        # Inputs (ground): dt, time, and field displ/accel
        # inputs = {
        #     "field": {"dt":dt, "time",time, "acceleration":accel}
        #          }
        # Outputs (structure): FE model displ/accel, field displ/accel
        # outputs = {
        #     "model": {"displacement":displ, "acceleration":accel},
        #     "field": {"displacement":displ, "acceleration":accel}
        #           }
        inputs = {"field": {}}
        outputs = {"model": {}, "field": {}}

        # Measurements from the field.
        # Input acceleration (in/s²) is used as model and system identification input 
        # Output displacement (in) and acceleration (in/s²) are used to compare
        # with FE model outputs and system identification outputs. 
        try:
            inputs, outputs, nt = get_event_data(structure=STRUCTURE,
                                                 event=event,
                                                 event_idx=event_idx,
                                                 event_id=event_id,
                                                 events=events,
                                                 channels_dofs=channels_dofs,
                                                 inputs=inputs,
                                                 outputs=outputs,
                                                 verbose=VERBOSE)
        except:
            if VERBOSE:
                print(f"Error getting field measurements for event {event_id}. Skipping event.")
            continue
        inputs, outputs = save_field_data(inputs, outputs, nt, event_id, rewrite=True)
        if FIELD_ONLY:
            if VERBOSE:
                print(f"Saved field data for event {event_id}; skipping FE analysis.")
            continue

        # Create, load, and analyze the FE model, and save the model outputs.
        model_response = run_finite_element(
            event_id=event_id,
            structure=STRUCTURE,
            inputs=inputs,
            nt=nt,
            output_element=FRAME_OUTPUT_ELEMENT if STRUCTURE=="frame" else BRIDGE_OUTPUT_ELEMENT,
            output_response=FRAME_OUTPUT_RESPONSE if STRUCTURE=="frame" else BRIDGE_OUTPUT_RESPONSE,
            frame_zerolength=FRAME_ZEROLENGTH if STRUCTURE=="frame" else None,
            frame_coupons=FRAME_COUPONS if STRUCTURE=="frame" else None,
            elastic=ELASTIC,
            multisupport=MULTISUPPORT,
            verbose=VERBOSE
        )
        if model_response is None:
            continue
        else:
            model, displ, accel, response_x, response_y, freqs_before, freqs_after, output_nodes = model_response

        # Save model outputs
        inputs, outputs = save_model_outputs(
            elastic=ELASTIC,
            event_id=event_id,
            channels_dofs=channels_dofs,
            inputs=inputs,
            outputs=outputs,
            nt=nt,
            displ=displ,
            accel=accel,
            response_x=response_x,
            response_y=response_y,
            freqs_before=freqs_before,
            freqs_after=freqs_after,
            output_response=FRAME_OUTPUT_RESPONSE if STRUCTURE=="frame" else BRIDGE_OUTPUT_RESPONSE,
            rewrite=True,
            verbose=VERBOSE
        )
