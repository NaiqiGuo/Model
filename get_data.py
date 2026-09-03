"""
Get input and output data from field and model event responses.
Performs finite element analysis.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
import os
import glob
import pickle
from get_249_data import get_249_data, scale_249_units
import numpy as np
import quakeio
import xara.units.iks as units
from utilities import (
    get_measurements,
    get_node_outputs,
    create_and_save_csv,
)

from models.painter import create_bridge, apply_load_bridge
from models.frame import create_frame, apply_load_frame
from models.analysis import analyze

from utilities_experimental import(
    save_strain_stress, # TODO CC: verify and move to utilities
    save_force_deformation,
    triangulate_wirepot
)


def load_events(cfg):
    structure = cfg.structure
    if structure == "frame":
        # events are a list of filepaths to txt
        frame_event_pattern = str(cfg.upload_dir / "CE249_2024_Lab4data" / "ce249Run*.txt")
        events = sorted(glob.glob(frame_event_pattern))
    elif structure == "bridge":
        # events are a list of quakeio objects
        bridge_upload_dir = cfg.upload_dir / "CE89324"
        if cfg.from_scratch:
            if cfg.verbose:
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
    if cfg.verbose:
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
        channels_dofs["output"]["nodes"] = [5, 5, 10, 10, 15, 15]

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
            # Field
            channels_dofs["input"]["field"]["channels"] = {
                "accel": [1, 3, 15, 17, 18, 20],
                "displ": [1, 3, 15, 17, 18, 20],
            }
            channels_dofs["input"]["field"]["dofs"] = [-1, 2, -1, 2, -1, 2]
            # Model: use channels 1, 3 for both of the column bases (13 north, 14 south)
            channels_dofs["input"]["model"]["channels"] = {
                "accel": [1, 3, 1, 3, 15, 17, 18, 20],
                "displ": [1, 3, 1, 3, 15, 17, 18, 20],
            }
            channels_dofs["input"]["model"]["dofs"] = [-1, 2, -1, 2, -1, 2, -1, 2]
            channels_dofs["input"]["model"]["nodes"] = [13, 13, 14, 14, 12, 12, 11, 11]
        channels_dofs["output"]["channels"] = {
            "accel": [4, 7, 9], # A2, A3, A4
            "displ": [4, 7, 9], # WP2, WP3, WP4
        }
        channels_dofs["output"]["dofs"] = [2, 2, 2]
        channels_dofs["output"]["nodes"] = [9, 3, 10]

    return channels_dofs


@dataclass(frozen=True)
class RunConfig:
    """Run-constant analysis configuration, built from CLI args."""
    structure: str
    multisupport: bool
    elastic: bool
    frame_coupons: bool | None
    frame_zerolength: str | None
    field_only: bool
    from_scratch: bool
    verbose: int
    sid_method: str
    upload_dir: Path
    base_dir: Path
    model_out_dir: Path
    field_out_dir: Path
    output_element: int
    output_response: str
    channels_dofs: dict

    @classmethod
    def from_args(cls, args) -> "RunConfig":
        structure = args.structure
        upload_dir = Path(__file__).resolve().parent / "uploads"

        frame_output_element = int(os.environ.get("FRAME_OUTPUT_ELEMENT", "102"))
        frame_output_response = "force_deformation" \
                                    if args.frame_coupons and args.frame_zerolength == "element" \
                                    and frame_output_element in [*range(101,117), *range(201,217)] \
                                    else "stress_strain"
        bridge_output_element = int(os.environ.get("BRIDGE_OUTPUT_ELEMENT", "107"))
        bridge_output_response = "force_deformation" if bridge_output_element in ["107","108","109","110"] else "stress_strain"

        if structure == "frame":
            output_element = frame_output_element
            output_response = frame_output_response
            frame_coupons = args.frame_coupons
            frame_zerolength = args.frame_zerolength
        else:
            output_element = bridge_output_element
            output_response = bridge_output_response
            frame_coupons = None
            frame_zerolength = None

        base_dir = Path("Modeling")
        model_out_dir = base_dir / structure / ("elastic" if args.elastic else "inelastic")
        field_out_dir = base_dir / structure / "field"
        os.makedirs(model_out_dir, exist_ok=True)
        os.makedirs(field_out_dir, exist_ok=True)

        channels_dofs = set_channels_dofs(structure, args.multisupport)

        return cls(
            structure=structure,
            multisupport=args.multisupport,
            elastic=args.elastic,
            frame_coupons=frame_coupons,
            frame_zerolength=frame_zerolength,
            field_only=args.field_only,
            from_scratch=args.from_scratch,
            verbose=args.verbose,
            sid_method="srim",
            upload_dir=upload_dir,
            base_dir=base_dir,
            model_out_dir=model_out_dir,
            field_out_dir=field_out_dir,
            output_element=output_element,
            output_response=output_response,
            channels_dofs=channels_dofs,
        )


class EventAnalysis:
    """
    Analysis of a single event. Reads field measurements, runs the FE model,
    and saves the results. One instance per event.

    State schema:
      inputs  (ground):    {"field": {dt, time, acceleration, displacement},
                            "model": {dt, time, acceleration, displacement}}
      outputs (structure): {"field": {acceleration, displacement},
                            "model": {acceleration, displacement}}
    """

    def __init__(self, cfg: RunConfig, event, event_idx: int):
        self.cfg = cfg
        self.event = event
        self.event_idx = event_idx
        self.event_id = self._event_id()

        self.inputs = {"field": {}}
        self.outputs = {"model": {}, "field": {}}
        self.nt = None

        # FE results, populated by run_fe():
        self.model = None
        self.displ = None
        self.accel = None
        self.response_x = None
        self.response_y = None
        self.freqs_before = None
        self.freqs_after = None

    def _event_id(self) -> str:
        if self.cfg.structure == "frame":
            # filepaths are like .../ce249Run244.txt
            return Path(self.event).stem.replace("ce249Run", "")  # "244"
        return str(self.event_idx + 1)

    def run(self):
        """
        Process this event. Load and save field measurements,
        then run FE model and save model outputs.
        """
        cfg = self.cfg
        if cfg.verbose:
            print(f"\nEvent: {self.event}; Event ID: {self.event_id}")

        # Measurements from the field.
        # Input acceleration (in/s²) is used as model and system identification input.
        # Output displacement (in) and acceleration (in/s²) are used to compare
        # with FE model outputs and system identification outputs.
        try:
            self.load_measurements()
        except:
            if cfg.verbose:
                print(f"Error getting field measurements for event {self.event_id}. Skipping event.")
            return

        self.save_field()
        if cfg.field_only:
            if cfg.verbose:
                print(f"Saved field data for event {self.event_id}; skipping FE analysis.")
            return

        if not self.run_fe():
            return
        self.save_model()

    def load_measurements(self):
        cfg = self.cfg
        channels_dofs = cfg.channels_dofs
        inputs = self.inputs
        outputs = self.outputs

        if cfg.structure == "frame":
            array, sensor_names, sensor_units, time_raw, inputs["field"]["dt"] = get_249_data(self.event)

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

            if cfg.verbose >= 2:
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

        elif cfg.structure == "bridge":
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
                self.event, channels=accel_channels,
                scale=measurement_units_accel, response="accel")
            measurements_displ, _  = get_measurements(
                self.event, channels=displ_channels,
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

            if cfg.verbose >= 2:
                print("Bridge input accel channels:", channels_dofs["input"]["field"]["channels"]["accel"])
                print("Bridge input displ channels:", channels_dofs["input"]["field"]["channels"]["displ"])
                print("Bridge output accel channels:", channels_dofs["output"]["channels"]["accel"])
                print("Bridge output displ channels:", channels_dofs["output"]["channels"]["displ"])

        # Verify inputs; shape should be (len(input_channels), nt)
        nin, self.nt = inputs["field"]["acceleration"].shape
        assert nin==len(channels_dofs["input"]["field"]["channels"]["accel"]), (
            "Number of rows in input acceleration array does not match number of input channels."
        )
        if cfg.verbose >= 2:
            print(f"Event {self.event_id} time series length: {self.nt} samples, Time step (dt) = {inputs['field']['dt']}")

    def save_field(self, rewrite=True):
        self.inputs["field"]["time"] = np.arange(self.nt) * self.inputs["field"]["dt"]

        # Save measured ground inputs and measured structural responses.
        for location, quantities in (
            ("ground", self.inputs["field"]),
            ("structure", self.outputs["field"]),
        ):
            for q_name, q in quantities.items():
                create_and_save_csv(
                    path=self.cfg.field_out_dir / q_name / location / f"{self.event_id}.csv",
                    array=q,
                    rewrite=rewrite,
                )

    def run_fe(self) -> bool:
        """Create, load, and analyze the FE model. Returns True on success."""
        cfg = self.cfg

        if cfg.structure == 'frame':
            output_elements = [cfg.output_element]
            yFiber = 4.5 # Near the edge of the 10 in x 10 in coupon cross-section
            zFiber = 0.0
            response_mode = "material" if cfg.output_response == "force_deformation" else "fiber"
            fiber_response_dof = None
            material_deformation_dof = 2 if cfg.output_response == "force_deformation" else None
            material_force_dof = 2 if cfg.output_response == "force_deformation" else None

            if cfg.frame_zerolength not in {"element", "section"}:
                raise ValueError(
                    f"Unsupported FRAME_ZEROLENGTH={cfg.frame_zerolength!r}; "
                    "expected 'element' or 'section'."
                )
            model = create_frame(elastic=cfg.elastic,
                                 multisupport=cfg.multisupport,
                                 coupons=cfg.frame_coupons,
                                 material='steel',
                                 zerolength=cfg.frame_zerolength,
                                 verbose=cfg.verbose)

            model = apply_load_frame(model,
                                     inputx=self.inputs["model"]["acceleration"][0],
                                     inputy=self.inputs["model"]["acceleration"][1],
                                     dt=self.inputs["model"]["dt"])

        elif cfg.structure == 'bridge':
            output_elements = [cfg.output_element]
            yFiber = 22.5 # Inside the column core (column total diameter 60 in; 3.175 in cover)
            zFiber = 0.0
            response_mode = "material" if cfg.output_response == "force_deformation" else "fiber"
            fiber_response_dof = None
            material_deformation_dof = 2 if cfg.output_response == "force_deformation" else None
            material_force_dof = 8 if cfg.output_response == "force_deformation" else None

            model = create_bridge(elastic=cfg.elastic,
                                  separate_deck_ends=True,
                                  verbose=cfg.verbose)


            model = apply_load_bridge(model,
                                      inputs=self.inputs["model"]["acceleration"],
                                      dt=self.inputs["model"]["dt"],
                                      multisupport=cfg.multisupport,
                                      input_nodes=cfg.channels_dofs["input"]["model"]["nodes"],
                                      input_dofs=cfg.channels_dofs["input"]["model"]["dofs"])


        try:
            displ, accel, response_x, response_y, freqs_before, freqs_after = analyze(model,
                                                                    nt=self.nt,
                                                                    dt=self.inputs["field"]["dt"],
                                                                    output_nodes=cfg.channels_dofs["output"]["nodes"],
                                                                    output_elements=output_elements,
                                                                    yFiber=yFiber,
                                                                    zFiber=zFiber,
                                                                    response_mode=response_mode,
                                                                    fiber_response_dof=fiber_response_dof,
                                                                    material_deformation_dof=material_deformation_dof,
                                                                    material_force_dof=material_force_dof,
                                                                    n_modes=5,
                                                                    verbose=cfg.verbose
                                                                )

        except RuntimeError as e:
            if cfg.verbose:
                print(f"Error encountered when analyzing event {self.event_id}:")
                print(e)
            return False

        self.model = model
        self.displ = displ
        self.accel = accel
        self.response_x = response_x
        self.response_y = response_y
        self.freqs_before = freqs_before
        self.freqs_after = freqs_after
        return True

    def save_model(self, rewrite=True):
        """
        Save all model outputs: frequencies, element-response pairs, and
        model displacement/acceleration.
        """
        cfg = self.cfg

        for quantity,label in zip(
                                [self.freqs_before,self.freqs_after],
                                ["frequency_pre_eq","frequency_post_eq"]):
            create_and_save_csv(
                path=cfg.model_out_dir / label  / "structure" / f"{self.event_id}.csv",
                array=quantity,
                rewrite=rewrite
                )

        if cfg.output_response == "force_deformation":
            fd_path = cfg.model_out_dir / "force_deformation" / "structure" / f"{self.event_id}.csv"
            fd_path.parent.mkdir(parents=True, exist_ok=True)
            save_force_deformation(self.response_y, self.response_x, self.inputs["field"]["dt"], filename=fd_path)
        else:
            ss_path = cfg.model_out_dir / "strain_stress" / "structure" / f"{self.event_id}.csv"
            ss_path.parent.mkdir(parents=True, exist_ok=True)
            save_strain_stress(self.response_y, self.response_x, self.inputs["field"]["dt"], filename=ss_path)

        # FE model outputs, used as true outputs in system identification
        # Note, slice [1:] is because extra first timestep is recorded during analysis
        # Displacement outputs (inches)
        self.outputs["model"]["displacement"] = get_node_outputs(self.displ, nodes=cfg.channels_dofs["output"]["nodes"], dofs=cfg.channels_dofs["output"]["dofs"])[:, 1:]
        # Acceleration outputs (inches/second/second)
        self.outputs["model"]["acceleration"] = get_node_outputs(self.accel, nodes=cfg.channels_dofs["output"]["nodes"], dofs=cfg.channels_dofs["output"]["dofs"])[:, 1:]

        assert self.inputs["model"]["acceleration"].shape[1] == self.outputs["model"]["displacement"].shape[1], (
            "system identification training inputs and outputs have different length of time samples.")
        self.inputs["model"]["time"] = np.arange(self.nt) * self.inputs["model"]["dt"]

        if cfg.verbose >= 2:
            for qdict,qdict_name in zip([self.inputs,self.outputs],["inputs","outputs"]):
                print(qdict_name, "saved:")
                for source,quantities in qdict.items():
                    print(source, list(quantities.keys()))

        # Use create_and_save_csv to save csvs, with argument rewrite
        for location,quantities in zip(["ground","structure"],[self.inputs["model"],self.outputs["model"]]):
            for q_name,q in quantities.items():
                create_and_save_csv(
                    path = cfg.model_out_dir / q_name / location / f"{self.event_id}.csv",
                    array = q,
                    rewrite=rewrite
                )


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
    cfg = RunConfig.from_args(parse_data_args())

    # Print analysis configuration
    if cfg.verbose:
        print(f"structure={cfg.structure}")
        print(f"elastic={cfg.elastic}")

    # Load events
    events = load_events(cfg)

    # Perform model analysis and record responses
    for event_idx,event in enumerate(events):
        EventAnalysis(cfg, event, event_idx).run()
