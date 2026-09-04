"""
Get system realization from model event responses.
Performs system identification.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pickle
from mdof import sysid
from mdof.utilities.config import Config
from mdof.utilities.testing import intensity_bounds, truncate_by_bounds
from utilities import create_and_save_csv


def get_event_ids(in_dir: Path):
    """
    Get event IDs from the input directory.

    :param in_dir: Path to the input directory containing event data.
    :return: List of event IDs (str).
    """
    events = sorted((in_dir / "acceleration" / "ground").glob("[0-9]*.csv"), key=lambda event_path: int(event_path.stem))
    event_ids = [event.stem for event in events]
    return event_ids


def window_data(target, reference):
    """
    Window a target signal based on the reference signal's intensity bounds.

    :param target: Signal to be windowed.
    :param reference: The reference signal to determine bounds.
    :return: Windowed signal.
    """
    bounds = intensity_bounds(reference, lb=0.01, ub=0.99)
    windowed_signal = truncate_by_bounds(target, bounds)
    return windowed_signal


@dataclass(frozen=True)
class RunConfig:
    """Run-constant analysis configuration, built from CLI args."""
    structure: str
    source: str
    sid_method: str
    sid_options: Config
    windowed: bool
    verbose: int
    in_modeling_dir: Path
    out_sid_dir: Path

    @classmethod
    def from_args(cls, args) -> "RunConfig":
        structure = args.structure
        source = args.source

        in_modeling_dir = Path("Modeling") / structure / source
        out_sid_dir = Path("System ID") / structure / source
        out_sid_dir.mkdir(parents=True, exist_ok=True)

        # System identification parameters
        n = 4
        sid_options = Config(
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
            j           = 4400,
            verbose     = args.sid_verbose,
        )

        return cls(
            structure=structure,
            source=args.source,
            sid_method=args.sid_method,
            sid_options=sid_options,
            windowed=args.windowed,
            verbose=args.verbose,
            in_modeling_dir=in_modeling_dir,
            out_sid_dir=out_sid_dir
        )


class Train:
    """
    Perform a single system identification realization.
    Reads dt, time, inputs, and outputs;
    saves training data,
    runs system ID,
    and saves trained system realization.

    One instance per event, per quantity (displacement, acceleration).
    """

    def __init__(self, cfg: RunConfig, event_id: int, quantity: str):
        self.cfg = cfg
        self.event_id = event_id
        self.quantity = quantity
        self.sid_inputs = None
        self.sid_outputs = None
        self.dt = None
        self.sid_time = None
        self.realization = None

    def process_raw_data(self):
        """
        Process dt, time, inputs, and outputs.
        """
        cfg = self.cfg
        event_id = self.event_id

        # Read dt, time, inputs, and outputs
        in_dir = cfg.in_modeling_dir
        self.sid_dt = np.loadtxt(in_dir / "dt" / "ground" / f"{event_id}.csv")
        self.sid_time = np.loadtxt(in_dir / "time" / "ground" / f"{event_id}.csv")
        self.sid_inputs = np.atleast_2d(np.loadtxt(
            in_dir / self.quantity / "ground" / f"{event_id}.csv",
        ))
        self.sid_outputs = np.atleast_2d(np.loadtxt(
            in_dir / self.quantity / "structure" / f"{event_id}.csv",
        ))

        # Window data if specified
        if cfg.windowed:
            reference_signal = self.sid_outputs[0].copy()
            self.sid_time = window_data(self.sid_time, reference_signal)
            self.sid_inputs = window_data(self.sid_inputs, reference_signal)
            self.sid_outputs = window_data(self.sid_outputs, reference_signal)

    def save_training_data(self):
        """
        Save training data to the output directory.
        """
        cfg = self.cfg
        event_id = self.event_id
        quantity = self.quantity
        if any(x is None for x in [self.sid_inputs, self.sid_outputs, self.sid_time, self.sid_dt]):
            raise ValueError("Training data not processed. Call `process_raw_data()` before fitting.")
        create_and_save_csv(cfg.out_sid_dir / "dt"        / quantity / "System ID Training Data" / f"{event_id}.csv", self.sid_dt, rewrite=True)
        create_and_save_csv(cfg.out_sid_dir / "time"      / quantity / "System ID Training Data" / f"{event_id}.csv", self.sid_time, rewrite=True)
        create_and_save_csv(cfg.out_sid_dir / "ground"    / quantity / "System ID Training Data" / f"{event_id}.csv", self.sid_inputs, rewrite=True)
        create_and_save_csv(cfg.out_sid_dir / "structure" / quantity / "System ID Training Data" / f"{event_id}.csv", self.sid_outputs, rewrite=True)

    def fit(self):
        """
        Fit realization to data.
        """
        if any(x is None for x in [self.sid_inputs, self.sid_outputs, self.sid_time, self.sid_dt]):
            raise ValueError("Training data not processed. Call `process_raw_data()` before fitting.")
        cfg = self.cfg
        self.realization = sysid(self.sid_inputs, self.sid_outputs, method=cfg.sid_method, **cfg.sid_options)

    def save_realization(self):
        """Save trained system realization"""
        if self.realization is None:
            raise ValueError("Realization has not been fitted yet. Call `fit()` before saving.")
        A,B,C,D, *rest = self.realization
        system  = (A,B,C,D)
        system_path = (self.cfg.out_sid_dir / self.quantity / "System ID Results" / 'system realization' / f"{self.event_id}.pkl")
        system_path.parent.mkdir(parents=True, exist_ok=True)
        with open(system_path, "wb") as f:
            pickle.dump(system, f)

    def run(self):
        """
        Run full training pipeline for single event/quantity.
        """
        self.process_raw_data()
        self.save_training_data()
        self.fit()
        self.save_realization()


def parse_data_args():
    parser = argparse.ArgumentParser(description="Train system identification models from input and output data.")
    parser.add_argument("--structure", type=str, default="bridge", choices=["frame", "bridge"], help="Structure type: 'frame' or 'bridge'.")
    parser.add_argument("--source", type=str, default="field", choices=["field", "elastic", "inelastic"], help="Source of data: 'field', 'elastic', or 'inelastic'.")
    parser.add_argument("--sid_method", type=str, default="srim", choices=["srim"], help="System ID method.")
    parser.add_argument("--no_windowing", action="store_false", dest="windowed", help="Disable training data truncation.")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level: 0 (silent), 1 (progress), 2 (progress + validation).")
    parser.add_argument("--sid_verbose", type=int, default=1, help="System ID verbosity level: 0 (silent), 1 (progress), 2 (progress + validation).")
    return parser.parse_args()


if __name__ == "__main__":
    # Read custom analysis configuration parameters from command line arguments
    cfg = RunConfig.from_args(parse_data_args())

    # Print analysis configuration
    if cfg.verbose:
        print(f"structure={cfg.structure}")
        print(f"source={cfg.source}")

    # Perform system ID and record both training data and trained system realization
    event_ids = get_event_ids(cfg.in_modeling_dir)
    failed_events = []
    for event_id in event_ids:
        for quantity in ["displacement", "acceleration"]:
            if cfg.verbose:
                print(f"\nSystem ID for Event {event_id}, {quantity}")
            try:
                Train(cfg, event_id, quantity).run()
            except Exception as e:
                failed_events.append((event_id, cfg.source, quantity, e))
                if cfg.verbose:
                    print(f">>>> System ID for event {event_id} FAILED for {cfg.source},{quantity}")
                    print(f">>>> Error: {e}")
                continue

    if cfg.verbose and len(failed_events) > 0:
        print(f"Failed events: {failed_events}")
