"""
Get system realization from model event responses.
Performs system identification.
"""

from pathlib import Path
import numpy as np
import pickle
from mdof import sysid
from mdof.utilities.config import Config
from mdof.utilities.testing import intensity_bounds, truncate_by_bounds
import pickle
from mdof.utilities.testing import intensity_bounds, truncate_by_bounds

# Analysis configuration
SID_METHOD = 'srim'
STRUCTURE = "bridge" # "frame", "bridge"
MULTISUPPORT = False
ELASTIC = False

# Verbosity
# False means print nothing;
# True or 1 means print progress messages only;
# 2 means print progress and validation messages
VERBOSE = 1
SYSTEM_ID_VERBOSE = 2
WINDOWED = True

# Main output directory
BASE_DIR = Path("Modeling")
MODEL_OUT_DIR = BASE_DIR / STRUCTURE / ("elastic" if ELASTIC else "inelastic")
FIELD_OUT_DIR = BASE_DIR / STRUCTURE / "field"

WINDOWED_INTENSITY = True


if __name__ == "__main__":
    # Print analysis configuration
    if VERBOSE:
        print(f"{STRUCTURE=}")
        print(f"{ELASTIC=}")

    # Retreive data
    elastic_name = "elastic" if ELASTIC else "inelastic"
    outputs = {elastic_name: {}, "field": {}}
    quantities = ["displacement", "acceleration"]

    event_files = sorted((FIELD_OUT_DIR / "acceleration" / "ground").glob("*.csv"))
    event_ids = [event.stem for event in event_files]
    failed_events = []
    for event_id in event_ids:
        if VERBOSE:
            print(f"\nSystem ID for Event {event_id}")

        try:
            for source in outputs.keys():
                 for q in quantities:
                    outputs[source][q] = np.loadtxt(
                        (MODEL_OUT_DIR if source == elastic_name else FIELD_OUT_DIR) / q / "structure" / f"{event_id}.csv",
                    )

        except FileNotFoundError:
            if VERBOSE:
                print(f"No data for event {event_id}; skipping")
            continue

        # Perform system identification and save systems
        n = 4
        n = 4
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
            j           = 4400,
            verbose     = SYSTEM_ID_VERBOSE,
        )

        for source in [elastic_name, "field"]:
             for quantity in quantities:
                if WINDOWED_INTENSITY:
                    bounds = intensity_bounds(outputs[source][quantity][0], lb=0.01, ub=0.99)
                    inputs_truncated = truncate_by_bounds(inputs, bounds)
                    outputs[source][quantity] = truncate_by_bounds(outputs[source][quantity], bounds)
                try:
                    input_dir = FIELD_OUT_DIR if source == "field" else MODEL_OUT_DIR
                    sid_inputs = np.atleast_2d(np.loadtxt(
                        input_dir / "acceleration" / "ground" / f"{event_id}.csv",
                    ))
                    sid_outputs = outputs[source][quantity]
                    if WINDOWED:
                        sig = np.asarray(sid_outputs[0]).reshape(-1).copy()
                        bounds = intensity_bounds(sig, lb=0.01, ub=0.99)
                        sid_inputs = truncate_by_bounds(sid_inputs, bounds)
                        sid_outputs = truncate_by_bounds(sid_outputs, bounds)

                    system = sysid(sid_inputs, sid_outputs, method=SID_METHOD, **options)
                except Exception as e:
                    failed_events.append((event_id, source, quantity, e))
                    if VERBOSE:
                        print(f"\n>>>> System ID for event {event_id} FAILED for {source},{quantity}")
                        print(f">>>> Error: {e}")
                    continue
            
                A,B,C,D, *rest = system 
                system  = (A,B,C,D)

                system_path = (Path('System ID') /   
                                STRUCTURE /
                                source / 
                                quantity /
                                'System ID Results' /
                                'system realization' /
                                f"{event_id}.pkl"
                                )
                system_path.parent.mkdir(parents=True, exist_ok=True)
                with open(system_path, "wb") as f:
                    pickle.dump(system, f)
    
    if VERBOSE and len(failed_events) > 0:
        print(f"Failed events: {failed_events}")
