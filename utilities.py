import numpy as np
import csv
import os
from pathlib import Path
from mdof.utilities.config import extract_channels

def write_freq_csv(event_id,
                   freqs_before,
                   freqs_after,
                   freq_csv_path="natural_frequencies.csv"):
    
    n_modes = len(freqs_before)
    
    file_exists = os.path.exists(freq_csv_path)

    with open(freq_csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            header = ["event_id"]
            header += [f"f{i+1}_before_Hz" for i in range(n_modes)] #Hz
            header += [f"f{i+1}_after_Hz" for i in range(n_modes)]
            writer.writerow(header)
        row = [event_id] + list(freqs_before) + list(freqs_after)
        writer.writerow(row)

def get_node_outputs(outputs, nodes, dofs):
    """
    Get an array of outputs for the given nodes and dofs. Each
    output must be separately and explicity defined. The length
    of `nodes` must equal the length of `dofs` (see param
    definitions below)

    :param outputs: dictionary of node output time series, e.g.
                    displacements or accelerations:
                    { node_id: [ [u1,u2,u3,u4,u5,u6], ... ] }

    :param nodes: list of output nodes, e.g. [5,5,10,10,15,15]
    
    :param dofs:  list of output dofs, e.g. [1,2,1,2,1,2]
                  where
                  1 = X translation
                  2 = Y translation
                  3 = Z translation
                  4 = X rotation
                  5 = Y rotation
                  6 = Z rotation
                  
    Returns an ndarray, shape=(len(dofs), nt). Row order:
      [Node1 DOF1, Node2 DOF2, Node3 DOF3, ...]
      
    """
    
    assert len(nodes)==len(dofs), "Length of `nodes` must equal length of `dofs`."

    rows = []
    for node, dof in zip(nodes, dofs):
        arr = np.asarray(outputs[node])   # shape: (nt+1, 6)
        rows.append(arr[:, dof - 1])      # dof is 1-based

    return np.vstack(rows)

def get_measurements(i, events, channels, scale=1, response="accel"):
    event = events[i]
    channel_data, dt = extract_channels(event, channels, response=response)
    channel_data = scale * channel_data
    measurements = {ch: channel_data[idx] for idx, ch in enumerate(channels)}
    return measurements, dt

def create_and_save_csv(path:Path, array, rewrite=False):
    if not rewrite and path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.atleast_1d(array))