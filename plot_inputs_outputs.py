from Experimental_System_ID import get_inputs, get_outputs, analyze, create_model
import matplotlib.pyplot as plt
import quakeio
from pathlib import Path
import pickle
import numpy as np
from mdof.utilities.testing import intensity_bounds, truncate_by_bounds
import os

LOAD_EVENTS = False
windowed_plot = True

save_dir = "plots"
os.makedirs(save_dir, exist_ok=True)

if __name__ == "__main__":

    if LOAD_EVENTS:
        events = sorted([
            print(file) or quakeio.read(file, exclusions=["*filter*"])
            for file in list(Path(f"/../uploads/CE89324/").glob("????????*.[zZ][iI][pP]"))
        ], key=lambda event: abs(event["peak_accel"]))
        with open("events.pkl","wb") as f:
            pickle.dump(events,f)
    else:
        with open("events.pkl","rb") as f:
            events = pickle.load(f)

    input_channels = [1,3]
    input_labels = ['Channel 1', 'Channel 3']
    output_labels = ['1X', '2X', '3X', '1Y', '2Y', '3Y']

    num_events = min(16, len(events))
    for i in range(num_events):
        print(f"Event {i+1}")
        inputs, dt = get_inputs(i, events, input_channels, scale=2.54)
        nt = inputs.shape[1]
        model = create_model(column="forceBeamColumn",
                             girder="forceBeamColumn",
                             inputx=inputs[0],
                             inputy=inputs[1],
                             dt=dt)
        disp = analyze(model, output_nodes=[9, 14, 19], nt=nt, dt=dt)
        outputs = get_outputs(disp)   
        outputs = outputs[:, 1:]

        t_in = np.arange(inputs.shape[1]) * dt
        t_out = np.arange(outputs.shape[1]) * dt

        if windowed_plot:
            bounds = intensity_bounds(outputs[0], lb=0.001, ub=0.999)
            inputs_trunc = truncate_by_bounds(inputs, bounds)
            t_in_trunc = t_in[bounds[0]:bounds[1]]
            outputs_trunc = truncate_by_bounds(outputs, bounds)
            t_out_trunc = t_out[bounds[0]:bounds[1]]
        else:
            inputs_trunc = inputs
            t_in_trunc = t_in
            outputs_trunc = outputs
            t_out_trunc = t_out

        # --------- input ---------
        plt.figure(figsize=(8,3))
        for ch in range(inputs_trunc.shape[0]):
            plt.plot(t_in_trunc, inputs_trunc[ch], alpha=0.7, label=input_labels[ch])
        plt.title(f"Event {i+1} Inputs")
        plt.xlabel("Time (s)")
        plt.ylabel("Input Accel")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_dir}/event_{i+1}_inputs.png")
        # plt.show()
        plt.close()

        # --------- output ---------
        plt.figure(figsize=(8,4))
        for ch in range(outputs_trunc.shape[0]):
            plt.plot(t_out_trunc, outputs_trunc[ch], alpha=0.7, label=output_labels[ch])
        plt.title(f"Event {i+1} Outputs")
        plt.xlabel("Time (s)")
        plt.ylabel("Output")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_dir}/event_{i+1}_outputs.png")
        # plt.show()
        plt.close()