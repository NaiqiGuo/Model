from Experimental_System_ID import get_inputs
import matplotlib.pyplot as plt
import quakeio
from pathlib import Path
import pickle

LOAD_EVENTS = False

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

if __name__ == "__main__":
    # TODO: Finish this script to plot all of the input data. Include:
    # legend: Channel 1, Channel 2
    # transparency so we can see both lines
    # put all plots on one figure
    # truncate each record to zoom in a bit

    input_channels = [1,3]
    for i in range(len(events)):
        inputs = get_inputs(i, events, input_channels)
        for i in inputs:
            plt.plot(i.T)
            plt.show()