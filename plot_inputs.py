from Experimental_System_ID import get_inputs
import matplotlib.pyplot as plt
import quakeio
from pathlib import Path

events = sorted([
    print(file) or quakeio.read(file, exclusions=["*filter*"])
    for file in list(Path(f"../uploads/CE89324/").glob("????????*.[zZ][iI][pP]"))
], key=lambda event: abs(event["peak_accel"]))

if __name__ == "__main__":
    for i in range(len(events)):
        inputs = get_inputs(i)
        plt.plot(inputs)
        plt.show()