import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    replot = True

    while replot:
    
        desired_structure = input("tell me what you want to plot: ")

        print(desired_structure)

        desired_source = input("tell me what you want to plot: ")

        print(desired_source)


        # When plot the data, allow adding of another timeseries
        # to the same axis. Maybe it needs to close the current axis
        # and reload it and/or reshow it.


    replot_string = input("plot another series?")
    if replot_string != "No":
        replot = False
