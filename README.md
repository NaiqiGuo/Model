# Model

A suite of structures, their vibration responses to strong ground motion events, and analysis to investigate the relationships between ``ground truth'' damage state and estimated damage states obtained from inverse system identification.

## Getting Started

1. `get_systems.py` : finite element model and its system identification.
    - Choose an analysis configuration
        - "frame" or "bridge"
        - `MULTISUPPORT = True` or `False`
        - `ELASTIC = True` or `False`
    - Loads a suite of events
    - For each event:
        - performs FEM analysis and saves:
            - pre- and post- earthquake natural frequencies from FEM eigenvalue analysis
            - displacement response histories at select output nodes
            - strain/stress response histories at select output elements
        - performs system identification and saves:
            - timestep (dt)
            - time array
            - inputs array
            - outputs array
            - system matrices (A,B,C,D)






