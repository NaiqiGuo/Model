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
2. `plot_inputs_outputs.py`: plot the inputs and outputs used for system ID. Primarily used for debugging.
3.  `plot_series.py`: plot timeseries.
    1. Prompts the user for:
        1. structure
        2. event
        3. quantity
    2. Adds onto an axis:
        1. source
        2. location
    3. Save the plot if desired.


## Overall Directory Structure
[tree.nathanfriend](https://tree.nathanfriend.com/?s=(%27options!(%27fancyC~fullPath!false~trailingSlashC~rootDotC)~G(%27G%27ModelingJ8AE227.E5B0dtHAtxt4*545KL30LinF59*bridg80*59KL30L599System%20IDJe%27)~version!%271%27)*%20%20-9**0AE5K3FdisplacementB4K*5...8e-fieldKtimeH9%5CnA4*226.B4structureC!trueEcsv4*FelasticKGsource!H4groundJ9*framK-*L5-%01LKJHGFECBA985430-*)

```
.
└── Modeling/
    ├── frame/
    │   ├── field/
    │   │   ├── time/
    │   │   │   ├── ground/
    │   │   │   │   ├── 226.csv
    │   │   │   │   ├── 227.csv
    │   │   │   │   └── ...
    │   │   │   └── structure/
    │   │   │       ├── 226.csv
    │   │   │       └── ...
    │   │   ├── dt/
    │   │   │   ├── ground/
    │   │   │   │   ├── 226.txt
    │   │   │   │   └── ...
    │   │   │   └── ...
    │   │   └── ...
    │   ├── elastic/
    │   │   ├── displacement/
    │   │   │   └── structure/
    │   │   │       ├── 226.csv
    │   │   │       └── ...
    │   │   └── ...
    │   └── inelastic/
    │       └── ...
    └── bridge/
        ├── field/
        │   ├── time/
        │   │   ├── ground/
        │   │   │   ├── 226.csv
        │   │   │   └── ...
        │   │   └── ...
        │   └── ...
        ├── elastic/
        │   ├── displacement/
        │   │   └── structure/
        │   │       ├── 226.csv
        │   │       └── ...
        │   └── ...
        └── ...
```

## Modeling Directory Structure

Level | Name      | Quantities
------|-----------|------------
1     | Structure | frame, bridge
2     | Source    | field, elastic, inelastic
3     | Quantity  | time, dt, displacement, acceleration, stress, strain, frequency pre-eq, frequency post-eq
4     | Location  | ground (input), structure (output)
5     | Event     | 1, 2, 3, ... or 226, 227, 228, ... etc.

See below for list of quantities and locations available in each Source's subdirectory.

Source     | Quantities | Locations
-----------|------------|-----------
field      | time, dt, displacement, acceleration | ground (input), structure (output)
elastic    | displacement, acceleration, stress, strain, frequency pre-eq, frequency post-eq | structure (output)
inelastic  | displacement, acceleration, stress, strain, frequency pre-eq, frequency post-eq | structure (output)

## System ID Directory Structure

Needed for system ID
1. true input (truncated and aligned)
2. true output displacement (truncated and aligned)
3. true output acceleration (truncated and aligned)
4. time array (truncated and aligned)

Results of system ID
1. system (A,B,C,D)
2. frequency ID
3. mode shapes
4. predicted output: displacement
5. predicted output: acceleration
6. prediction error: displacement
7. prediction error: acceleration
8. heatmap (encompasses all events)

Level | Name      | Quantities
------|-----------|------------
1     | Structure | frame, bridge
2     | Source    | field, elastic, inelastic
3     | Quantity  | displacement, acceleration
4a     | System ID Training Data | time, dt, ground (true input), structure (true output), structure (predicted output)
4b     | System ID Results  | system realization, frequency ID, mode shapes, heatmap, prediction, prediction error
5     | Event     | 1, 2, 3, ... or 226, 227, 228, ... etc.

## Environment

#### Method 1
1. Install numba: `conda install numba`
2. Install requirements: `pip install -r requirements.txt`

#### Method 2
1. Set up a xara-friendly environment: https://xara.so/user/guides/compile.html
2. Install requirements: `pip install -r requirements.txt`






