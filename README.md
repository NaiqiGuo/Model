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
[tree.nathanfriend](https://tree.nathanfriend.com/?s=(%27optjs!(%27fancyY~fullPath!false~trail_gSlashY~rootDotY)~w(%27w%27Model_g8framL4227.csvFKE-4B-dtFVFQtxtFKB-BOG9FE-4B-BO_G-B8bridgL4BFBW-BOG9FE-4B-BOBWWH8framN3A4EJ4K0F*QpklF*frequency%20IDJ4KK*heatmap.pngC3A4EJ4BF0-5OGX3A4K0-5C3A4EJ4BF0-5OO_GX3A4K0-5C3BFUB8bridgN3A4K0-5C3A4BF0-5OGX3A*K0F*K*BCJ*BOO_GJ*B8%27)~versj!%271%27)*%20%20-O*0UsZrealizatj%2F3JH%20Tra__g%20Data%2F*F*4*QcsvF*5**QpklF*K*B8W*9-displacementAVJB...C-acceleratjEstructureF-*GelasticHSZIDJ%2FFK*BFLeOfield-timeFV-Ne%2FOfieldXO8*Q*226.UH%20ResultsJ*VgroundW%5CnX%2F9Y!trueZystem%20_injionwsource!%01wj_ZYXWVUQONLKJHGFECBA985430-*)


```
.
├── Modeling/
│   ├── frame/
│   │   ├── field/
│   │   │   ├── time/
│   │   │   │   ├── ground/
│   │   │   │   │   ├── 226.csv
│   │   │   │   │   ├── 227.csv
│   │   │   │   │   └── ...
│   │   │   │   └── structure/
│   │   │   │       ├── 226.csv
│   │   │   │       └── ...
│   │   │   ├── dt/
│   │   │   │   ├── ground/
│   │   │   │   │   ├── 226.txt
│   │   │   │   │   └── ...
│   │   │   │   └── ...
│   │   │   └── ...
│   │   ├── elastic/
│   │   │   ├── displacement/
│   │   │   │   └── structure/
│   │   │   │       ├── 226.csv
│   │   │   │       └── ...
│   │   │   └── ...
│   │   └── inelastic/
│   │       └── ...
│   └── bridge/
│       ├── field/
│       │   ├── time/
│       │   │   ├── ground/
│       │   │   │   ├── 226.csv
│       │   │   │   └── ...
│       │   │   └── ...
│       │   └── ...
│       ├── elastic/
│       │   ├── displacement/
│       │   │   └── structure/
│       │   │       ├── 226.csv
│       │   │       └── ...
│       │   └── ...
│       └── ...
└── System ID/
    ├── frame/
    │   ├── field/
    │   │   ├── displacement/
    │   │   │   ├── System ID Training Data/  
    │   │   │   │   ├── ground/
    │   │   │   │   │   └── 226.csv
    │   │   │   │   └── structure/
    │   │   │   │       ├── 226.csv
    │   │   │   │       └── ...
    │   │   │   └── System ID Results/
    │   │   │       ├── system realization/
    │   │   │       │   └── 226.pkl
    │   │   │       ├── frequency ID/
    │   │   │       │   ├── 226.csv
    │   │   │       │   └── ...
    │   │   │       ├── ...
    │   │   │       └── heatmap.png
    │   │   └── acceleration/
    │   │       ├── System ID Training Data/  
    │   │       │   ├── ground/
    │   │       │   │   └── 226.csv
    │   │       │   ├── structure/
    │   │       │   │   └── 226.csv
    │   │       │   └── ...
    │   │       └── System ID Results/
    │   │           ├── system realization/
    │   │           │   ├── 226.pkl
    │   │           │   └── ...
    │   │           └── ...
    │   ├── elastic/
    │   │   ├── displacement/
    │   │   │   ├── System ID Training Data/  
    │   │   │   │   └── ground/
    │   │   │   │       ├── 226.csv
    │   │   │   │       └── ...
    │   │   │   └── System ID Results/
    │   │   │       ├── system realization/
    │   │   │       │   ├── 226.pkl
    │   │   │       │   └── ...
    │   │   │       └── ...
    │   │   └── acceleration/
    │   │       ├── System ID Training Data/  
    │   │       │   ├── ground/
    │   │       │   │   └── 226.csv
    │   │       │   ├── structure/
    │   │       │   │   └── 226.csv
    │   │       │   └── ...
    │   │       └── System ID Results/
    │   │           ├── system realization/
    │   │           │   ├── 226.pkl
    │   │           │   └── ...
    │   │           └── ...
    │   └── inelastic/
    │       ├── displacement/
    │       │   ├── System ID Training Data/  
    │       │   │   └── ground/
    │       │   │       ├── 226.csv
    │       │   │       └── ...
    │       │   └── System ID Results/
    │       │       ├── system realization/
    │       │       │   ├── 226.pkl
    │       │       │   └── ...
    │       │       └── ...
    │       └── acceleration/
    │           ├── System ID Training Data/  
    │           │   └── ...
    │           └── System ID Results/
    │               └── ...
    └── bridge/
        ├── field/
        │   ├── displacement/
        │   │   ├── System ID Training Data/  
        │   │   │   └── ground/
        │   │   │       ├── 226.csv
        │   │   │       └── ...
        │   │   └── System ID Results/
        │   │       ├── system realization/
        │   │       │   ├── 226.pkl
        │   │       │   └── ...
        │   │       └── ...
        │   └── acceleration/
        │       ├── System ID Training Data/  
        │       │   ├── ground/
        │       │   │   └── 226.csv
        │       │   └── ...
        │       └── System ID Results/
        │           ├── system realization/
        │           │   ├── 226.pkl
        │           │   └── ...
        │           └── ...
        ├── elastic/
        │   ├── displacement/
        │   │   ├── System ID Training Data/  
        │   │   │   └── ground/
        │   │   │       └── ...
        │   │   └── System ID Results/
        │   │       ├── system realization/
        │   │       │   └── ...
        │   │       └── ...
        │   └── acceleration/
        │       └── ...
        └── inelastic/
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

Level | Name      | Quantities
------|-----------|------------
1     | Structure | frame, bridge
2     | Source    | field, elastic, inelastic
3     | Output Quantity | displacement, acceleration
4a*   | System ID Training Data | ground acceleration (true input), structure response (true output), time, dt
4b    | System ID Results  | system realization (A,B,C,D), frequency ID, mode shapes, prediction, prediction error, heatmap (encompasses all events)
5     | Event     | 1, 2, 3, ... or 226, 227, 228, ... etc.

*All time series are truncated and aligned according to true output.


## Environment

#### Method 1
1. Install numba: `conda install numba`
2. Install requirements: `pip install -r requirements.txt`

#### Method 2
1. Set up a xara-friendly environment: https://xara.so/user/guides/compile.html
2. Install requirements: `pip install -r requirements.txt`






