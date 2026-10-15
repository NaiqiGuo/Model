# Model

A suite of structures, their vibration responses to strong ground motion events, and analysis to investigate the relationships between ``ground truth'' damage state and estimated damage states obtained from inverse system identification.

## Getting Started

1. `get_data.py` : finite element model analysis and data extraction.
    - Choose an analysis configuration via command-line flags
        - `--structure frame` or `--structure bridge`
        - `--multisupport` (bridge only)
        - `--elastic` (otherwise inelastic)
        - `--field_only` to save measured field data without running the FE model
    - Loads a suite of events
    - For each event:
        - saves measured field inputs (ground) and outputs (structure)
        - performs FEM analysis and saves:
            - pre- and post- earthquake natural frequencies from FEM eigenvalue analysis
            - displacement response histories at select output nodes
            - strain/stress (or force/deformation) response histories at select output elements
            - model inputs and outputs used for system identification
2. `get_systems.py` : system identification on the data produced by `get_data.py`.
    - Choose an analysis configuration via command-line flags
        - `--structure frame` or `--structure bridge`
        - `--source field`, `--source elastic`, or `--source inelastic`
        - `--sid_method srim`
        - `--no_windowing` to disable training data truncation
    - For each event, saves:
        - timestep (dt)
        - time array
        - inputs array
        - outputs array
        - system matrices (A,B,C,D)
3. `get_prediction.py` : simulates predictions from the system realizations produced by
   `get_systems.py` and compares them against measured output.
    - Choose an analysis configuration via command-line flags
        - `--structure frame` or `--structure bridge`
        - `--source field`, `--source elastic`, or `--source inelastic`
        - `--output_quantity displacement` or `--output_quantity acceleration`
        - `--no_windowing`, `--no_signal_align`
    - For each event, saves:
        - simulated (predicted) output
        - true vs. predicted timeseries plots (PNG + interactive HTML)
        - normalized L2 prediction error
    - After all events, saves cross-event error heatmaps
4. `plot_inputs_outputs.py`: plot the inputs and outputs used for system ID. Primarily used for debugging.
5. `plot_series.py`: plot timeseries.
    1. Prompts the user for:
        1. structure
        2. event
        3. quantity
    2. Adds onto an axis:
        1. source
        2. location
    3. Save the plot if desired.

## Details on windowing and alignment throughout pipeline
1. `get_systems`: window inputs and outputs, create A,B,C,D with system ID. save inputs and outputs into System ID / structure / source / quantity / System ID Training Data.
2. `get_prediction`: take inputs from System ID Training Data. Use A,B,C,D to predict output from inputs. Window and align inputs, predicted outputs, and true outputs, save into System ID / structure / source / quantity / System ID Results / _processed. Use processed data for error computation and heatmap.


## Running

### Export and compare results

You can also run Build and the four comparisons from the VS Code **Inspect**
sidebar. Install `vscode-inspect/model-inspect-0.1.7.vsix` using Extensions →
**… → Install from VSIX…**, then select your Python interpreter and input folders.
See [the extension guide](vscode-inspect/README.md) for setup and usage.

Export all three categories and the active Python environment with one command:

```bash
python inspect.py build naiqi
```

The supplied prefix determines the output location. To save in a particular run,
use `python inspect.py build runs/your_run/naiqi` instead. Existing error CSV names
are preserved. Each structure/quantity/source folder additionally contains
`training/{dt,time,ground,structure}/<event>.csv` and
`systems/<event>/{A,B,C,D}.csv`. Environment records go in
`<prefix>_environment/packages.txt` and `environment.json`.

Compare one category per command, using the exact input directories you want:

```bash
python inspect.py compare errors chrystal_bridge runs/your_run/naiqi_bridge
python inspect.py compare realizations chrystal_bridge runs/your_run/naiqi_bridge
python inspect.py compare training chrystal_bridge runs/your_run/naiqi_bridge
python inspect.py compare errors chrystal_frame runs/your_run/naiqi_frame
python inspect.py compare realizations chrystal_frame runs/your_run/naiqi_frame
python inspect.py compare training chrystal_frame runs/your_run/naiqi_frame
python inspect.py compare environment reference_environment analysis_export_environment
```

Replace `your_run` with your actual directory name. No latest-run lookup is used.
Each command prints its result and saves one report beside its second input
folder: `compare_bridge_errors.txt`, `compare_bridge_realizations.txt`,
`compare_bridge_training.txt`, and the corresponding three frame files.
The standalone environment comparison writes `compare_environment.txt` beside
the second environment folder. It compares recorded Python version, platform
and package versions, ignoring paths and capture timestamps for match status.
These snapshots describe the export environment, not necessarily the original
computation environment; matching versions do not verify identical BLAS builds.
Rerunning the same category in the same output location updates that report.
Old `compare_bridge.txt` and `compare_frame.txt` files are left untouched.

`realizations` compares the existing `systems/` CSV exports; their directory
names do not change. Matrices are compared before prediction-time stabilization.
Different state coordinates can yield different matrices with equivalent
input/output behavior. Numeric comparisons check shapes, non-finite values, and
symmetric relative/absolute tolerances. Missing data is reported; a category
absent from both folders is not considered a successful match. Comparison results
are recorded in the report and do not determine the command's exit status.

Each report includes environment information when available. Package differences
are informational. Required
source data must exist for all configured events during export. Files are not
deleted, but reusing the same explicit export prefix updates same-name files.

`build` also collects heatmaps into `<name>_heatmaps/`; no separate command is needed.

The current `run_full_comparison.sh` runs data generation, system identification,
and prediction, saving logs in its timestamped run directory. Export and compare
with the commands above afterward. `Modeling/` and `System ID/` remain shared
working directories. All inspection logic is in `inspect.py`.

Get data for all four configurations (frame/bridge × inelastic/elastic):

```bash
for s in frame bridge; do for e in "" "--elastic"; do python get_data.py --structure "$s" $e; done; done
```

Run system ID for all structures and sources (frame/bridge × field/elastic/inelastic):

```bash
for s in frame bridge; do for src in field elastic inelastic; do python get_systems.py --structure "$s" --source "$src"; done; done
```

Run prediction for all structures and sources (frame/bridge × field/elastic/inelastic):

```bash
for s in frame bridge; do for src in field elastic inelastic; do python get_prediction.py --structure "$s" --source "$src" --annotate_plots; done; done
```

## Overall Directory Structure
[tree.nathanfriend](https://tree.nathanfriend.com/?s=(%27optjs!(%27fancyY~fullPath!false~trail_gSlashY~rootDotY)~w(%27w%27Model_g8framL4227.csvFKE-4B-dtFVFQtxtFKB-BOG9FE-4B-BO_G-B8bridgL4BFBW-BOG9FE-4B-BOBWWH8framN3A4EJ4K0F*QpklF*frequency%20IDJ4KK*heatmap.pngC3A4EJ4BF0-5OGX3A4K0-5C3A4EJ4BF0-5OO_GX3A4K0-5C3BFUB8bridgN3A4K0-5C3A4BF0-5OGX3A*K0F*K*BCJ*BOO_GJ*B8%27)~versj!%271%27)*%20%20-O*0UsZrealizatj%2F3JH%20Tra__g%20Data%2F*F*4*QcsvF*5**QpklF*K*B8W*9-displacementAVJB...C-acceleratjEstructureF-*GelasticHSZIDJ%2FFK*BFLeOfield-timeFV-Ne%2FOfieldXO8*Q*226.UH%20ResultsJ*VgroundW%5CnX%2F9Y!trueZystem%20_injionwsource!%01wj_ZYXWVUQONLKJHGFECBA985430-*)


```
.
├── Modeling/
│   ├── bridge/
│   │   ├── field/
│   │   │   ├── acceleration/
│   │   │   │   ├── ground/
│   │   │   │   │   ├── 1.csv
│   │   │   │   │   ├── 2.csv
│   │   │   │   │   └── ...
│   │   │   │   └── structure/
│   │   │   │       ├── 1.csv
│   │   │   │       └── ...
│   │   │   ├── displacement/
│   │   │   │   ├── ground/
│   │   │   │   │   ├── 1.txt
│   │   │   │   │   └── ...
│   │   │   │   └── ...
│   │   │   └── ...
│   │   ├── elastic/
│   │   │   ├── acceleration/
│   │   │   │   └── ground/
│   │   │   │       ├── 1.csv
│   │   │   │       └── ...
│   │   │   └── ...
│   │   └── inelastic/
│   │       └── ...
│   └── frame/
│       ├── field/
│       │   ├── acceleration/
│       │   │   ├── ground/
│       │   │   │   ├── 226.csv
│       │   │   │   └── ...
│       │   │   └── ...
│       │   └── ...
│       ├── elastic/
│       │   ├── acceleration/
│       │   │   └── ground/
│       │   │       ├── 226.csv
│       │   │       └── ...
│       │   └── ...
│       └── ...
└── System ID/
    ├── bridge/
    │   ├── acceleration/
    │   │   ├── field/
    │   │   │   ├── System ID Training Data/  
    │   │   │   │   ├── ground/
    │   │   │   │   │   └── 1.csv
    │   │   │   │   |   └── ...
    │   │   │   │   ├── structure/
    │   │   │   │   |   ├── 1.csv
    │   │   │   │   |   └── ...
    │   │   │   │   ├── dt/
    │   │   │   │   |   ├── 1.csv
    │   │   │   │   |   └── ...
    │   │   │   │   └── time/
    │   │   │   │       └── ...
    │   │   │   └── System ID Results/
    │   │   │       ├── system realization/
    │   │   │       │   ├── 1.pkl
    │   │   │       │   └── ...
    │   │   │       ├── prediction plots/
    │   │   │       │   ├── 1.pkl
    │   │   │       │   └── ...
    │   │   │       ├── inputs_processed
    │   │   │       │   ├── 1.csv
    │   │   │       │   └── ...
    │   │   │       ├── outputs_pred_processed
    │   │   │       │   ├── 1.csv
    │   │   │       │   └── ...
    │   │   │       ├── outputs_true_processed
    │   │   │       │   ├── 1.csv
    │   │   │       │   └── ...
    │   │   │       ├── time_processed
    │   │   │       │   ├── 1.csv
    │   │   │       │   └── ...
    │   │   │       ├── dt
    │   │   │       │   ├── 1.csv
    │   │   │       │   └── ...
    │   │   │       ├── errors
    │   │   │       │   ├── 1.csv
    │   │   │       │   └── ...
    │   │   │       ├── frequency ID (not yet implemented)/
    │   │   │       │   ├── 1.csv
    │   │   │       │   └── ...
    │   │   │       ├── heatmap.png
    │   │   │       └── heatmap_square.png
    │   │   ├── elastic/
    │   │   │   ├── System ID Training Data/
    │   │   │   └── System ID Results/
    │   │   └── inelastic/
    │   │       ├── System ID Training Data/
    │   │       └── System ID Results/
    │   │
    │   └── displacement/
    │       ├── field/
    │       │   ├── System ID Training Data/
    │       │   └── System ID Results/
    │       ├── elastic/
    │       │   ├── System ID Training Data/
    │       │   └── System ID Results/
    │       └── inelastic/
    │           ├── System ID Training Data/
    │           └── System ID Results/
    │
    └── frame/
        ├── acceleration/
        │   ├── field/
        │   │   ├── System ID Training Data/  
        │   │   │   ├── ground/
        │   │   │   │   └── 226.csv
        │   │   │   |   └── ...
        │   │   │   ├── structure/
        │   │   │   |   ├── 226.csv
        │   │   │   |   └── ...
        │   │   │   └── ...
        │   │   └── System ID Results/
        │   │       └── ...
        │   ├── elastic/
        │   │   ├── System ID Training Data/  
        │   │   └── System ID Results/
        │   └── inelastic/
        │       └── ...
        └── displacement/
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




