"""
(Re)create the FROZEN golden fixtures in tests/golden/ from Modeling/.

The golden files are the ground truth. They are not meant to be regenerated
routinely. This file should only be run when the models have been 
deliberately and correctly changed, the diff have been reviewed, and the
new results have been blessed as the new truth.
The tests compare against these; if the tests fail, treat it as "the outputs
changed." Investigate before refreshing.

Each record is truncated to the first TRUNC_SECONDS to keep the repo small.

    python tests/refresh_golden.py
"""
from pathlib import Path
import numpy as np

TRUNC_SECONDS = 15.0
ROOT = Path(__file__).resolve().parent.parent
MODELING = ROOT / "Modeling"
GOLDEN = Path(__file__).resolve().parent / "golden"

# quantity -> kind:
#   "series" = 2D (channels x nt); truncate columns to the first 15 s
#   "time"   = 1D time vector;     truncate to the first 15 s
#   "scalar" = dt / frequency vectors; copied verbatim (already tiny)
KIND = {
    "acceleration": "series", "displacement": "series",
    "time": "time",
    "dt": "scalar", "frequency_pre_eq": "scalar", "frequency_post_eq": "scalar",
}

FIELD_FILES = [
    ("acceleration", "ground"), ("displacement", "ground"),
    ("displacement", "structure"), ("dt", "ground"), ("time", "ground"),
]
MODEL_FILES = [
    ("displacement", "structure"), ("acceleration", "structure"),
    ("acceleration", "ground"),  # model input, used to drive the opt-in solve test
    ("frequency_pre_eq", "structure"), ("frequency_post_eq", "structure"),
    ("dt", "ground"), ("time", "ground"),
]

# (structure, case, event, files)
CASES = [
    ("frame",  "field",     "226", FIELD_FILES),
    ("frame",  "elastic",   "226", MODEL_FILES),
    ("frame",  "inelastic", "226", MODEL_FILES),
    ("bridge", "field",     "1",   FIELD_FILES),
    ("bridge", "elastic",   "1",   MODEL_FILES),
    ("bridge", "inelastic", "1",   MODEL_FILES),
]


def dt_of(structure, case, event):
    p = MODELING / structure / case / "dt" / "ground" / f"{event}.csv"
    return float(p.read_text().strip())


def main():
    for structure, case, event, files in CASES:
        n = int(round(TRUNC_SECONDS / dt_of(structure, case, event)))
        for quantity, location in files:
            rel = f"{structure}/{case}/{quantity}/{location}/{event}.csv"
            src, dst = MODELING / rel, GOLDEN / rel
            if not src.exists():
                print(f"skip (missing) {rel}")
                continue
            kind = KIND[quantity]
            if kind == "scalar":
                arr = np.loadtxt(src, ndmin=1)
            elif kind == "time":
                arr = np.loadtxt(src, ndmin=1)[:n]
            else:
                arr = np.loadtxt(src, ndmin=2)[:, :n]
            dst.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(dst, arr, fmt="%.18e")
            print(f"wrote {rel}  {np.loadtxt(dst, ndmin=2).shape}")


if __name__ == "__main__":
    main()
