import numpy as np
import glob
import xara.units.iks as units

def _parse_time_to_seconds(t: str) -> float:
    t = t.strip()
    # case 1: already numeric seconds
    try:
        return float(t)
    except ValueError:
        pass
    # case 2: timestamp like 0015:12:27.680000 or 15:12:27.680000
    # interpret as HH:MM:SS.ssssss
    parts = t.split(":")
    if len(parts) != 3:
        raise ValueError(f"Unrecognized Time format: {t}")
    hh = int(parts[0])
    mm = int(parts[1])
    ss = float(parts[2])
    return hh * 3600.0 + mm * 60.0 + ss


def _norm(u: str) -> str:
    return (u or "").strip().lower().replace(" ", "")

def scale_249_units(sensor_units, standard="iks"):
    if (standard or "").lower() != "iks":
        raise ValueError(f"Only standard='iks' supported, got {standard!r}")
    scales = []
    for u in sensor_units:
        uu = _norm(u)
        if uu in ("g's", "gs", "g"):          #  "g's"
            scales.append(units.gravity)       # g --> in/s^2
        elif uu in ("in/s", "ips"):
            scales.append(1.0)                # already in/s
        elif uu in ("inches", "inch", "in"):
            scales.append(1.0)                # already inch
        elif uu in ("mv",):
            scales.append(1.0)                # need check
        else:
            raise ValueError(f"Unhandled unit in 249 file: {u!r} (normalized {uu!r})")
        
    return np.asarray(scales, dtype=float)

def get_249_data(path):
    with open(path) as f:
        lines = f.readlines()
    header_idx = next(i for i,l in enumerate(lines) if l.startswith("Time\t"))
    units_idx = header_idx + 1
    data_start = units_idx + 2

    headers = lines[header_idx].strip().split("\t")
    units = lines[units_idx].strip().split("\t")

    sensor_names = headers[2:]
    sensor_units = units[2:]

    time = []
    data = []
    for l in lines[data_start:]:
        parts = l.strip().split("\t")
        if len(parts) < len(headers):
            continue
        time.append(_parse_time_to_seconds(parts[0]))
        data.append([float(x) for x in parts[2:]])

    array = np.array(data).T
    sensor_names = list(sensor_names)
    sensor_units = list(sensor_units)

    time = np.array(time)            # (nt,)
    dt = float(np.median(np.diff(time))) if time.size >= 2 else None

    return array, sensor_names, sensor_units, time, dt


if __name__ == "__main__":
    globpath="uploads/CE249_2024_Lab4data/*.txt"
    for path in glob.glob(globpath):
        print(path)
        array, sensor_names, sensor_units, time, dt = get_249_data(path)
        print("unique units:", sorted(set(sensor_units)))

