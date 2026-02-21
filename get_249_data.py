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


def _clean_249_unit_string(u: str) -> str:
    return (u or "").strip().lower().replace(" ", "")


def scale_249_units(units: str, standard='iks')->float:

    import xara.units.iks
    import xara.units.ips 
    import xara.units.fks
    # See https://github.com/chrystalchern/xara_units/tree/main/src/xara_units
    unit_system = getattr(xara.units, standard)
    units = _clean_249_unit_string(units)

    if units in {"g", "gs", "g's"}:
        return unit_system.gravity
    elif units in {"m/s", "mps"}:
        return unit_system.m/unit_system.second
    elif units in {"m/s2", "mps2", "m/s/s", "m/s^2"}:
        return unit_system.m/unit_system.second/unit_system.second
    elif units in {"in/s", "ips"}:
        return unit_system.inch/unit_system.second
    elif units in {"in/s2", "inps2", "in/s/s", "in/s^2"}:
        return unit_system.inch/unit_system.second/unit_system.second
    elif units in {"in", "inch", "inches"}:
        return unit_system.inch
    else:
        raise NotImplementedError(f"Unhandled unit: {units}")
    

def get_249_data(path):
    with open(path) as f:
        lines = f.readlines()
    header_idx = next(i for i,l in enumerate(lines) if l.startswith("Time\t"))
    units_idx = header_idx + 1
    data_start = units_idx + 2

    # TODO CC check: Use rstrip(), not strip() to keep names, units, and data aligned
    headers = lines[header_idx].rstrip().split("\t")
    units = lines[units_idx].rstrip().split("\t")

    sensor_names = headers[2:]
    sensor_units = units[2:]

    time = []
    data = []
    for l in lines[data_start:]:
        parts = l.rstrip().split("\t")
        if len(parts) < len(headers):
            continue
        time.append(_parse_time_to_seconds(parts[0]))
        data.append([float(x) for x in parts[2:]])

    array = np.array(data).T
    
    sensor_names = list(sensor_names)
    sensor_units = list(sensor_units)

    time = np.array(time)            # (nt,)
    dt = float(np.median(np.diff(time))) if time.size >= 2 else None
    # TODO Check CC: baseline correct by subtracting average of first five seconds, for all channels
    mask_5s = (time - time[0]) <= 5.0
    baseline = array[:, mask_5s].mean(axis=1, keepdims=True)  # (n_channels, 1)
    array = array - baseline

    return array, sensor_names, sensor_units, time, dt


if __name__ == "__main__":
    globpath="uploads/CE249_2024_Lab4data/*.txt"
    for path in glob.glob(globpath):
        print(path)
        array, sensor_names, sensor_units, time, dt = get_249_data(path)
        print("unique units:", sorted(set(sensor_units)))

    s = scale_249_units("g's")
    print(s, type(s))

