import numpy as np
import glob

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

    data = []
    for l in lines[data_start:]:
        parts = l.strip().split("\t")
        if len(parts) < len(headers):
            continue
        data.append([float(x) for x in parts[2:]])

    array = np.array(data).T
    sensor_names = list(sensor_names)
    sensor_units = list(sensor_units)

    return array, sensor_names, sensor_units


if __name__ == "__main__":
    globpath="uploads/CE249_2024_Lab4data/*.txt"
    for path in glob.glob(globpath):
        print(path)

        array, sensor_names, sensor_units = get_249_data(path)
        print(array.shape, len(sensor_names), len(sensor_units))