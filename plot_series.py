from pathlib import Path
import re
import pickle
import numpy as np
import matplotlib.pyplot as plt


MODELING_DIR = Path("Modeling")
SYSTEM_ID_DIR = Path("System ID")
SYSTEM_ID_TRAINING = "System ID Training Data"
SYSTEM_ID_RESULTS = "System ID Results"
Q_META = {
    "displacement": {"ylabel": "Displacement (in)"},
    "acceleration": {"ylabel": "Acceleration (in/s^2)"},
}


def get_channel_labels(structure: str, location: str, n_channels: int):
    if location == "structure":
        dof_map = {1: "X", 2: "Y", 3: "Z", 4: "RX", 5: "RY", 6: "RZ"}
        if structure == "frame":
            nodes = [5, 5, 10, 10, 15, 15]
            dofs = [1, 2, 1, 2, 1, 2]
            labels = [f"Node{n}{dof_map[d]}" for n, d in zip(nodes, dofs)]
        elif structure == "bridge":
            nodes = [9, 3, 10]
            dofs = [2, 2, 2]
            labels = [f"Node{n}{dof_map[d]}" for n, d in zip(nodes, dofs)]
        else:
            labels = []
        if len(labels) == n_channels:
            return labels
    elif location == "ground":
        if structure == "frame":
            labels = ["Channel0 (X)", "Channel2 (Y)"]
            if len(labels) == n_channels:
                return labels
        elif structure == "bridge":
            labels = ["Channel1 (-X)", "Channel3 (Y)"]
            if len(labels) == n_channels:
                return labels
    return [f"ch{ch}" for ch in range(n_channels)]


def window_bounds(signal: np.ndarray, lb: float = 0.001, ub: float = 0.999):
    n = signal.shape[0]
    abs_cum = np.cumsum(np.abs(signal))
    total = float(abs_cum[-1]) if n > 0 else 0.0
    if n == 0 or total <= 0:
        return 0, n
    start = int(np.searchsorted(abs_cum, lb * total, side="left"))
    end = int(np.searchsorted(abs_cum, ub * total, side="right"))
    start = max(0, min(start, n - 1))
    end = max(start + 1, min(end, n))
    return start, end


def parse_strain_stress(csv_path: Path, element_id: int | None = None):
    data = np.genfromtxt(csv_path, names=True, dtype=float, encoding="utf-8")
    names = data.dtype.names or ()
    stress_ids = {
        int(m.group(1))
        for name in names
        for m in [re.match(r"ele(\d+)_stress$", name)]
        if m
    }
    strain_ids = {
        int(m.group(1))
        for name in names
        for m in [re.match(r"ele(\d+)_strain$", name)]
        if m
    }
    available_ids = sorted(stress_ids & strain_ids)
    if not available_ids:
        return []

    selected_ids = available_ids
    if element_id is not None:
        if element_id not in available_ids:
            return []
        selected_ids = [element_id]

    pairs = []
    for eid in selected_ids:
        stress = np.atleast_1d(data[f"ele{eid}_stress"]).astype(float)
        strain = np.atleast_1d(data[f"ele{eid}_strain"]).astype(float)
        n = min(stress.shape[0], strain.shape[0])
        pairs.append((eid, strain[:n], stress[:n]))
    return pairs


def available_elements(csv_path: Path):
    data = np.genfromtxt(csv_path, names=True, dtype=float, encoding="utf-8")
    names = data.dtype.names or ()
    stress_ids = {
        int(m.group(1))
        for name in names
        for m in [re.match(r"ele(\d+)_stress$", name)]
        if m
    }
    strain_ids = {
        int(m.group(1))
        for name in names
        for m in [re.match(r"ele(\d+)_strain$", name)]
        if m
    }
    return sorted(stress_ids & strain_ids)


def normalize_dataset(name: str):
    s = (name or "").strip().lower()
    if s in {"system id", "systemid", "sid"}:
        return "System ID"
    return "Modeling"


def system_id_result_dirname(result_type: str):
    key = (result_type or "").strip().lower().replace(" ", "_")
    mapping = {
        "prediction": "prediction",
        "prediction_error": "prediction error",
        "frequency_id": "frequency ID",
        "mode_shapes": "mode shapes",
        "system_realization": "system realization",
    }
    return mapping.get(key, "prediction")


def system_id_result_variants(dirname: str):
    variants = {dirname}
    variants.add(dirname.replace(" ", "_"))
    variants.add(dirname.replace("_", " "))
    variants.add(dirname.lower())
    variants.add(dirname.replace(" ", "").lower())
    return [v for v in variants if v]


def normalize_selection(quantity: str, location: str, source: str, dataset: str, sid_section: str):
    if dataset == "Modeling":
        if quantity in {"strain_stress", "frequency_pre_eq", "frequency_post_eq"} and location != "structure":
            print(f"{quantity} is only available at location=structure; switching location to structure")
            location = "structure"

        if location == "ground" and source != "field":
            print("ground data is only available under source=field; switching source to field")
            source = "field"

        if quantity in {"dt", "time"} and source != "field":
            print(f"{quantity} is only available under source=field; switching source to field")
            source = "field"

        if quantity in {"strain_stress", "frequency_pre_eq", "frequency_post_eq"} and source == "field":
            print(f"{quantity} is only available under source=elastic/inelastic; switching source to elastic")
            source = "elastic"
    else:
        if sid_section == "results":
            if quantity not in {"displacement", "acceleration"}:
                print("System ID results are organized under displacement/acceleration; switching quantity to acceleration")
                quantity = "acceleration"
            if location not in {"ground", "structure"}:
                location = "structure"
        else:
            if quantity in {"strain_stress", "frequency_pre_eq", "frequency_post_eq"}:
                print("System ID training data supports displacement/acceleration/time/dt; switching quantity to acceleration")
                quantity = "acceleration"
            if location == "ground" and source != "field":
                print("System ID ground data is only available under source=field; switching source to field")
                source = "field"
    return quantity, location, source


def series_path_candidates(
    dataset: str,
    structure: str,
    source: str,
    quantity: str,
    location: str,
    event_id: str,
    sid_section: str = "results",
    sid_result_type: str = "prediction",
):
    candidates = []
    exts = [".csv", ".txt", ".npy", ".pkl"]
    if dataset == "Modeling":
        base = MODELING_DIR / structure / source / quantity / location
        candidates.extend([base / f"{event_id}{ext}" for ext in exts])
        return candidates

    root = SYSTEM_ID_DIR / structure / source / quantity
    if sid_section == "training":
        bases = [
            root / SYSTEM_ID_TRAINING / location,
            root / SYSTEM_ID_TRAINING,
        ]
        if quantity in {"dt", "time"}:
            bases.extend([
                SYSTEM_ID_DIR / structure / source / "displacement" / SYSTEM_ID_TRAINING / quantity / location,
                SYSTEM_ID_DIR / structure / source / "displacement" / SYSTEM_ID_TRAINING / quantity,
                SYSTEM_ID_DIR / structure / source / "acceleration" / SYSTEM_ID_TRAINING / quantity / location,
                SYSTEM_ID_DIR / structure / source / "acceleration" / SYSTEM_ID_TRAINING / quantity,
            ])
        for base in bases:
            candidates.extend([base / f"{event_id}{ext}" for ext in exts])
        return candidates

    result_dir = system_id_result_dirname(sid_result_type)
    bases = []
    for result_variant in system_id_result_variants(result_dir):
        base = root / SYSTEM_ID_RESULTS / result_variant
        bases.append(base / location)
        bases.append(base)
    for base in bases:
        candidates.extend([base / f"{event_id}{ext}" for ext in exts])
    return candidates


def load_series_array(path_used: Path):
    suffix = path_used.suffix.lower()
    if suffix == ".csv":
        try:
            arr = np.loadtxt(path_used, delimiter=",")
        except ValueError:
            arr = np.loadtxt(path_used, delimiter=",", skiprows=1)
        return arr
    if suffix == ".txt":
        return np.loadtxt(path_used)
    if suffix == ".npy":
        return np.load(path_used)
    if suffix == ".pkl":
        with open(path_used, "rb") as f:
            return pickle.load(f)
    raise ValueError(f"unsupported file type: {path_used}")


def list_event_ids(
    dataset: str,
    structure: str,
    source: str,
    quantity: str,
    location: str,
    sid_section: str = "results",
    sid_result_type: str = "prediction",
):
    ids = set()
    stem_target = set()
    probe = ["1", "226", "event"]
    for p in probe:
        for cand in series_path_candidates(
            dataset=dataset,
            structure=structure,
            source=source,
            quantity=quantity,
            location=location,
            event_id=p,
            sid_section=sid_section,
            sid_result_type=sid_result_type,
        ):
            stem_target.add(cand.parent)

    for directory in stem_target:
        if not directory.exists():
            continue
        for ext in ("*.csv", "*.txt", "*.npy", "*.pkl"):
            for f in directory.glob(ext):
                ids.add(f.stem)
    return sorted(ids)


def load_dt(structure: str, source: str, event_id: str, location: str, dataset: str, sid_section: str):
    if dataset == "System ID":
        dt_candidates = series_path_candidates(
            dataset="System ID",
            structure=structure,
            source=source,
            quantity="dt",
            location=location,
            event_id=event_id,
            sid_section="training",
            sid_result_type="prediction",
        )
        dt_candidates.extend(
            series_path_candidates(
                dataset="System ID",
                structure=structure,
                source=source,
                quantity="dt",
                location="ground",
                event_id=event_id,
                sid_section="training",
                sid_result_type="prediction",
            )
        )
        for dt_path in dt_candidates:
            if dt_path.exists():
                try:
                    return float(np.loadtxt(dt_path))
                except ValueError:
                    with open(dt_path, "r") as f:
                        return float(f.read().strip())

    candidates = [
        MODELING_DIR / structure / "field" / "dt" / location / f"{event_id}.txt",
        MODELING_DIR / structure / "field" / "dt" / "ground" / f"{event_id}.txt",
        MODELING_DIR / structure / "field" / "dt" / "structure" / f"{event_id}.txt",
    ]
    for dt_path in candidates:
        if dt_path.exists():
            try:
                return float(np.loadtxt(dt_path))
            except ValueError:
                with open(dt_path, "r") as f:
                    return float(f.read().strip())
    return None


if __name__ == "__main__":
    replot = True

    while replot:
        dataset_default = normalize_dataset(input("Base dataset. Modeling or System ID? [Modeling]: ").strip() or "Modeling")
        structure = input("Which structure do you want. frame or bridge? ").strip() or "frame"
        event_source = input("Which event source. field, elastic, or inelastic? ").strip() or "field"
        quantity = input("Which quantity. time, dt, displacement, acceleration, strain_stress, frequency_pre_eq, or frequency_post_eq? ").strip() or "acceleration"
        event_location = input("Which location. ground or structure? ").strip() or "ground"
        sid_section_default = "results"
        sid_result_type_default = "prediction"
        if dataset_default == "System ID":
            sid_section_default = (input("System ID section. training or results? [results]: ").strip().lower() or "results")
            if sid_section_default not in {"training", "results"}:
                sid_section_default = "results"
            if sid_section_default == "results":
                sid_result_type_default = (
                    input("System ID result type. prediction, prediction_error, frequency_id, mode_shapes, or system_realization? [prediction]: ").strip().lower()
                    or "prediction"
                )
        quantity, event_location, event_source = normalize_selection(
            quantity, event_location, event_source, dataset_default, sid_section_default
        )
        use_window = False
        selected_element = None
        element_prompted = False
        if quantity in Q_META and not (dataset_default == "System ID" and sid_section_default == "results" and sid_result_type_default in {"frequency_id", "mode_shapes", "system_realization"}):
            use_window = (input("Window time range by intensity bounds? [y/N]: ").strip().lower() == "y")

        event_ids = list_event_ids(
            dataset=dataset_default,
            structure=structure,
            source=event_source,
            quantity=quantity,
            location=event_location,
            sid_section=sid_section_default,
            sid_result_type=sid_result_type_default,
        )
        print("event ids:", event_ids)
        event_id = input("event id: ").strip()

        is_text_only = quantity in {"dt", "time", "frequency_pre_eq", "frequency_post_eq"} or (
            dataset_default == "System ID"
            and sid_section_default == "results"
            and sid_result_type_default in {"frequency_id", "mode_shapes", "system_realization"}
        )
        fig = ax = None
        if not is_text_only:
            fig, ax = plt.subplots(figsize=(10, 4))
        loaded_count = 0
        missing_dt_count = 0

        add_series = True
        first_series = True
        series_specs = []
        while add_series:
            if first_series:
                series_event_id = event_id
                source = event_source
                location = event_location
                dataset = dataset_default
                sid_section = sid_section_default
                sid_result_type = sid_result_type_default
                first_series = False
            else:
                series_event_id = input(f"event id [{event_id}]: ").strip() or event_id
                dataset = normalize_dataset(input(f"dataset (Modeling/System ID) [{dataset_default}]: ").strip() or dataset_default)
                source = input(f"source (field/elastic/inelastic) [{event_source}]: ").strip() or event_source
                location = input(f"location (ground/structure) [{event_location}]: ").strip() or event_location
                sid_section = sid_section_default
                sid_result_type = sid_result_type_default
                if dataset == "System ID":
                    sid_section = (
                        input(f"System ID section (training/results) [{sid_section_default}]: ").strip().lower()
                        or sid_section_default
                    )
                    if sid_section not in {"training", "results"}:
                        sid_section = "results"
                    if sid_section == "results":
                        sid_result_type = (
                            input(
                                f"System ID result type (prediction/prediction_error/frequency_id/mode_shapes/system_realization) "
                                f"[{sid_result_type_default}]: "
                            ).strip().lower()
                            or sid_result_type_default
                        )
                quantity, location, source = normalize_selection(quantity, location, source, dataset, sid_section)

            candidates = series_path_candidates(
                dataset=dataset,
                structure=structure,
                source=source,
                quantity=quantity,
                location=location,
                event_id=series_event_id,
                sid_section=sid_section,
                sid_result_type=sid_result_type,
            )
            path_used = next((p for p in candidates if p.exists()), None)
            if path_used is None:
                print(f"missing files for {dataset} | {series_event_id} | {source}/{location} | {quantity}")
                add_series = (input("add another? [y/N]: ").strip().lower() == "y")
                continue
            arr = load_series_array(path_used)

            if path_used.suffix.lower() == ".pkl":
                if isinstance(arr, tuple) and len(arr) >= 4:
                    print(f"{series_event_id} | {dataset} | {source}/{location} | {sid_result_type} | A,B,C,D shapes:")
                    for name, mat in zip(["A", "B", "C", "D"], arr[:4]):
                        shape = np.asarray(mat).shape
                        print(f"  {name}: {shape}")
                else:
                    print(f"{series_event_id} | {dataset} | {source}/{location} | {sid_result_type} | pkl type={type(arr)}")
                print(f"loaded: {path_used}")
                loaded_count += 1
                series_specs.append((series_event_id, source, location, dataset))
                add_series = (input("add another? [y/N]: ").strip().lower() == "y")
                continue

            if np.isscalar(arr) or np.ndim(arr) == 0:
                arr = np.array([float(arr)])

            if quantity == "strain_stress":
                if path_used.suffix.lower() != ".csv":
                    print(f"strain_stress expects csv with headers; got: {path_used}")
                    add_series = (input("add another? [y/N]: ").strip().lower() == "y")
                    continue
                if not element_prompted:
                    avail = available_elements(path_used)
                    print(f"available elements: {avail}")
                    element_text = input("Element id for stress-strain (blank=all): ").strip()
                    if element_text:
                        try:
                            selected_element = int(element_text)
                        except ValueError:
                            print(f"invalid element id '{element_text}', using all elements")
                            selected_element = None
                    element_prompted = True
                pairs = parse_strain_stress(path_used, element_id=selected_element)
                if not pairs:
                    avail = available_elements(path_used)
                    print(f"no valid strain/stress columns found in {path_used}; available elements: {avail}")
                    add_series = (input("add another? [y/N]: ").strip().lower() == "y")
                    continue
                for eid, strain, stress in pairs:
                    ax.plot(strain, stress, linewidth=1.2, label=f"{series_event_id} | {source}/{location} ele{eid}")
            elif is_text_only:
                if quantity == "dt":
                    value = float(arr.reshape(-1)[0])
                    print(f"{series_event_id} | {dataset} | {source}/{location} | dt = {value}")
                elif quantity == "time":
                    flat = arr.reshape(-1)
                    if flat.size <= 20:
                        shown = np.array2string(flat, separator=", ")
                        print(f"{series_event_id} | {dataset} | {source}/{location} | time = {shown}")
                    else:
                        head = np.array2string(flat[:5], separator=", ")
                        tail = np.array2string(flat[-5:], separator=", ")
                        print(
                            f"{series_event_id} | {dataset} | {source}/{location} | time n={flat.size} "
                            f"start={flat[0]} end={flat[-1]} head={head} tail={tail}"
                        )
                else:
                    flat = arr.reshape(-1)
                    if flat.size == 0:
                        print(f"{series_event_id} | {dataset} | {source}/{location} | {quantity}: empty")
                    else:
                        print(
                            f"{series_event_id} | {dataset} | {source}/{location} | {quantity} "
                            f"n_modes={flat.size} min={flat.min()} max={flat.max()}"
                        )
                        for i, f in enumerate(flat, start=1):
                            print(f"  mode {i:02d}: {f}")
            elif arr.ndim == 1:
                i0, i1 = 0, arr.shape[0]
                if use_window:
                    i0, i1 = window_bounds(arr)
                    arr = arr[i0:i1]
                if quantity in Q_META:
                    dt = load_dt(structure, source, series_event_id, location, dataset, sid_section)
                    if dt is None:
                        x = np.arange(i0, i1)
                        missing_dt_count += 1
                    else:
                        x = np.arange(i0, i1) * dt
                else:
                    x = np.arange(i0, i1)
                if dataset == "System ID":
                    series_label = f"{series_event_id} | {dataset}:{sid_section}:{sid_result_type} | {source}/{location}"
                else:
                    series_label = f"{series_event_id} | {dataset} | {source}/{location}"
                if use_window:
                    series_label += f" window[{i0}:{i1}]"
                ax.plot(x, arr, label=series_label)
            else:
                i0, i1 = 0, arr.shape[1]
                if use_window:
                    i0, i1 = window_bounds(arr[0])
                    arr = arr[:, i0:i1]
                if quantity in Q_META:
                    dt = load_dt(structure, source, series_event_id, location, dataset, sid_section)
                    if dt is None:
                        x = np.arange(i0, i1)
                        missing_dt_count += 1
                    else:
                        x = np.arange(i0, i1) * dt
                else:
                    x = np.arange(i0, i1)
                channel_labels = get_channel_labels(structure, location, arr.shape[0])
                for ch in range(arr.shape[0]):
                    if dataset == "System ID":
                        series_label = f"{series_event_id} | {dataset}:{sid_section}:{sid_result_type} | {source}/{location} {channel_labels[ch]}"
                    else:
                        series_label = f"{series_event_id} | {dataset} | {source}/{location} {channel_labels[ch]}"
                    if use_window:
                        series_label += f" window[{i0}:{i1}]"
                    ax.plot(x, arr[ch], label=series_label)

            print(f"loaded: {path_used}")
            loaded_count += 1
            series_specs.append((series_event_id, source, location, dataset))
            add_series = (input("add another? [y/N]: ").strip().lower() == "y")

        if loaded_count > 0 and not is_text_only:
            compare_groups = sorted({f"{spec[3]}::{spec[1]}" for spec in series_specs})
            if len(compare_groups) > 1:
                print(f"comparison plot across groups: {', '.join(compare_groups)}")

            ax.set_title(f"{structure} | {quantity} | {loaded_count} series")
            if quantity in Q_META and missing_dt_count == 0:
                ax.set_xlabel("Time (s)")
            elif quantity in Q_META:
                ax.set_xlabel("sample (dt missing for some series)")
            else:
                ax.set_xlabel("sample")

            if quantity in Q_META:
                if quantity == "acceleration" and event_location == "ground":
                    ax.set_ylabel("Input Acceleration (in/s^2)")
                else:
                    ax.set_ylabel(Q_META[quantity]["ylabel"])
            elif quantity == "strain_stress":
                ax.set_xlabel("Strain")
                ax.set_ylabel("Stress")
            else:
                ax.set_ylabel(quantity)
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            plt.show()

            if input("save figure? [y/N]: ").strip().lower() == "y":
                plots_dir = MODELING_DIR / structure / "plots"
                plots_dir.mkdir(parents=True, exist_ok=True)
                first_event = series_specs[0][0] if series_specs else event_id
                out_path = plots_dir / f"{first_event}_{quantity}_series.png"
                fig.savefig(out_path, dpi=300)
                print(f"saved: {out_path}")
        elif loaded_count > 0 and is_text_only:
            print(f"printed {loaded_count} series")
        else:
            print("no series loaded")

        replot = (input("plot another? [y/N]: ").strip().lower() == "y")
