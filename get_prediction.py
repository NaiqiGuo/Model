"""
Simulate predictions from identified systems and compute prediction error.
"""

import argparse
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.lines import Line2D
import plotly.graph_objects as go

from mdof.simulate import simulate
from mdof.utilities.testing import intensity_bounds, truncate_by_bounds, align_signals
from mdof.validation import stabilize_discrete
import utilities_visualization

from utilities import create_and_save_csv

Q_MAP = {
    "acceleration": {"name": "Acceleration", "units": "in/s²"},
    "displacement": {"name": "Displacement", "units": "in"},
}

def get_event_ids(in_dir: Path):
    """
    Get event IDs from the input directory.

    :param in_dir: Path to the input directory containing event data.
    :return: List of event IDs (str).
    """
    events = sorted((in_dir / "acceleration" / "System ID Training Data" / "ground").glob("[0-9]*.csv"), key=lambda event_path: int(event_path.stem))
    event_ids = [event.stem for event in events]
    return event_ids

def get_output_labels(structure: str):
    if structure == "frame":
        # return ['Floor 1, X', 'Floor 2, X', 'Floor 3, X']
        return ['Floor 1', 'Floor 2', 'Floor 3']
    elif structure == "bridge":
        # return ['West Abutment-Deck Interface, Y', 'Column 1 Top, Y', 'East Abutment-Deck Interface, Y']
        return ['Deck West End', 'North Column Top', 'Deck East End']

def normalized_l2_error(true, test):
    assert true.shape == test.shape, f"Shapes are different for true series ({true.shape}) and test series ({test.shape})."
    denom = np.linalg.norm(true)
    error = np.linalg.norm(test - true)
    if denom == 0:
        return error
    return error / denom


@dataclass(frozen=True)
class RunConfig:
    """Run-constant analysis configuration, built from CLI args."""
    structure: str
    source: str
    sid_method: str
    output_quantity: str
    windowed: bool
    align_signals: bool
    out_labels: list[str]
    annotated: bool
    verbose: int
    in_sid_dir: Path
    out_sid_dir: Path

    @classmethod
    def from_args(cls, args) -> "RunConfig":
        in_sid_dir = Path("System ID") / args.structure / args.source
        out_sid_dir = Path("System ID") / args.structure / args.source
        if not in_sid_dir.exists():
            raise FileNotFoundError(f"Input directory with training data does not exist. \
                                      Run `get_systems.py` first to generate system realizations \
                                      for {args.structure}/{args.source}.")
        
        out_labels = get_output_labels(args.structure)

        return cls(
            structure=args.structure,
            source=args.source,
            sid_method=args.sid_method,
            output_quantity=args.output_quantity,
            windowed=args.windowed,
            align_signals=args.align_signals,
            out_labels=out_labels,
            annotated=args.annotated,
            verbose=args.verbose,
            in_sid_dir=in_sid_dir,
            out_sid_dir=out_sid_dir,
        )


@dataclass
class WindowPlan:
    """Windowing plan, computed once (before the per-event prediction loop)."""
    bounds_by_event: dict = field(default_factory=dict)
    median_length: int | None = None

def compute_window_plan(cfg: RunConfig,
                        event_ids: list[str],
                        quantity: str) -> WindowPlan:
    """
    Scan every event's true output once to determine intensity-window bounds
    and the median window length, so each event can be truncated to a
    consistent length later.
    """
    if not cfg.windowed:
        return WindowPlan()

    bounds_by_event = {}
    window_lengths = []
    window_lengths_seconds = []
    for event_id in event_ids:
        dt       = np.loadtxt(cfg.in_sid_dir / quantity / "System ID Training Data" / "dt"        / f"{event_id}.csv")
        out_true = np.loadtxt(cfg.in_sid_dir / quantity / "System ID Training Data" / "structure" / f"{event_id}.csv").copy()
        bounds_scan = intensity_bounds(out_true, lb=0.01, ub=0.99)
        bounds_by_event[event_id] = bounds_scan
        window_lengths.append(bounds_scan[1] - bounds_scan[0])
        window_lengths_seconds.append((bounds_scan[1] - bounds_scan[0]) * dt)

    median_length = int(np.median(window_lengths))
    median_seconds = np.median(window_lengths_seconds)

    if cfg.verbose:
        print(
            f"{cfg.structure} {cfg.source} {quantity} truncation median window length: "
            f"{median_length} samples ({median_seconds:.3f} s)"
            )

    return WindowPlan(bounds_by_event=bounds_by_event, median_length=median_length)


class Predict:
    """
    Evaluate a system realization's prediction for a single event and quantity.

    Reads the true data and the identified system from training data directory;
    simulates the predicted output of the trained system for the true inputs;
    windows and aligns true and predicted signals;
    saves windowed and aligned true and predicted signals;
    computes error of true vs. predicted outputs;
    and plots true vs. predicted timeseries.

    One instance per quantity, per event.
    """

    def __init__(
        self,
        cfg: RunConfig,
        event_id: str,
        quantity: str,
        window_plan: WindowPlan,
    ):
        self.cfg = cfg
        self.event_id = event_id
        self.quantity = quantity # displacement or acceleration
        self.window_plan = window_plan

        # Training data
        self.inputs = None 
        self.outputs = None
        self.dt = None
        self.time = None 
        self.system = None
        # Ground truth
        self.time_true = None
        self.in_true = None
        self.out_true = None
        # Prediction
        self.out_pred = None
        self.errors = None

    def load_training_data(self):
        """Load the true input, true output, and dt for this event."""
        cfg = self.cfg
        event_id = self.event_id
        quantity = self.quantity

        self.dt      = np.loadtxt(cfg.in_sid_dir / quantity / "System ID Training Data" / "dt"        / f"{event_id}.csv")
        self.time    = np.loadtxt(cfg.in_sid_dir / quantity / "System ID Training Data" / "time"      / f"{event_id}.csv")
        self.inputs  = np.loadtxt(cfg.in_sid_dir / quantity / "System ID Training Data" / "ground"    / f"{event_id}.csv")
        self.outputs = np.loadtxt(cfg.in_sid_dir / quantity / "System ID Training Data" / "structure" / f"{event_id}.csv")

    def load_system(self):
        """Load the identified system realization and stabilize it."""
        cfg = self.cfg
        sys_path = cfg.in_sid_dir / self.quantity / "System ID Results" / "system realization" / f"{self.event_id}.pkl"
        with open(sys_path, "rb") as f:
            A, B, C, D = pickle.load(f)
        self.system = (stabilize_discrete(A), B, C, D)

    def simulate(self):
        """Simulate the predicted output from the identified system."""
        if self.system is None or self.inputs is None:
            raise ValueError("Call load_training_data() and load_system() before simulate().")
        self.out_pred = simulate(self.system, self.inputs)

    def window(self):
        """Truncate inputs/true/predicted/time to the event's intensity-bound window."""
        cfg = self.cfg
        if not cfg.windowed:
            self.time_true = self.time
            self.in_true = self.inputs
            self.out_true = self.outputs
            return

        bounds = self.window_plan.bounds_by_event.get(self.event_id)
        ilb, iub = bounds
        window_length = iub - ilb
        median_length = self.window_plan.median_length
        target_length = max(window_length, median_length)
        center = (ilb + iub) // 2
        half = target_length // 2
        ilb = max(0, center - half)
        iub = ilb + target_length
        if iub > self.outputs.shape[1]:
            iub = self.outputs.shape[1]
            ilb = max(0, iub - target_length)
        bounds = (ilb, iub)

        self.time_true = truncate_by_bounds(self.time, bounds)
        self.in_true = truncate_by_bounds(self.inputs, bounds)
        self.out_true = truncate_by_bounds(self.outputs, bounds)
        self.out_pred = truncate_by_bounds(self.out_pred, bounds)

    def align(self):
        """Align true vs. predicted signals based on outputs."""
        cfg = self.cfg
        if not cfg.align_signals:
            return
        if cfg.verbose == 2:
            print(f">>> Aligning signals for Event {self.event_id} {self.quantity}.")

        lag, out_true_aln, out_pred_aln, time_aln = align_signals(self.out_true,
                                                                  self.out_pred,
                                                                  self.time_true,
                                                                  verbose=False)
        self.time_true = time_aln
        self.in_true = self.in_true[:,:-lag]
        self.out_true = out_true_aln
        self.out_pred = out_pred_aln

    def save_true(self):
        """Save true signals (time, input, output)."""
        cfg = self.cfg
        for array_name, array in [
                                  ("dt", self.dt),
                                  ("time_processed", self.time_true),
                                  ("inputs_processed", self.in_true),
                                  ("outputs_true_processed", self.out_true),
                                 ]:
            create_and_save_csv(cfg.out_sid_dir / self.quantity / "System ID Results" /
                                array_name / f"{self.event_id}.csv",
                                array,
                                rewrite=True)

    def save_prediction(self):
        """Save predicted output."""
        cfg = self.cfg
        create_and_save_csv(cfg.out_sid_dir / self.quantity / "System ID Results" /
                            "outputs_pred_processed" / f"{self.event_id}.csv",
                            self.out_pred,
                            rewrite=True)

    def compute_errors(self):
        """Compute the normalized L2 error; save error csv."""
        cfg = self.cfg
        self.errors = np.array([
            normalized_l2_error(true, pred)
            for true, pred in zip(self.out_true, self.out_pred)
        ])
        create_and_save_csv(cfg.out_sid_dir / self.quantity / "System ID Results" /
                            "errors" / f"{self.event_id}.csv",
                            self.errors,
                            rewrite=True)

    def plot(self):
        """Plot true vs. predicted output timeseries (matplotlib PNG + plotly HTML)."""
        cfg = self.cfg
        out_labels = cfg.out_labels
        annotated = cfg.annotated
        prediction_plot_dir = cfg.out_sid_dir / self.quantity / "System ID Results" / "prediction plots"
        prediction_plot_dir.mkdir(parents=True, exist_ok=True)

        n_outputs = len(out_labels)
        fig_plt, axs = plt.subplots(n_outputs, 1, figsize=(8, 1.75 * n_outputs), sharex=True)
        fig_plt.subplots_adjust(hspace=0.5, top=0.85, bottom=0.17, left=0.13, right=0.85)
        fig_go = go.Figure()

        colors_go = iter(["blue", "darkorange", "green"])
        y_limits = (
            min(np.min(self.out_true), np.min(self.out_pred)),
            max(np.max(self.out_true), np.max(self.out_pred)),
        )

        for i,out_label in enumerate(out_labels):
            color = next(colors_go)
            axs[i].plot(self.time_true, self.out_true[i], color="black", linestyle='-', label="True")
            axs[i].plot(self.time_true, self.out_pred[i], color="red", linestyle='--', label="Pred")
            if annotated:
                axs[i].set_title(out_label, fontsize=16, fontweight="bold", pad=10)
                axs[i].set_ylim(*y_limits)
                axs[i].tick_params(axis="both", labelsize=14)
            fig_go.add_scatter(
                x=self.time_true, y=self.out_true[i],
                mode="lines", line=dict(color=color), name=f"True {out_label}",
            )
            fig_go.add_scatter(
                x=self.time_true, y=self.out_pred[i],
                mode="lines", line=dict(color=color, dash="dash"), name=f"Pred {out_label}",
            )
        fig_go.update_layout(
            title=f"Event {self.event_id} Prediction ({cfg.source})",
            xaxis_title="Time (s)",
            yaxis_title=f"{Q_MAP[self.quantity]['name']} ({Q_MAP[self.quantity]['units']})",
            legend=dict(orientation="h", yanchor="bottom", y=0.0, xanchor="left", x=0, font=dict(size=18)),
        )
        fig_go.update_xaxes(rangeslider=dict(visible=True))
        fig_go.write_html(prediction_plot_dir / f"prediction.html", include_plotlyjs="cdn")

        if annotated:
            fig_plt.align_ylabels()
            fig_plt.supylabel(f"{Q_MAP[self.quantity]['name']} ({Q_MAP[self.quantity]['units']})", fontsize=16, fontweight=900, x=0.04)
            fig_plt.supxlabel("Time (s)", fontsize=16, fontweight=900, y=0.06)
            fig_plt.suptitle(
                f"Event {self.event_id} Prediction ({cfg.source})",
                fontsize=18, fontweight="bold", y=0.965,
            )
            legend_items = [
                {"x": 0.87, "y0": 0.60, "y1": 0.67, "label": "True", "color": "black", "linestyle": "-"},
                {"x": 0.87, "y0": 0.42, "y1": 0.49, "label": "Pred", "color": "red", "linestyle": "--"},
            ]
            for item in legend_items:
                fig_plt.add_artist(Line2D(
                    [item["x"], item["x"]], [item["y0"], item["y1"]],
                    transform=fig_plt.transFigure, color=item["color"], linestyle=item["linestyle"], linewidth=1.0,
                ))
                fig_plt.text(
                    item["x"] + 0.03, (item["y0"] + item["y1"]) / 2, item["label"],
                    rotation=90, va="center", ha="center", fontsize=18, fontweight="bold", transform=fig_plt.transFigure,
                )
        fig_plt.savefig(prediction_plot_dir / f"{self.event_id}.png", dpi=350)
        plt.close(fig_plt)

    def run(self):
        """
        Run the full prediction pipeline for single quantity/event.
        """
        self.load_training_data()
        self.load_system()
        self.simulate()
        self.window()
        self.align()
        self.save_true()
        self.save_prediction()
        self.compute_errors()
        self.plot()


class PredictionHeatmaps:
    """
    Build the cross-event error heatmaps from all events' errors,
    for a single structure/source/quantity.
    """

    def __init__(self,
                 cfg: RunConfig,
                 heatmap_errors: np.ndarray,
                 event_ids: list[str],
                 quantity: str,
                 out_labels: list[str]):
        
        self.cfg = cfg
        self.heatmap_data = heatmap_errors
        self.quantity = quantity
        self.event_ids = event_ids
        self.out_labels = out_labels

        self.heatmap_dir = cfg.out_sid_dir / quantity / "System ID Results"
        self.heatmap_dir.mkdir(parents=True, exist_ok=True)

        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad(color="lightgray")
        self.cmap = cmap

        finite_values = heatmap_errors[np.isfinite(heatmap_errors)]
        solved_vmax = max(np.max(0.7*finite_values), 1.5)
        if cfg.structure == "frame":
            # self.vmax = 1.0
            self.vmax = solved_vmax
        elif cfg.structure == "bridge":
            # self.vmax = 1.0
            self.vmax = solved_vmax

    def save_full(self):
        """Non-square heatmap with per-cell error values and a colorbar."""
        cfg = self.cfg
        n_events = len(self.event_ids)
        fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
        im = ax.imshow(
            np.ma.masked_invalid(self.heatmap_data),
            vmin=0, vmax=self.vmax,
            aspect='auto',
            origin='lower',
            cmap=self.cmap,
        )
        cbar = fig.colorbar(im, ax=ax, extend='max')
        cbar.set_label(r"$\varepsilon$ (Error)", fontsize=18)
        cbar.ax.tick_params(labelsize=16)
        ax.set_xlabel("Event", fontsize=18)
        ax.set_xticks(np.arange(n_events))
        ax.set_xticklabels(self.event_ids, rotation=45, fontsize=16)
        ax.set_yticks(np.arange(len(self.out_labels)))
        ax.set_yticklabels(self.out_labels, fontsize=16)
        for i in range(n_events):
            for j in range(len(self.out_labels)):
                val = self.heatmap_data[j, i]
                if np.isfinite(val):
                    text_value = f"{val:.2f}"
                    color = 'black' if val > self.vmax/2 else 'white'
                else:
                    text_value = "N/A"
                    color = 'black'
                ax.text(i, j, text_value, ha='center', va='center', color=color, fontsize=9)
        fig.savefig(self.heatmap_dir / f"heatmap.png", dpi=400)
        plt.close(fig)

    def save_square(self):
        """Square, compact heatmap with no cell text labels."""
        cfg = self.cfg
        n_events = len(self.event_ids)
        fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
        im = ax.imshow(
            np.ma.masked_invalid(self.heatmap_data),
            vmin=0, vmax=self.vmax,
            aspect='equal',
            origin='lower',
            cmap=self.cmap,
        )
        cbar = fig.colorbar(im, ax=ax, extend='max', fraction=0.02, pad=0.04)
        cbar.set_label("$\\epsilon$: Normalized $L_2$ Error", fontsize=20)
        cbar.ax.tick_params(labelsize=15)
        ax.set_xlabel("Event", fontsize=22)
        ax.set_xticks(np.arange(n_events))
        ax.set_xticklabels(self.event_ids, rotation=45, fontsize=15)
        ax.set_yticks(np.arange(len(self.out_labels)))
        ax.set_yticklabels(self.out_labels, fontsize=15)
        fig.savefig(self.heatmap_dir / f"heatmap_square.png", dpi=400)
        plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Simulate predictions from identified systems and compute prediction error.")
    parser.add_argument("--structure", type=str, default="bridge", choices=["frame", "bridge"], help="Structure type: 'frame' or 'bridge'.")
    parser.add_argument("--source", type=str, default="field", choices=["field", "elastic", "inelastic"], help="Source of data: 'field', 'elastic', or 'inelastic'.")
    parser.add_argument("--sid_method", type=str, default="srim", choices=["srim"], help="System ID method.")
    parser.add_argument("--output_quantity", type=str, default="displacement", choices=["displacement", "acceleration"], help="Output quantity to predict.")
    parser.add_argument("--no_windowing", action="store_false", dest="windowed", help="Disable window truncation after prediction and before aligning/computing error/plotting.")
    parser.add_argument("--no_signal_align", action="store_false", dest="align_signals", help="Disable alignment of true/predicted signals via cross-correlation before computing error.")
    parser.add_argument("--annotate_plots", action="store_true", dest="annotated", help="Include plot annotations. Without this flag, plots show lines only; no text.")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level: 0 (silent), 1 (progress), 2 (progress + alignment detail).")
    return parser.parse_args()


if __name__ == "__main__":
    cfg = RunConfig.from_args(parse_args())

    if cfg.verbose:
        print(f"structure={cfg.structure}")
        print(f"source={cfg.source}")

    event_ids = get_event_ids(cfg.in_sid_dir)
    n_events = len(event_ids)
    failed_events = []

    for quantity in ["displacement", "acceleration"]:
        window_plan = compute_window_plan(cfg, event_ids, quantity)
        errors = np.full((n_events, len(cfg.out_labels)), np.nan)
        for i,event_id in enumerate(tqdm(event_ids)):
            predict = Predict(cfg, event_id, quantity, window_plan)
            try:
                predict.run()
            except Exception as e:
                failed_events.append((event_id, cfg.source, quantity, e))
                if cfg.verbose:
                    print(f"\n>>>> Prediction for event {event_id} FAILED for {cfg.source},{quantity}")
                    print(f">>>> Error: {e}")
                continue
            errors[i] = predict.errors

        heatmap_errors = errors.T # n_outputs x n_events
        heatmaps = PredictionHeatmaps(cfg, heatmap_errors, event_ids, quantity, cfg.out_labels)
        heatmaps.save_full()
        heatmaps.save_square()

    if cfg.verbose and failed_events:
        print(f"Failed events: {failed_events}")
