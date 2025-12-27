from pathlib import Path
import quakeio
import pickle
import csv
import numpy as np
import os
from mdof import sysid
from mdof.validation import stabilize_discrete
from mdof.utilities.config import Config
from model_utils import (
    get_inputs, analyze, get_node_displacements, stabilize_with_lmi,
    stabilize_by_radius_clipping, write_freq_csv,
    save_displacements, save_strain_stress,
    create_bridge_model,save_event_modes_to_csv     # make sure this is defined in model_utils
)

if __name__ == "__main__":
    # Global switch for bridge model
    ELASTIC = False  # True = elastic bridge, False = inelastic bridge

    # --------------------------------------------------------
    # Load events
    # --------------------------------------------------------
    LOAD_EVENTS = False
    if LOAD_EVENTS:
        events = sorted([
            print(file) or quakeio.read(file, exclusions=["*filter*"])
            # for file in list(Path("/Users/guonaiqi/Documents/GitHub/mdof_studies/uploads/CE89324/").glob("????????*.[zZ][iI][pP]"))
            for file in list(Path("../uploads/CE89324/").glob("????????*.[zZ][iI][pP]"))
        ], key=lambda event: abs(event["peak_accel"]))
        with open("events_bridge.pkl", "wb") as f:
            pickle.dump(events, f)
    else:
        with open("events.pkl", "rb") as f:
            # if you prefer, you can also create a separate events_bridge.pkl
            events = pickle.load(f)

    print(f"Total events loaded: {len(events)}")

    START_EVENT_ID = 1
    END_EVENT_ID   = 22

    # Input channels [x, y]
    input_channels = [1, 3]

    # --------------------------------------------------------
    # Output directories, specific to bridge model
    # --------------------------------------------------------
    if ELASTIC:
        output_dir = "event_outputs_ABCD_bridge_model_elastic"
    else:
        output_dir = "event_outputs_ABCD_bridge_model_inelastic"

    os.makedirs(output_dir, exist_ok=True)

    # --------------------------------------------------------
    # Main loop over events
    # --------------------------------------------------------
    for i, event in enumerate(events):
        event_id = i + 1   

        if event_id < START_EVENT_ID or event_id > END_EVENT_ID:
            continue

        # inputs has shape (2, nt) for [x, y] components
        inputs, dt = get_inputs(
            i,
            events=events,
            input_channels=input_channels,
            scale=0.1  # 2.54
        )
        print(f"\nBridge model, event {i+1} inputs shape: {inputs.shape}, dt = {dt}")

        # ------------------------------------------------
        # Build Painter Street bridge model
        # ------------------------------------------------
        model = create_bridge_model(
            elastic=ELASTIC,
            inputx=inputs[0],
            inputy=inputs[1],
            dt=dt
        )

        nt = inputs.shape[1]

        # Choose output nodes on the deck.
        # Here we use nodes 2, 3, 5.
        # You can adjust this list to match the sensor layout if needed.
        output_nodes = [2, 3, 5]

        
        # Time history analysis
        try:
            disp, stresses, strains, freqs_before, freqs_after = analyze(
                model,
                output_nodes=output_nodes,
                nt=nt,
                dt=dt,
                output_elements=[2,3,5],
                yFiber=8.0,
                zFiber=0.0
            )
            
        except RuntimeError as e:
            print(f"Error for event {i+1}:")
            print(e)
            continue

        write_freq_csv(
            event_id=event_id,
            freqs_before=freqs_before,
            freqs_after=freqs_after,
            freq_csv_path=(
                "natural_frequencies_bridge_elastic.csv"
                if ELASTIC else
                "natural_frequencies_bridge_inelastic.csv"
            )
        )

        outputs = get_node_displacements(disp)
        outputs = outputs[:, 1:]   # drop time row
        assert inputs.shape[1] == outputs.shape[1], \
            "inputs and outputs have different length of time samples."
        time = np.arange(nt) * dt

        # Save displacement and strain/stress for this event
        event_id = i + 1

        if ELASTIC:
            disp_dir = "event_disp_bridge_elastic"
            ss_dir   = "event_strain_stress_bridge_elastic"
        else:
            disp_dir = "event_disp_bridge_inelastic"
            ss_dir   = "event_strain_stress_bridge_inelastic"

        os.makedirs(disp_dir, exist_ok=True)
        os.makedirs(ss_dir, exist_ok=True)

        disp_path = os.path.join(disp_dir, f"event_{event_id:02d}_disp.csv")
        save_displacements(disp, dt, disp_path)
        ss_path = os.path.join(ss_dir, f"event_{event_id:02d}_strain_stress.csv")
        save_strain_stress(stresses, strains, dt, ss_path)


        # System identification options
        # n = 6
        # options = Config(
        #     m           = 500,
        #     horizon     = 190,
        #     nc          = 190,
        #     order       = 2*n,
        #     period_band = (0.1, 0.6),
        #     damping     = 0.06,
        #     pseudo      = True,
        #     outlook     = 190,
        #     threads     = 8,
        #     chunk       = 200,
        #     i           = 250,
        #     j           = 4400
        # )

        n = 2  
        options = Config(
            m           = 120,       
            horizon     = 25,       
            nc          = 25,
            order       = 3,       
            period_band = (0.15, 0.8),
            damping     = 0.05,
            pseudo      = True,
            outlook     = 25,
            threads     = 4,       
            chunk       = 200,
            i           = 250,
            j           = 3500    
        )

        # ---- SRIM ----
        system_srim = sysid(inputs, outputs, method='srim', **options)
        A_s, B_s, C_s, D_s, *rest = system_srim
        system_srim = (A_s, B_s, C_s, D_s)

        with open(os.path.join(output_dir, f"system_srim_bridge_{event_id:02d}.pkl"), "wb") as f:
            pickle.dump(system_srim, f)
        print(f"Saved system_srim_bridge_{event_id:02d}.pkl")

        methods_dict = {
            'srim': (A_s, B_s, C_s, D_s),
            # you can enable other methods here if needed
        }
        # save methods to CSV, with event index i and method dict
        # # ----- True modes from FE eigenvectors -----
        # n_modes = 3
        # freq_true, Phi_true = get_true_modes_xara(
        #     model,
        #     floor_nodes=(2, 3, 5),
        #     dofs=(1, 2),
        #     n=n_modes
        # )

        # # ----- Estimated modes from SRIM -----
        # Phi_srim, eigvals_sel = phi_output(A_s, C_s)

        # Phi_srim = Phi_srim[:, :n_modes]

        # # ----- MAC -----
        # MAC_srim = mac_matrix(Phi_true, Phi_srim)

        # method_modes = {"srim": Phi_srim}
        # method_macs  = {"srim": MAC_srim}
        # algos = ["srim"]

        # modes_dir = "event_modes_bridge_inelastic" if not ELASTIC else "event_modes_bridge_elastic"
        # os.makedirs(modes_dir, exist_ok=True)

        # modes_path = os.path.join(modes_dir, f"event_{event_id:02d}_modes.csv")

        # save_event_modes_to_csv(
        #     event_id=event_id,
        #     Phi_true=Phi_true,
        #     method_modes=method_modes,
        #     method_macs=method_macs,
        #     algos=algos,
        #     filename=modes_path
        # )
