from pathlib import Path
import quakeio
import pickle
import csv
import numpy as np
import os
from mdof import sysid
from mdof.validation import stabilize_discrete
from mdof.utilities.config import Config
from utilities_experimental import (
    get_inputs, analyze, get_node_displacements, stabilize_with_lmi,
    stabilize_by_radius_clipping, write_freq_csv,
    save_displacements, save_strain_stress,
    create_bridge_model,save_event_modes_to_csv     # make sure this is defined in model_utils
)
import xara 
from xara import units
import math

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


#%%
def create_bridge_model(elastic: bool = True, girder: str = "elasticBeamColumn"):
    
    # #input check
    # if np.all(inputx is None) or np.all(inputy is None) or dt is None:
    #     raise ValueError("Missing inputx, inputy, or dt. Exiting.")

    # if girder != "elasticBeamColumn":
    #     raise ValueError("Only elasticBeamColumn allowed for girders.")
    
    model = xara.Model(ndm=3, ndf=6)

    if not hasattr(model, "meta") or not isinstance(model.meta, dict):
        model.meta = {}
    model.meta["column_elems"] = []

    # Nodes: (tag, (x, y, z))
    model.node(0, (0.0,                                  0.0,                  320.0)) # abutment 1
    model.node(1, (3180.0,                               0.0,                  320.0)) # abutment 2
    model.node(2, (1752.0,                               0.0,                  320.0)) # mid-deck
    model.node(3, (1641.4628263430573,    199.41297159396336,                  320.0)) # top of column 1
    model.node(4, (1641.4628263430573,    199.41297159396336,                    0.0)) # bottom of column 1
    model.node(5, (1862.5371736569432,   -199.41297159396336,                  320.0)) # top of column 2
    model.node(6, (1862.5371736569432,   -199.41297159396336,                    0.0)) # bottom of column 2

    # Boundary conditions, fully fixed at 0, 1, 4, 6
    # fix(tag, (DX, DY, DZ, RX, RY, RZ))
    model.fix(0, (1, 1, 1, 1, 1, 1))
    model.fix(1, (1, 1, 1, 1, 1, 1))
    model.fix(4, (1, 1, 1, 1, 1, 1))
    model.fix(6, (1, 1, 1, 1, 1, 1))

    # Materials: concrete and steel

    # Concrete strengths (ksi)
    fc_unconf = 4.0   # unconfined concrete
    fc_conf   = 5.0  # confined concrete

    # Concrete modulus (ksi), same formula as your frame model
    Ec = 57000.0 * math.sqrt(fc_unconf * 1000.0) / 1000.0

    # Steel properties
    fy = 60.0   # ksi
    Es = 30000.0

    if not elastic:
        # Nonlinear concrete (core and cover) using Concrete01
        #                    tag  f'c       epsc0         f'cu   epscu
        model.uniaxialMaterial("Concrete01", 1, -fc_conf,   -2*fc_conf/Ec,  -3.5,  -0.02)
        model.uniaxialMaterial("Concrete01", 2, -fc_unconf, -2*fc_unconf/Ec, 0.0,  -0.006)

        # Nonlinear reinforcing steel
        #                    tag fy   E0   b
        model.uniaxialMaterial("Steel01", 3, fy, Es, 0.02)
    else:
        # Elastic concrete for both core and cover
        model.uniaxialMaterial("Elastic", 1, Ec)
        model.uniaxialMaterial("Elastic", 2, Ec)

        # Elastic steel
        model.uniaxialMaterial("Elastic", 3, Es)

    
    # Section properties: 5 ft circular section
    # Geometry of circular section
    D_total = 60 #60.0        # total diameter in inches (5 ft)
    cover   = 2.0         # concrete cover in inches
    R_ext   = D_total / 2.0
    R_core  = R_ext - cover  # approximate core radius

    # mesh subdivisions for nonlinear fiber section
    numSubdivCirc = 32
    numSubdivRad  = 5
    divs = (numSubdivCirc, numSubdivRad)

    # tags
    
    colSec_fiber   = 1
    beamSec_elastic = 2   # beams stay elastic

    # ELASTIC SECTION (for elasticBeamColumn model)
    A_el = math.pi * R_ext**2
    I_el = math.pi * R_ext**4 / 4.0
    J_el = math.pi * R_ext**4 / 2.0
    nu = 0.2
    Gc = Ec / (2*(1+nu))
    GJ   = Gc * J_el
    
    model.section("Fiber", colSec_fiber, "-GJ", GJ)
    itg_col = 1
    npts_col = 4
    model.beamIntegration("Lobatto", itg_col, colSec_fiber, npts_col)

    numSubdivCirc, numSubdivRad = divs

    # core concrete
    model.patch("circ",
                1,                      # matTag = 1
                numSubdivCirc, numSubdivRad,
                0.0, 0.0,               # yCenter, zCenter
                0.0, R_core,            # intRad, extRad
                0.0, 2 * math.pi)

    # cover concrete
    model.patch("circ",
                2,                      # matTag = 2
                numSubdivCirc, numSubdivRad,
                0.0, 0.0,
                R_core, R_ext,
                0.0, 2 * math.pi)

    # longitudinal steel
    numBars = 36 # 36
    barArea = 1.56 #1.56                    # #11 bar area
    model.layer("circ",
                3,                    # steel matTag
                numBars, barArea,
                0.0, 0.0,            # yCenter, zCenter
                R_core,              # radius
                0.0, 2 * math.pi)
   

    beam_stiff_factor = 5.0
    A_beam = 864.0          # 24in × 36in = 864 in^2
    I_beam = 9.33e4 * beam_stiff_factor        # in^4  (strong axis)
    J_beam = 3.73e5 * beam_stiff_factor        # in^4  approximate torsion

    model.section("Elastic", beamSec_elastic, Ec, A_beam, I_beam, I_beam, Gc, J_beam)

    

    # Transformations and elements
    colTransf  = 1
    beamTransf = 2
    model.geomTransf("Linear", colTransf,  (1.0, 0.0, 0.0))
    model.geomTransf("Linear", beamTransf, 0.0, 0.0, 1.0)

    # columns: elastic vs nonlinear
    
    # columns as elasticBeamColumn with elastic section
    col_type = "forceBeamColumn"
    sec_col  = colSec_fiber

    model.element(col_type, 2, (4, 3), transform=colTransf, section=sec_col, shear=0)
    model.element(col_type, 3, (6, 5), transform=colTransf, section=sec_col, shear=0)
    

    model.meta["column_elems"] = [2, 3]

    # beams always elastic
    beam_type = "elasticBeamColumn"
    sec_beam  = beamSec_elastic

    model.element(beam_type, 1, (0, 2), transform=beamTransf, section=sec_beam, shear=0)
    model.element(beam_type, 4, (2, 1), transform=beamTransf, section=sec_beam, shear=0)
    model.element(beam_type, 5, (3, 2), transform=beamTransf, section=sec_beam, shear=0)
    model.element(beam_type, 6, (5, 2), transform=beamTransf, section=sec_beam, shear=0)

    # Mass, damping and earthquake excitation
    # lump_mass = 1.0  # placeholder
    # for nd in [2, 3, 5]:
    #     # mass(tag, (MX, MY, MZ, RX, RY, RZ))
    #     model.mass(nd, (lump_mass, lump_mass, 0.0, 0.0, 0.0, 0.0))

    # ---------- Gravity load and mass: template style ----------
    output_nodes = [2,3,5]
    print('pre-gravity disp')
    for node in output_nodes:
        print(node, model.nodeDisp(node))
    
    
    A_col     = math.pi * (D_total/2.0)**2 

    #  fc * A
    P_col_cap = fc_conf * A_col   #  kips

    # 10% 
    P_grav_total = 0.05 * P_col_cap         # kips
    P_per_col    = P_grav_total / 2.0      # kips

    # 
    g = units.gravity   # in/s^2
    m_per_node = P_per_col / g             # kip / (in/s^2) 

    # 
    for nd in [2, 3, 5]:
        # mass(MX, MY, MZ, RX, RY, RZ)
        model.mass(nd, (m_per_node/90, m_per_node/90, m_per_node/10, 0.0, 0.0, 0.0)) #909010

    # Plain + Constant
    model.pattern("Plain", 1, "Constant")
    # for nd in [2, 3, 5]:
    #     model.load(nd, (0.0, 0.0, -P_per_col/2.0, 0.0, 0.0, 0.0), pattern=1)

    for nd in [2]:
        model.load(nd, (0,0,-P_grav_total,0,0,0), pattern=1)


    print("post-gravity")
    for n in output_nodes:
        print(n, model.nodeDisp(n))

    # rayleigh(alphaM, betaK, betaKinit, betaKcomm)
    model.rayleigh(0.0319, 0.0, 0.0125, 0.0)

    # # Ground motion: fault normal (x) and fault parallel (y) components
    # # Time series for the two components
    # model.timeSeries("Path", 2, values=inputx.tolist(), dt=dt, factor=1.0)
    # model.timeSeries("Path", 3, values=inputy.tolist(), dt=dt, factor=1.0)

    # # Uniform excitation patterns in global X and Y directions
    # # pattern("UniformExcitation", tag, dof, accel=seriesTag)
    # model.pattern("UniformExcitation", 2, 1, accel=2)   # dof 1 = global X
    # model.pattern("UniformExcitation", 3, 2, accel=3)   # dof 2 = global Y

    return model

#%%
def create_bridge_model(elastic: bool = True, girder: str = "elasticBeamColumn"):
    
    # #input check
    # if np.all(inputx is None) or np.all(inputy is None) or dt is None:
    #     raise ValueError("Missing inputx, inputy, or dt. Exiting.")

    # if girder != "elasticBeamColumn":
    #     raise ValueError("Only elasticBeamColumn allowed for girders.")
    
    model = xara.Model(ndm=3, ndf=6)

    if not hasattr(model, "meta") or not isinstance(model.meta, dict):
        model.meta = {}
    model.meta["column_elems"] = []

    # Nodes: (tag, (x, y, z))
    model.node(0, (0.0,                                  0.0,                  320.0)) # abutment 1
    model.node(1, (3180.0,                               0.0,                  320.0)) # abutment 2
    model.node(2, (1752.0,                               0.0,                  320.0)) # mid-deck
    model.node(3, (1641.4628263430573,    199.41297159396336,                  320.0)) # top of column 1
    model.node(4, (1641.4628263430573,    199.41297159396336,                    0.0)) # bottom of column 1
    model.node(5, (1862.5371736569432,   -199.41297159396336,                  320.0)) # top of column 2
    model.node(6, (1862.5371736569432,   -199.41297159396336,                    0.0)) # bottom of column 2

    model.node(9, (54, 0.0, 320.0))  
    model.node(10, (3102, 0.0, 320.0))  

    # Boundary conditions, fully fixed at 0, 1, 4, 6
    # fix(tag, (DX, DY, DZ, RX, RY, RZ))
    model.fix(0, (1, 1, 1, 1, 1, 1))
    model.fix(1, (1, 1, 1, 1, 1, 1))
    model.fix(4, (1, 1, 1, 1, 1, 1))
    model.fix(6, (1, 1, 1, 1, 1, 1))

    # Materials: concrete and steel

    # Concrete strengths (ksi)
    fc_unconf = 4.0   # unconfined concrete
    fc_conf   = 5.0  # confined concrete

    # Concrete modulus (ksi), same formula as your frame model
    Ec = 57000.0 * math.sqrt(fc_unconf * 1000.0) / 1000.0

    # Steel properties
    fy = 60.0   # ksi
    Es = 30000.0

    if not elastic:
        # Nonlinear concrete (core and cover) using Concrete01
        #                    tag  f'c       epsc0         f'cu   epscu
        model.uniaxialMaterial("Concrete01", 1, -fc_conf,   -2*fc_conf/Ec,  -3.5,  -0.02)
        model.uniaxialMaterial("Concrete01", 2, -fc_unconf, -2*fc_unconf/Ec, 0.0,  -0.006)

        # Nonlinear reinforcing steel
        #                    tag fy   E0   b
        model.uniaxialMaterial("Steel01", 3, fy, Es, 0.02)
    else:
        # Elastic concrete for both core and cover
        model.uniaxialMaterial("Elastic", 1, Ec)
        model.uniaxialMaterial("Elastic", 2, Ec)

        # Elastic steel
        model.uniaxialMaterial("Elastic", 3, Es)

    
    # Section properties: 5 ft circular section
    # Geometry of circular section
    D_total = 60 #60.0        # total diameter in inches (5 ft)
    cover   = 2.0         # concrete cover in inches
    R_ext   = D_total / 2.0
    R_core  = R_ext - cover  # approximate core radius

    # mesh subdivisions for nonlinear fiber section
    numSubdivCirc = 32
    numSubdivRad  = 5
    divs = (numSubdivCirc, numSubdivRad)

    # tags
    
    colSec_fiber   = 1
    beamSec_elastic = 2   # beams stay elastic

    # ELASTIC SECTION (for elasticBeamColumn model)
    A_el = math.pi * R_ext**2
    I_el = math.pi * R_ext**4 / 4.0
    J_el = math.pi * R_ext**4 / 2.0
    nu = 0.2
    Gc = Ec / (2*(1+nu))
    GJ   = Gc * J_el
    
    model.section("Fiber", colSec_fiber, "-GJ", GJ)
    itg_col = 1
    npts_col = 4
    model.beamIntegration("Lobatto", itg_col, colSec_fiber, npts_col)

    numSubdivCirc, numSubdivRad = divs

    # core concrete
    model.patch("circ",
                1,                      # matTag = 1
                numSubdivCirc, numSubdivRad,
                0.0, 0.0,               # yCenter, zCenter
                0.0, R_core,            # intRad, extRad
                0.0, 2 * math.pi)

    # cover concrete
    model.patch("circ",
                2,                      # matTag = 2
                numSubdivCirc, numSubdivRad,
                0.0, 0.0,
                R_core, R_ext,
                0.0, 2 * math.pi)

    # longitudinal steel
    numBars = 36 # 36
    barArea = 1.56 #1.56                    # #11 bar area
    model.layer("circ",
                3,                    # steel matTag
                numBars, barArea,
                0.0, 0.0,            # yCenter, zCenter
                R_core,              # radius
                0.0, 2 * math.pi)
   

    beam_stiff_factor = 5.0
    A_beam = 864.0          # 24in × 36in = 864 in^2
    I_beam = 9.33e4 * beam_stiff_factor        # in^4  (strong axis)
    J_beam = 3.73e5 * beam_stiff_factor        # in^4  approximate torsion

    model.section("Elastic", beamSec_elastic, Ec, A_beam, I_beam, I_beam, Gc, J_beam)

    

    # Transformations and elements
    colTransf  = 1
    beamTransf = 2
    model.geomTransf("Linear", colTransf,  (1.0, 0.0, 0.0))
    model.geomTransf("Linear", beamTransf, 0.0, 0.0, 1.0)

    # columns: elastic vs nonlinear
    
    # columns as elasticBeamColumn with elastic section
    col_type = "forceBeamColumn"
    sec_col  = colSec_fiber

    model.element(col_type, 2, (4, 3), transform=colTransf, section=sec_col, shear=0)
    model.element(col_type, 3, (6, 5), transform=colTransf, section=sec_col, shear=0)
    

    model.meta["column_elems"] = [2, 3]

    # beams always elastic
    beam_type = "elasticBeamColumn"
    sec_beam  = beamSec_elastic

    
    model.element(beam_type, 101, (0, 9), transform=beamTransf, section=sec_beam, shear=0)
    
    model.element(beam_type, 102, (9, 2),  transform=beamTransf, section=sec_beam, shear=0)
    
    model.element(beam_type, 103, (2, 10), transform=beamTransf, section=sec_beam, shear=0)
    model.element(beam_type, 104, (10, 1), transform=beamTransf, section=sec_beam, shear=0)

    model.element(beam_type, 105, (3, 2), transform=beamTransf, section=sec_beam, shear=0)
    model.element(beam_type, 106, (5, 2), transform=beamTransf, section=sec_beam, shear=0)

    # model.element(beam_type, 1, (0, 2), transform=beamTransf, section=sec_beam, shear=0)
    # model.element(beam_type, 4, (2, 1), transform=beamTransf, section=sec_beam, shear=0)
    # model.element(beam_type, 5, (3, 2), transform=beamTransf, section=sec_beam, shear=0)
    # model.element(beam_type, 6, (5, 2), transform=beamTransf, section=sec_beam, shear=0)

    # Mass, damping and earthquake excitation
    # lump_mass = 1.0  # placeholder
    # for nd in [2, 3, 5]:
    #     # mass(tag, (MX, MY, MZ, RX, RY, RZ))
    #     model.mass(nd, (lump_mass, lump_mass, 0.0, 0.0, 0.0, 0.0))

    # ---------- Gravity load and mass: template style ----------
    output_nodes = [2,3,5]
    print('pre-gravity disp')
    for node in output_nodes:
        print(node, model.nodeDisp(node))
    
    
    A_col     = math.pi * (D_total/2.0)**2 

    #  fc * A
    P_col_cap = fc_conf * A_col   #  kips

    # 10% 
    P_grav_total = 0.05 * P_col_cap         # kips
    P_per_col    = P_grav_total / 2.0      # kips

    # 
    g = units.gravity   # in/s^2
    m_per_node = P_per_col / g             # kip / (in/s^2) 

    # 
    for nd in [2, 3, 5]:
        # mass(MX, MY, MZ, RX, RY, RZ)
        model.mass(nd, (m_per_node/90, m_per_node/90, m_per_node/10, 0.0, 0.0, 0.0)) #909010

    # Plain + Constant
    model.pattern("Plain", 1, "Constant")
    # for nd in [2, 3, 5]:
    #     model.load(nd, (0.0, 0.0, -P_per_col/2.0, 0.0, 0.0, 0.0), pattern=1)

    for nd in [2, 3, 5]:
        model.load(nd, (0,0,-P_grav_total,0,0,0), pattern=1)


    print("post-gravity")
    for n in output_nodes:
        print(n, model.nodeDisp(n))

    # rayleigh(alphaM, betaK, betaKinit, betaKcomm)
    model.rayleigh(0.0319, 0.0, 0.0125, 0.0)

    # # Ground motion: fault normal (x) and fault parallel (y) components
    # # Time series for the two components
    # model.timeSeries("Path", 2, values=inputx.tolist(), dt=dt, factor=1.0)
    # model.timeSeries("Path", 3, values=inputy.tolist(), dt=dt, factor=1.0)

    # # Uniform excitation patterns in global X and Y directions
    # # pattern("UniformExcitation", tag, dof, accel=seriesTag)
    # model.pattern("UniformExcitation", 2, 1, accel=2)   # dof 1 = global X
    # model.pattern("UniformExcitation", 3, 2, accel=3)   # dof 2 = global Y

    return model

#%% plate

def create_bridge_model(elastic: bool = True, girder: str = "elasticBeamColumn"):
    
    # #input check
    # if np.all(inputx is None) or np.all(inputy is None) or dt is None:
    #     raise ValueError("Missing inputx, inputy, or dt. Exiting.")

    # if girder != "elasticBeamColumn":
    #     raise ValueError("Only elasticBeamColumn allowed for girders.")
    
    model = xara.Model(ndm=3, ndf=6)

    if not hasattr(model, "meta") or not isinstance(model.meta, dict):
        model.meta = {}
    model.meta["column_elems"] = []

    # # Nodes: (tag, (x, y, z))
    # model.node(0, (0.0,                                  0.0,                  320.0)) # abutment 1
    # model.node(1, (3180.0,                               0.0,                  320.0)) # abutment 2
    # model.node(2, (1752.0,                               0.0,                  320.0)) # mid-deck
    # model.node(3, (1641.4628263430573,    199.41297159396336,                  320.0)) # top of column 1
    # model.node(4, (1641.4628263430573,    199.41297159396336,                    0.0)) # bottom of column 1
    # model.node(5, (1862.5371736569432,   -199.41297159396336,                  320.0)) # top of column 2
    # model.node(6, (1862.5371736569432,   -199.41297159396336,                    0.0)) # bottom of column 2

    # model.node(7, (1641.4628263430573, 0.0, 320.0))  # centerline node under col top 3
    # model.node(8, (1862.5371736569432, 0.0, 320.0))  # centerline node under col top 5

    # model.node(9, (54, 0.0, 320.0))  
    # model.node(10, (3102, 0.0, 320.0))

    y_off = 199.41297159396336

    # keep original centerline abutment nodes if you want, but DO NOT connect deck to them
    model.node(0, (0.0,    0.0, 320.0))
    model.node(1, (3180.0, 0.0, 320.0))

    # mid-deck centerline node (optional output only, not connected in scheme A unless you add it)
    # model.node(2, (1752.0, 0.0, 320.0))

    # column top/bottom
    model.node(3, (1641.4628263430573,  y_off, 320.0))  # top col 1 (+y)
    model.node(4, (1641.4628263430573,  y_off,   0.0))  # bot col 1
    model.node(5, (1862.5371736569432, -y_off, 320.0))  # top col 2 (-y)
    model.node(6, (1862.5371736569432, -y_off,   0.0))  # bot col 2

    # girder abutment nodes on two lines (scheme A)
    model.node(11, (0.0,    y_off, 320.0))   # left, +y girder
    model.node(12, (0.0,   -y_off, 320.0))   # left, -y girder
    model.node(13, (3180.0, y_off, 320.0))   # right, +y girder
    model.node(14, (3180.0,-y_off, 320.0))   # right, -y girder  

    # Boundary conditions, fully fixed at 0, 1, 4, 6
    # fix(tag, (DX, DY, DZ, RX, RY, RZ))
    model.fix(0, (1, 1, 1, 1, 1, 1))
    model.fix(1, (1, 1, 1, 1, 1, 1))
    model.fix(4, (1, 0, 1, 1, 1, 1))
    model.fix(6, (1, 0, 1, 1, 1, 1))

    # model.fix(11, (1, 1, 1, 1, 1, 1))
    # model.fix(12, (1, 1, 1, 1, 1, 1))
    # model.fix(13, (1, 1, 1, 1, 1, 1))
    # model.fix(14, (1, 1, 1, 1, 1, 1))

    model.fix(11, (1, 1, 1, 0, 0, 0))
    model.fix(12, (0, 1, 1, 0, 0, 0))
    model.fix(13, (0, 1, 1, 0, 0, 0))
    model.fix(14, (0, 0, 1, 0, 0, 0))

    # Materials: concrete and steel

    # Concrete strengths (ksi)
    fc_unconf = 4.0   # unconfined concrete
    fc_conf   = 5.0  # confined concrete

    # Concrete modulus (ksi), same formula as your frame model
    Ec = 57000.0 * math.sqrt(fc_unconf * 1000.0) / 1000.0

    # Steel properties
    fy = 60.0   # ksi
    Es = 30000.0

    if not elastic:
        # Nonlinear concrete (core and cover) using Concrete01
        #                    tag  f'c       epsc0         f'cu   epscu
        model.uniaxialMaterial("Concrete01", 1, -fc_conf,   -2*fc_conf/Ec,  -3.5,  -0.02)
        model.uniaxialMaterial("Concrete01", 2, -fc_unconf, -2*fc_unconf/Ec, 0.0,  -0.006)

        # Nonlinear reinforcing steel
        #                    tag fy   E0   b
        model.uniaxialMaterial("Steel01", 3, fy, Es, 0.02)
    else:
        # Elastic concrete for both core and cover
        model.uniaxialMaterial("Elastic", 1, Ec)
        model.uniaxialMaterial("Elastic", 2, Ec)

        # Elastic steel
        model.uniaxialMaterial("Elastic", 3, Es)

    
    # Section properties: 5 ft circular section
    # Geometry of circular section
    D_total = 60 #60.0        # total diameter in inches (5 ft)
    cover   = 2.0         # concrete cover in inches
    R_ext   = D_total / 2.0
    R_core  = R_ext - cover  # approximate core radius

    # mesh subdivisions for nonlinear fiber section
    numSubdivCirc = 32
    numSubdivRad  = 5
    divs = (numSubdivCirc, numSubdivRad)

    # tags
    
    colSec_fiber   = 1
    beamSec_elastic = 2   # beams stay elastic

    # ELASTIC SECTION (for elasticBeamColumn model)
    A_el = math.pi * R_ext**2
    I_el = math.pi * R_ext**4 / 4.0
    J_el = math.pi * R_ext**4 / 2.0
    nu = 0.2
    Gc = Ec / (2*(1+nu))
    GJ   = Gc * J_el
    
    model.section("Fiber", colSec_fiber, "-GJ", GJ)
    itg_col = 1
    npts_col = 4
    model.beamIntegration("Lobatto", itg_col, colSec_fiber, npts_col)

    numSubdivCirc, numSubdivRad = divs

    # core concrete
    model.patch("circ",
                1,                      # matTag = 1
                numSubdivCirc, numSubdivRad,
                0.0, 0.0,               # yCenter, zCenter
                0.0, R_core,            # intRad, extRad
                0.0, 2 * math.pi)

    # cover concrete
    model.patch("circ",
                2,                      # matTag = 2
                numSubdivCirc, numSubdivRad,
                0.0, 0.0,
                R_core, R_ext,
                0.0, 2 * math.pi)

    # longitudinal steel
    numBars = 36 # 36
    barArea = 1.56 #1.56                    # #11 bar area
    model.layer("circ",
                3,                    # steel matTag
                numBars, barArea,
                0.0, 0.0,            # yCenter, zCenter
                R_core,              # radius
                0.0, 2 * math.pi)
   

    beam_stiff_factor = 5.0
    A_beam = 864.0          # 24in × 36in = 864 in^2
    I_beam = 9.33e4 * beam_stiff_factor        # in^4  (strong axis)
    J_beam = 3.73e5 * beam_stiff_factor        # in^4  approximate torsion

    model.section("Elastic", beamSec_elastic, Ec, A_beam, I_beam, I_beam, Gc, J_beam)

    

    # Transformations and elements
    colTransf  = 1
    beamTransf = 2
    model.geomTransf("Linear", colTransf,  (1.0, 0.0, 0.0))
    model.geomTransf("Linear", beamTransf, 0.0, 0.0, 1.0)

    # columns: elastic vs nonlinear
    
    # columns as elasticBeamColumn with elastic section
    col_type = "forceBeamColumn"
    sec_col  = colSec_fiber

    model.element(col_type, 2, (4, 3), transform=colTransf, section=sec_col, shear=0)
    model.element(col_type, 3, (6, 5), transform=colTransf, section=sec_col, shear=0)
    

    model.meta["column_elems"] = [2, 3]

    # beams always elastic
    beam_type = "elasticBeamColumn"
    sec_beam  = beamSec_elastic

    # replace your deck elements (0-2 and 2-1) with these
    # model.element(beam_type, 101, (0, 9), transform=beamTransf, section=sec_beam, shear=0)
    # model.element(beam_type, 102, (9, 7),  transform=beamTransf, section=sec_beam, shear=0)
    # model.element(beam_type, 103, (7, 2),  transform=beamTransf, section=sec_beam, shear=0)
    # model.element(beam_type, 104, (2, 8),  transform=beamTransf, section=sec_beam, shear=0)
    # model.element(beam_type, 105, (8, 10), transform=beamTransf, section=sec_beam, shear=0)
    # model.element(beam_type, 106, (10, 1), transform=beamTransf, section=sec_beam, shear=0)

    # model.element(beam_type, 1, (0, 2), transform=beamTransf, section=sec_beam, shear=0)
    # model.element(beam_type, 4, (2, 1), transform=beamTransf, section=sec_beam, shear=0)
    # model.element(beam_type, 5, (3, 2), transform=beamTransf, section=sec_beam, shear=0)
    # model.element(beam_type, 6, (5, 2), transform=beamTransf, section=sec_beam, shear=0)

    # model.rigidLink("beam", 7, 3)  # master=7 (centerline), slave=3 (col top)
    # model.rigidLink("beam", 8, 5)

    # +y girder: 11 -- 3 -- 13
    model.element(beam_type, 201, (11, 3), transform=beamTransf, section=sec_beam, shear=0)
    model.element(beam_type, 202, (3, 13), transform=beamTransf, section=sec_beam, shear=0)

    # -y girder: 12 -- 5 -- 14
    model.element(beam_type, 203, (12, 5), transform=beamTransf, section=sec_beam, shear=0)
    model.element(beam_type, 204, (5, 14), transform=beamTransf, section=sec_beam, shear=0)

    # diaphragms / cross-beams tying the two girders together
    model.element(beam_type, 301, (11, 12), transform=beamTransf, section=sec_beam, shear=0)
    model.element(beam_type, 302, (3,  5),  transform=beamTransf, section=sec_beam, shear=0)
    model.element(beam_type, 303, (13, 14), transform=beamTransf, section=sec_beam, shear=0)


    # Mass, damping and earthquake excitation
    # lump_mass = 1.0  # placeholder
    # for nd in [2, 3, 5]:
    #     # mass(tag, (MX, MY, MZ, RX, RY, RZ))
    #     model.mass(nd, (lump_mass, lump_mass, 0.0, 0.0, 0.0, 0.0))

    # ---------- Gravity load and mass: template style ----------
    output_nodes = [3,5]
    print('pre-gravity disp')
    for node in output_nodes:
        print(node, model.nodeDisp(node))
    
    
    A_col     = math.pi * (D_total/2.0)**2 

    #  fc * A
    P_col_cap = fc_conf * A_col   #  kips

    # 10% 
    P_grav_total = 0.05 * P_col_cap         # kips
    P_per_col    = P_grav_total / 2.0      # kips

    # 
    g = units.gravity   # in/s^2
    m_per_node = P_per_col / g             # kip / (in/s^2) 

    # 
    for nd in [3, 5]:
        # mass(MX, MY, MZ, RX, RY, RZ)
        model.mass(nd, (m_per_node/90, m_per_node/90, m_per_node/10, 0.0, 0.0, 0.0)) #909010

    # Plain + Constant
    model.pattern("Plain", 1, "Constant")
    # for nd in [2, 3, 5]:
    #     model.load(nd, (0.0, 0.0, -P_per_col/2.0, 0.0, 0.0, 0.0), pattern=1)

    for nd in [3, 5]:
        model.load(nd, (0,0,-P_grav_total,0,0,0), pattern=1)


    print("post-gravity")
    for n in output_nodes:
        print(n, model.nodeDisp(n))

    # rayleigh(alphaM, betaK, betaKinit, betaKcomm)
    model.rayleigh(0.0319, 0.0, 0.0125, 0.0)

    # # Ground motion: fault normal (x) and fault parallel (y) components
    # # Time series for the two components
    # model.timeSeries("Path", 2, values=inputx.tolist(), dt=dt, factor=1.0)
    # model.timeSeries("Path", 3, values=inputy.tolist(), dt=dt, factor=1.0)

    # # Uniform excitation patterns in global X and Y directions
    # # pattern("UniformExcitation", tag, dof, accel=seriesTag)
    # model.pattern("UniformExcitation", 2, 1, accel=2)   # dof 1 = global X
    # model.pattern("UniformExcitation", 3, 2, accel=3)   # dof 2 = global Y

    return model