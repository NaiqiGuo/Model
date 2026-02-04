import xara
import xara.units.iks as units
import math
import numpy as np
import pandas as pd
import csv
import os
import cvxpy as cp 
import plotly.graph_objects as go
from pathlib import Path


def create_bridge_model1111(elastic: bool = True, girder: str = "elasticBeamColumn"):
    
    model = xara.Model(ndm=3, ndf=6)

    if not hasattr(model, "meta") or not isinstance(model.meta, dict):
        model.meta = {}
    model.meta["column_elems"] = []

    # # Nodes: (tag, (x, y, z))
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
    model.fix(4, (1, 1, 1, 1, 1, 1))
    model.fix(6, (1, 1, 1, 1, 1, 1))

    # model.fix(11, (1, 1, 1, 1, 1, 1))
    # model.fix(12, (1, 1, 1, 1, 1, 1))
    # model.fix(13, (1, 1, 1, 1, 1, 1))
    # model.fix(14, (1, 1, 1, 1, 1, 1))

    # ----------------------------
    # Bearing springs at abutments
    # ----------------------------

    # 1) remove full fix at 11-14: do NOT call model.fix on 11..14
    #    (keep column bases 4,6 fixed)
    # 2) create "ground" nodes coincident with bearing nodes
    ground_map = {11: 1111, 12: 1212, 13: 1313, 14: 1414}
    for n, gnd in ground_map.items():
        # copy coordinates manually (since xara nodeCoord may not exist)
        # you already know these coordinates:
        if n == 11:
            model.node(gnd, (0.0,    y_off, 320.0))
        elif n == 12:
            model.node(gnd, (0.0,   -y_off, 320.0))
        elif n == 13:
            model.node(gnd, (3180.0, y_off, 320.0))
        elif n == 14:
            model.node(gnd, (3180.0,-y_off, 320.0))
        # fully fix ground nodes
        model.fix(gnd, (1,1,1,1,1,1))
    # 3) define uniaxial spring materials (Elastic)
    # stiffness units: kip/in
    kV      = 1e8   # vertical
    kT      = 1e5   # transverse (Y)
    kL_fix  = 1e6   # longitudinal at fixed end (X)
    kL_exp  = 1e3   # longitudinal at expansion end (X)

    matX_fix = 2011
    matX_exp = 2012
    matY     = 2013
    matZ     = 2014
    model.uniaxialMaterial("Elastic", matX_fix, kL_fix)
    model.uniaxialMaterial("Elastic", matX_exp, kL_exp)
    model.uniaxialMaterial("Elastic", matY,     kT)
    model.uniaxialMaterial("Elastic", matZ,     kV)
    # 4) add zeroLength elements. Each bearing node connected to its ground node
    # left end (x=0): choose as FIXED end in X
    # right end (x=3180): choose as EXPANSION end in X
    # dofs in OpenSees: 1=X, 2=Y, 3=Z
    ele = 9000

    # left +y (11)
    model.element("zeroLength", ele, (11, ground_map[11]),
                "-mat", (matX_fix, matY, matZ),
                "-dir", (1, 2, 3))
    ele += 1
    # left -y (12)
    model.element("zeroLength", ele, (12, ground_map[12]),
                "-mat", (matX_fix, matY, matZ),
                "-dir", (1, 2, 3))
    ele += 1
    # right +y (13): expansion in X
    model.element("zeroLength", ele, (13, ground_map[13]),
                "-mat", (matX_exp, matY, matZ),
                "-dir", (1, 2, 3))
    ele += 1
    # right -y (14): expansion in X
    model.element("zeroLength", ele, (14, ground_map[14]),
                "-mat", (matX_exp, matY, matZ),
                "-dir", (1, 2, 3))
    ele += 1


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
        model.mass(nd, (m_per_node/90, m_per_node/90, m_per_node/90, 0.0, 0.0, 0.0)) #909010

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
    return model

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
    model.fix(4, (1, 1, 1, 1, 1, 1))
    model.fix(6, (1, 1, 1, 1, 1, 1))

    model.fix(11, (1, 1, 1, 1, 1, 1))
    model.fix(12, (1, 1, 1, 1, 1, 1))
    model.fix(13, (1, 1, 1, 1, 1, 1))
    model.fix(14, (1, 1, 1, 1, 1, 1))

    # model.fix(11, (1, 1, 1, 0, 0, 0))
    # model.fix(12, (0, 1, 1, 0, 0, 0))
    # model.fix(13, (0, 1, 1, 0, 0, 0))
    # model.fix(14, (0, 0, 1, 0, 0, 0))

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
   

    beam_stiff_factor = 24.0
    # A_beam = 864.0          # 24in × 36in = 864 in^2
    # I_beam = 9.33e4 * beam_stiff_factor        # in^4  (strong axis)
    # J_beam = 3.73e5 * beam_stiff_factor        # in^4  approximate torsion

    A_beam = 1600.0          # 40in × 40in = 1600 in^2
    I_beam = 2.13e5 * beam_stiff_factor        # in^4  (strong axis)
    J_beam = 3.60e5 * beam_stiff_factor        # in^4  approximate torsion

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
        model.mass(nd, (m_per_node/5, m_per_node/5, m_per_node/5, 0.0, 0.0, 0.0)) #909010

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


def create_bridge_model1(elastic: bool = True, girder: str = "elasticBeamColumn"):
    
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
    model.fix(4, (1, 0, 1, 1, 1, 1))
    model.fix(6, (1, 0, 1, 1, 1, 1))

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
    for nd in [2, 3, 5]:
        model.load(nd, (0.0, 0.0, -P_per_col/2.0, 0.0, 0.0, 0.0), pattern=1)

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

def apply_load_bridge(model, inputx=None, inputy=None, dt=None):
    """
    Add dynamic loads to bridge model
    """
    if np.all(inputx is None) or np.all(inputy is None) or dt is None:
        raise ValueError("Missing inputx, inputy, or dt. Exiting.")
    
    # Define earthquake excitation
    # ----------------------------
    # Set up the acceleration records for fault normal (x, dof 1) and fault parallel (y, dof 2)
    model.timeSeries("Path", 2, values=inputx.tolist(), dt=dt, factor=1.0)
    model.timeSeries("Path", 3, values=inputy.tolist(), dt=dt, factor=1.0)

    # Define the excitation using the given ground motion records
    #                         tag dir         accel series args
    model.pattern("UniformExcitation", 2, 1, accel=2)
    model.pattern("UniformExcitation", 3, 2, accel=3)

    return model



def save_event_io(i, inputs, outputs, dt, out_dir="event_data"):
    os.makedirs(out_dir, exist_ok=True)
    nt = inputs.shape[1]
    time = np.arange(nt) * dt
    
    csv_in = os.path.join(out_dir, f"event_{i+1}_inputs.csv")
    with open(csv_in, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["time","acc_X","acc_Y"])
        for t, x, y in zip(time, inputs[0], inputs[1]):
            writer.writerow([t, x, y])
    
    csv_out = os.path.join(out_dir, f"event_{i+1}_outputs.csv")
    with open(csv_out, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["time","1F_X","1F_Y","2F_X","2F_Y","3F_X","3F_Y"])
        for idx in range(nt):
            row = [time[idx]] + [outputs[r,idx] for r in range(6)]
            writer.writerow(row)




def apply_gravity_static(
    model,
    output_nodes,
    fixed_nodes=None,
    tol=1e-8,
    max_iter=50,
    n_steps=10,
):
    """
    Solve gravity equilibrium (static), then fix gravity loads as constant.
    Returns model (in-place).
    """

    if fixed_nodes is None:
        fixed_nodes = []  # [0,1,4,6] for bridge, [1,2,3,4] for frame

    print("pre-gravity disp")
    for n in output_nodes:
        print(n, model.nodeDisp(n))

    # fresh analysis objects
    model.wipeAnalysis()

    # static analysis setup
    model.system("BandGen")
    model.numberer("RCM")
    model.constraints("Transformation")
    model.test("NormDispIncr", tol, max_iter)
    model.algorithm("Newton")
    model.integrator("LoadControl", 1.0 / n_steps)
    model.analysis("Static")

    # apply gravity gradually
    for k in range(n_steps):
        ok = model.analyze(1)
        if ok != 0:
            raise RuntimeError(f"gravity static failed at step {k+1}/{n_steps}")

    # check reactions 
    if fixed_nodes:
        try:
            model.reactions()
            print("gravity reactions at fixed nodes")
            for nd in fixed_nodes:
                print(nd, model.nodeReaction(nd))
        except Exception as e:
            print("reaction check failed:", e)

    print("post-gravity disp")
    for n in output_nodes:
        print(n, model.nodeDisp(n))

    # #check
    # check_nodes = [0,1,2,3,4,5,6,9,10]
    # print("\n[gravity check] nodeDisp for key nodes")
    # for n in check_nodes:
    #     print(n, model.nodeDisp(n))
    # print("\n[gravity check] du = top - base")
    # for top, base in [(3,4),(5,6)]:
    #     uT = model.nodeDisp(top)
    #     uB = model.nodeDisp(base)
    #     du = [uT[i]-uB[i] for i in range(6)]
    #     print(f"{top}-{base} du_XY=({du[0]:+.6e}, {du[1]:+.6e}), du_Z={du[2]:+.6e}, du_R=({du[3]:+.6e},{du[4]:+.6e},{du[5]:+.6e})")
    

    # fix gravity loads as constant and reset time
    model.loadConst("-time", 0.0)

    # wipe analysis again so transient can be configured cleanly
    model.wipeAnalysis()

    return model


def stabilize_with_lmi(A_hat, epsilon=1e-10, solver='CVXOPT'):

    """
    Only the matrix A_s obtained from algorithms such as SRIM is stabilized by imposing a Lyapunov LMI constraint to obtain a stable A. 
    The matrices B_s, C_s, and D_s are directly used as output. 
    This corresponds to Section II of Lacy & Bernstein (2003).
    The solver can be set to either SCS or CVXOPT, and epsilon can be chosen starting from 1e-8.
    """
    n = A_hat.shape[0]

    # #Define only Q and P as variables
    P = cp.Variable((n, n), PSD=True)
    Q = cp.Variable((n, n))

    # —— Objective function ——— (Equation 2.17)
    objective = cp.Minimize(cp.norm(A_hat @ P - Q, "fro")**2)

    # —— Lyapunov LMI constraint ——— (Equation 2.18)
    M = cp.bmat([
        [P - epsilon * np.eye(n), Q],
        [Q.T, P]
    ])
    constraints = [M >> 0]

    # —— Solve the optimization problem —— 
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver)

    A_lmi = Q.value @ np.linalg.inv(P.value)
    return A_lmi


def stabilize_by_radius_clipping(A, alpha=0.995, rmin=None, make_real=True):
    w, V = np.linalg.eig(A)             
    r = np.abs(w)
    w_new = w.copy()
    over = r > 1.0
    w_new[over] = alpha * w[over] / r[over]
    if rmin is not None:
        under = np.abs(w_new) < rmin
        w_new[under] = rmin * w_new[under] / (np.abs(w_new[under]) + 1e-15)
    A_new = V @ np.diag(w_new) @ np.linalg.inv(V)
    if make_real and np.allclose(A_new.imag, 0, atol=1e-10):
        A_new = A_new.real
    return A_new


def get_true_modes_xara(model, floor_nodes=(9,14,19), dofs=(1,2), n=3, solver='-genBandArpack'): #-symmBandLapack -genBandArpack
    lambdas = model.eigen(n)  
    print(f"[Debug] eigen(n={n}) returned {len(lambdas)} values: {lambdas}")
    lambdas = np.asarray(lambdas, dtype=float)
    omega = np.sqrt(np.abs(lambdas))                    # rad/s
    freqs_hz = omega / (2*np.pi)
    rows = []
    for node in floor_nodes:
        for dof in dofs:
            rows.append([model.nodeEigenvector(node, k+1, dof) for k in range(n)])
    Phi_true = np.array(rows, dtype=float)              # (6, nmodes)
    Phi_true /= (np.linalg.norm(Phi_true, axis=0, keepdims=True) + 1e-12)
    idx = np.argsort(freqs_hz)         
    freqs_hz = freqs_hz[idx]
    Phi_true = Phi_true[:, idx]       
    return freqs_hz, Phi_true


def mac_matrix(Phi_true, Phi_est):
    print(type(Phi_est))
    print(np.shape(Phi_est))
    T = Phi_true / (np.linalg.norm(Phi_true, axis=0, keepdims=True) + 1e-12)
    E = Phi_est  / (np.linalg.norm(Phi_est,  axis=0, keepdims=True) + 1e-12)
    return np.abs(T.T @ E)**2     


def normalize_v(v):
    """
    Normalize an individual vector:
    element-wise signed complex magnitude, unitized
    i.e.:
    for each element, multiply the sign of its real part
    by the square root of the element multiplied by its
    complex conjugate.
    then, divide the vector by its Euclidean norm, or the
    square root of the sum of the elements squared.
    """
    vabs = np.abs(v)
    signed_vabs = np.sign(np.real(v)) * vabs
    normed_vabs = signed_vabs / np.linalg.norm(signed_vabs)
    return normed_vabs


def normalize_Psi(Psi):
    """
    Normalize a matrix of complex column vectors using
    `normalize_v` on each vector
    """
    normed_Psi = np.zeros(Psi.shape)
    for i in range(Psi.shape[1]):
        v = Psi[:,i]
        normed_Psi[:,i] = normalize_v(v)
    return normed_Psi


def phi_output(A, C):
    eigvals, U = np.linalg.eig(np.asarray(A, dtype=complex))
    angles = np.abs(np.angle(eigvals))
    idx = np.where(
        (np.abs(eigvals) < 1 - 1e-10) & 
        #(np.abs(eigvals) > 0.05) & 
        (np.imag(eigvals) > -1e-12)&
        (angles > 0.02)
        )[0]
    eigvals_sel = eigvals[idx]
    U_sel = U[:, idx]
    V = C @ U_sel
    Phi = normalize_Psi(V)  
    return Phi, eigvals_sel


def periods_from_A(A, dt):
    eigvals = np.linalg.eigvals(A)
    idx = np.abs(eigvals) < 1.0 
    eigvals = eigvals[idx]
    omega = np.abs(np.angle(eigvals)) / dt  # rad/s
    freqs = omega / (2 * np.pi)
    periods = 1 / freqs
    return np.sort(periods)


def save_event_modes_to_csv(event_id, Phi_true, method_modes, method_macs, algos, filename):
    # method_modes: {'srim': Phi_srim, ...}  Phi, shape=(dof, n_modes)
    # method_macs:  {'srim': MAC_srim, ...}  MAC,  shape=(n_modes, n_modes)
    with open(filename, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([f"Event {event_id} True Mode Shapes"])
        for i in range(Phi_true.shape[1]):
            writer.writerow([f"True Mode {i+1}"] + list(Phi_true[:, i]))

        for algo in algos:
            writer.writerow([f"{algo.upper()} Mode Shapes"])
            Phi = method_modes[algo]
            for i in range(Phi.shape[1]):
                writer.writerow([f"{algo.upper()} Mode {i+1}"] + list(Phi[:, i]))

        for algo in algos:
            writer.writerow([f"{algo.upper()} MAC vs True"])
            MAC = method_macs[algo]
            for row in MAC:
                writer.writerow([""] + list(row))
        
        writer.writerow([])  


def plot_q4_max_strain(sr, model, title, html_base):
    """
    sr = model.meta["strain_record"]
    Generate the time history curve of the maximum edge concrete strain/reinforcement strain (nonlinear) or edge strain (elastic).
    """

    t = np.asarray(sr["time"], float)
    cols = model.meta["column_elems"]

    if not sr.get("inelastic", False):
        # ELASTIC: eps_edge = eps0 + kappa*h/2
        h = float(sr.get("section_depth", np.nan))
        stack = []
        for ele in cols:
            eps0  = np.asarray(sr["eps0"][ele],  float)
            kappa = np.asarray(sr["kappa"][ele], float)
            edge  = eps0 + kappa * (h/2.0)
            stack.append(edge)
        edge_max = np.nanmax(np.vstack(stack), axis=0) if stack else np.full_like(t, np.nan)

        fig = go.Figure()
        fig.add_scatter(x=t, y=edge_max, mode="lines", name="Max Edge Strain (elastic)")
        fig.update_layout(title=title, xaxis_title="Time (s)", yaxis_title="Strain")
        fig.write_html(html_base + "_elastic_edge.html", include_plotlyjs="cdn")
        return

    # INELASTIC: concrete edge max & steel max
    conc_stack, steel_stack = [], []
    for ele in cols:
        conc_stack.append(np.asarray(sr["conc_edge_max"][ele], float))
        steel_stack.append(np.asarray(sr["steel_max"][ele],     float))
    conc_max  = np.nanmax(np.vstack(conc_stack),  axis=0) if conc_stack  else np.full_like(t, np.nan)
    steel_max = np.nanmax(np.vstack(steel_stack), axis=0) if steel_stack else np.full_like(t, np.nan)

    f1 = go.Figure(); f2 = go.Figure()
    f1.add_scatter(x=t, y=conc_max,  mode="lines", name="Max Concrete Edge Strain")
    f2.add_scatter(x=t, y=steel_max, mode="lines", name="Max Steel Strain")
    f1.update_layout(title=title+" — Concrete", xaxis_title="Time (s)", yaxis_title="Strain")
    f2.update_layout(title=title+" — Steel",    xaxis_title="Time (s)", yaxis_title="Strain")
    f1.write_html(html_base + "_inelastic_conc.html",  include_plotlyjs="cdn")
    f2.write_html(html_base + "_inelastic_steel.html", include_plotlyjs="cdn")


def save_displacements(displacements, dt, filename):

    nt = len(list(displacements.values())[0])
    time = np.arange(nt) * dt

    nodes = sorted(displacements.keys())

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)

        # header: time + each node 6 dof
        dof_labels = ["UX", "UY", "UZ", "RX", "RY", "RZ"]
        header = ["time"]
        for n in nodes:
            for lbl in dof_labels:
                header.append(f"node{n}_{lbl}")
        writer.writerow(header)

        # each line: t_k + each node 6DOF
        for k in range(nt):
            row = [time[k]]
            for n in nodes:
                u = displacements[n][k]  # 6
                row.extend(u)
            writer.writerow(row)


def save_strain_stress(stresses, strains, dt, filename):

    nt = len(list(stresses.values())[0])
    time = np.arange(nt) * dt

    elems = sorted(stresses.keys())

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)

        # header: time + stress + strain
        header = ["time"]
        header += [f"ele{e}_stress" for e in elems]
        header += [f"ele{e}_strain" for e in elems]
        writer.writerow(header)

        # each line: t_k + stress + strain
        for k in range(nt):
            row = [time[k]]
            for e in elems:
                row.append(stresses[e][k])
            for e in elems:
                row.append(strains[e][k])
            writer.writerow(row)


#get damage
def list_event_dirs(case_dir: Path):
    ev_dirs = [p for p in case_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    return sorted(ev_dirs, key=lambda x: int(x.name))

def load_dt(event_dir: Path) -> float:
    p = event_dir / "dt.txt"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    return float(p.read_text().strip())

def load_inputs(event_dir: Path) -> np.ndarray:
    p = event_dir / "inputs.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    x = np.loadtxt(p, dtype=float)
    # expected shape (n_in, nt)
    if x.ndim == 1:
        x = x[None, :]
    return x

def load_strain_stress_df(event_dir: Path) -> pd.DataFrame:
    p = event_dir / "strain_stress.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    return pd.read_csv(p)

def load_freq_vector(event_dir: Path, fname: str) -> np.ndarray:
    p = event_dir / fname
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    f = np.loadtxt(p, dtype=float)
    f = np.atleast_1d(f).astype(float)
    return f

def load_pred_true_processed(event_dir: Path, sid_method: str):
    pred_dir = event_dir / sid_method
    p_true = pred_dir / "outputs_true_processed.csv"
    p_pred = pred_dir / "outputs_pred_processed.csv"
    if not p_true.exists() or not p_pred.exists():
        raise FileNotFoundError(f"Missing processed outputs under {pred_dir}")

    y_true = np.loadtxt(p_true, dtype=float)
    y_pred = np.loadtxt(p_pred, dtype=float)

    # enforce shape (n_out, nt)
    if y_true.ndim == 1:
        y_true = y_true[None, :]
    if y_pred.ndim == 1:
        y_pred = y_pred[None, :]

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}")

    return y_true, y_pred


# Baseline selection by input intensity
def compute_intensity(inputs: np.ndarray, metric: str) -> float:
    """
    inputs: (n_in, nt)
    metric:
      - "pga": max over channels of max(abs(input))
      - "rms": max over channels of rms(input)
    """
    if metric == "pga":
        return float(np.max(np.abs(inputs)))
    if metric == "rms":
        rms_each = np.sqrt(np.mean(inputs**2, axis=1))
        return float(np.max(rms_each))
    raise ValueError("BASELINE_METRIC must be 'pga' or 'rms'")

def select_baseline_events(case_dir: Path, n_baseline: int, metric: str):
    ev_dirs = list_event_dirs(case_dir)
    rows = []
    for ev_dir in ev_dirs:
        ev = int(ev_dir.name)
        inputs = load_inputs(ev_dir)
        inten = compute_intensity(inputs, metric)
        rows.append({"event": ev, "intensity": inten})
    df_int = pd.DataFrame(rows).sort_values("intensity").reset_index(drop=True)
    baseline_ids = df_int.loc[: n_baseline - 1, "event"].tolist()
    return baseline_ids, df_int

# Metric computations
def compute_ksec_for_element(df: pd.DataFrame, ele_id: int, eps_stab: float = 1e-12) -> float:
    s_col = f"ele{ele_id}_stress"
    e_col = f"ele{ele_id}_strain"
    if s_col not in df.columns or e_col not in df.columns:
        raise KeyError(f"Missing columns {s_col} or {e_col} in strain_stress.csv")

    stress = df[s_col].to_numpy(dtype=float)
    strain = df[e_col].to_numpy(dtype=float)

    idx = int(np.argmax(np.abs(stress)))
    sigma = float(stress[idx])
    eps = float(strain[idx])
    

    return abs(sigma) / (abs(eps) + eps_stab)

def compute_kref_ele(case_dir: Path, baseline_ids, elements) -> dict:
    """
    For each element:
      kref_ele = median over baseline events of ksec_ele
    """
    kref_ele = {}
    for ele in elements:
        k_list = []
        for ev in baseline_ids:
            ev_dir = case_dir / str(ev)
            df_ss = load_strain_stress_df(ev_dir)
            k_list.append(compute_ksec_for_element(df_ss, ele))
        kref_ele[ele] = float(np.median(np.asarray(k_list, dtype=float)))
    return kref_ele

def compute_Dk_event(df_ss: pd.DataFrame, elements, kref_ele: dict, eps_stab: float = 1e-12) -> dict:
    """
    Per event:
      - compute ksec for each element
      - compute Dk_ele = 1 - ksec_ele/kref_ele
      - aggregate Dk_event = median(Dk_ele across elements)
    """
    out = {}
    dk_list = []
    for ele in elements:
        ksec = compute_ksec_for_element(df_ss, ele)
        kref = float(kref_ele[ele])
        dk = 1.0 - (ksec / (kref + eps_stab))
        out[f"ksec_ele{ele}"] = float(ksec)
        out[f"Dk_ele{ele}"] = float(dk)
        dk_list.append(dk)
    out["Dk_median"] = float(np.median(np.asarray(dk_list, dtype=float)))
    return out

def compute_fbase_per_mode(case_dir: Path, baseline_ids, freq_file: str) -> np.ndarray:
    """
    Baseline frequency per mode:
      fbase_modei = median over baseline events of f_event_modei
    """
    f_mat = []
    for ev in baseline_ids:
        ev_dir = case_dir / str(ev)
        f_vec = load_freq_vector(ev_dir, freq_file)
        if f_vec.size < 3:
            raise ValueError(f"{ev_dir/freq_file} has <3 modes")
        f_mat.append(f_vec[:3])
    f_mat = np.vstack(f_mat).astype(float)  # (n_base, 3)
    return np.median(f_mat, axis=0)

def compute_Df_event(f_event: np.ndarray, fbase_per_mode: np.ndarray, eps_stab: float = 1e-12) -> dict:
    """
    Per event:
      - Df_modei = 1 - f_event_i / fbase_i
      - Df_median = median(Df_mode1..3)
    """
    if f_event.size < 3:
        raise ValueError(f"Expected 3 modes, got {f_event.size}")
    if fbase_per_mode.size < 3:
        raise ValueError(f"Expected 3 baseline modes, got {fbase_per_mode.size}")

    f_event = f_event[:3].astype(float)
    fbase = fbase_per_mode[:3].astype(float)

    df_modes = 1.0 - (f_event / (fbase + eps_stab))
    return {
        "f1_event": float(f_event[0]),
        "f2_event": float(f_event[1]),
        "f3_event": float(f_event[2]),
        "f1_base": float(fbase[0]),
        "f2_base": float(fbase[1]),
        "f3_base": float(fbase[2]),
        "Df_mode1": float(df_modes[0]),
        "Df_mode2": float(df_modes[1]),
        "Df_mode3": float(df_modes[2]),
        "Df_median": float(np.median(df_modes)),
    }

def compute_Dr_residual_tail(y_true: np.ndarray, y_pred: np.ndarray, dt: float, tail_sec: float) -> dict:
    
    nt = y_true.shape[1]
    k_tail = int(round(tail_sec / dt))
    k_tail = max(1, min(k_tail, nt))

    y_true_tail = y_true[:, -k_tail:]
    y_pred_tail = y_pred[:, -k_tail:]

    
    res = np.mean(np.abs(y_true_tail), axis=1)

    out = {
        "Dr_tail_sec": float(tail_sec),
        "Dr_tail_samples": int(k_tail),
        "Dr_residual_mean": float(np.mean(res)),
        "Dr_residual_max": float(np.max(res)),
    }
    for i, val in enumerate(res, start=1):
        out[f"Dr_residual_ch{i}"] = float(val)
    return out


def apply_load_bridge_multi_support(
    model,
    inputs: np.ndarray,
    dt: float,
    node_channel_map: dict,
    input_channels: list,
    *,
    factor: float = 1.0,
    pattern_tag: int = 20,
    ts_tag_start: int = 2000,
    gm_tag_start: int = 3000,
):
    
    if inputs is None or dt is None:
        raise ValueError("Missing inputs or dt.")
    if inputs.ndim != 2:
        raise ValueError(f"inputs must be 2D (n_channels, nt). Got {inputs.shape}")

    # Build mapping: channel number -> row index in inputs
    ch_to_row = {ch: k for k, ch in enumerate(input_channels)}

    model.pattern("MultipleSupportExcitation", pattern_tag)

    ts_tag = ts_tag_start
    gm_tag = gm_tag_start

    for node, (chx, chy) in node_channel_map.items():
        if chx not in ch_to_row or chy not in ch_to_row:
            raise ValueError(
                f"Node {node} requests channels ({chx},{chy}), "
                f"but input_channels={input_channels}."
            )

        ix = ch_to_row[chx]
        iy = ch_to_row[chy]

        # 1) timeSeries for this support in X and Y
        ts_x = ts_tag; ts_tag += 1
        ts_y = ts_tag; ts_tag += 1
        model.timeSeries("Path", ts_x, values=inputs[ix].tolist(), dt=dt, factor=factor)
        model.timeSeries("Path", ts_y, values=inputs[iy].tolist(), dt=dt, factor=factor)

        # 2) GroundMotion objects
        gm_x = gm_tag; gm_tag += 1
        gm_y = gm_tag; gm_tag += 1
        model.groundMotion(gm_x, "Plain", accel=ts_x)
        model.groundMotion(gm_y, "Plain", accel=ts_y)

        # 3) imposed motions (DOF 1=X, DOF 2=Y)
        model.imposedMotion(node, 1, gm_x)
        model.imposedMotion(node, 2, gm_y)
    return model
