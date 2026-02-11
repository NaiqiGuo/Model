
import math
import xara
import xara.units.iks as units
import numpy as np
import tqdm

from xsection import CompositeSection
from xsection.library import Circle, Equigon
from xsection.library._girder import GirderSection


from mdof.utilities.config import extract_channels

# Verbosity
# False means print nothing;
# True or 1 means print progress messages only;
# 2 means print progress and validation messages
VERBOSE = 2



class Painter:
    # Labeled channel numbers from quakeio object
    input_channels = [1,3,15,17,18,20] # ordered arbitrarily
    # Nodes for excitation, order corresponds to input_channels
    input_nodes = [] # TODO NG: fill in input nodes.
    # DOFs for excitation, order corresponds to input_channels
    input_dofs = []  # TODO NG: fill in input dofs

    output_nodes = [3,5]
    output_elements = [3]



    # Concrete modulus (ksi)
    def __init__(self, units):
        self.units = units

        self.fy = 60.0*units.ksi   # ksi
        self.Es = 30e3*units.ksi

        self.fc_unconf = 4.0*units.ksi   # unconfined concrete
        self.fc_conf   = 5.0*units.ksi  # confined concrete

        self.poisson = 0.24

        self.Ec = 57.0 * math.sqrt(self.fc_unconf/units.psi)*units.ksi
        self.Gc = self.Ec / (2*(1+self.poisson))

        if VERBOSE >= 2:
            print(f"Ec: {self.Ec/units.ksi:.2f} ksi, Gc: {self.Gc/units.ksi:.2f} ksi")


    
    def add_section(self, model, tag, shape, elastic=True, fiber=False):
        fc_unconf = self.fc_unconf   # unconfined concrete
        fc_conf   = self.fc_conf  # confined concrete
        fy = self.fy   # ksi
        Es = self.Es
        Ec = self.Ec
        Gc = self.Gc

        if fiber or not elastic:
            raise NotImplementedError("Fiber and nonlinear sections not implemented yet.")


        if not fiber: #elastic:
            e = shape.elastic
            model.section("Elastic", tag, E=Ec, A=e.A, Iy=e.Iy, Iz=e.Iz, G=Gc, J=e.J)
            return

        core = 1 
        cover = 2
        steel = 3

        if not elastic:
            # Nonlinear concrete (core and cover) using Concrete01
            #                                   tag  f'c       epsc0         f'cu   epscu
            model.uniaxialMaterial("Concrete01", core, -fc_conf,   -2*fc_conf/Ec,  -3.5,  -0.02)
            model.uniaxialMaterial("Concrete01", cover, -fc_unconf, -2*fc_unconf/Ec, 0.0,  -0.006)

            # Nonlinear reinforcing steel
            #                                tag fy  E0   b
            model.uniaxialMaterial("Steel01", steel, fy, Es, 0.02)
        else:
            # Elastic concrete for both core and cover
            model.uniaxialMaterial("Elastic", core, Ec)
            model.uniaxialMaterial("Elastic", cover, Ec)
            # Elastic steel
            model.uniaxialMaterial("Elastic", steel, Es)
        
        # mesh subdivisions for nonlinear fiber section
        numSubdivCirc = 32
        numSubdivRad  = 5

        # Geometry of circular section
        D_total = 5.0*units.ft  # total diameter in inches (5 ft)
        cover   = 2.0           # concrete cover in inches
        R_ext   = D_total / 2.0
        R_core  = R_ext - cover  # approximate core radius

        GJ   = Gc * shape.elastic.J
        model.section("Fiber", tag, GJ=GJ)
        # core concrete
        model.patch("circ",
                    1,                      # matTag = 1
                    numSubdivCirc, numSubdivRad,
                    0.0, 0.0,               # yCenter, zCenter
                    0.0, R_core,            # intRad, extRad
                    0.0, 2 * math.pi, section=tag)

        # cover concrete
        model.patch("circ",
                    2,                      # matTag = 2
                    numSubdivCirc, numSubdivRad,
                    0.0, 0.0,
                    R_core, R_ext,
                    0.0, 2 * math.pi, section=tag)

        # longitudinal steel
        numBars = 36 # 36
        barArea = 1.56 #1.56              # #11 bar area
        model.layer("circ",
                    3,                    # steel matTag
                    numBars, barArea,
                    0.0, 0.0,             # yCenter, zCenter
                    R_core,               # radius
                    0.0, 2 * math.pi, section=tag)


    def create_column(self):
        units = self.units
        D_total = 5.0*units.ft  # total diameter in inches (5 ft)
        return Circle(radius=D_total/2, mesh_scale=1/8, divisions=24)

        d =  (11/8)*units.inch # diameter of longitudinal rebar
        ds = ( 4/8)*units.inch # diameter of the shear spiral
        cover = (3 + 1/8)*units.inch
        core_radius = (5/2)*units.foot - cover - ds - d/2
        nr = 36

        octagon  = Equigon(5.0*units.foot/2, z=0,
                            name="cover", divisions=8, mesh_type="T3")

        interior = Equigon(core_radius, z=1,
                            name="core", divisions=nr, mesh_type="T3")

        bar = Circle(d/2, z=2, mesh_scale=1/2, divisions=4, name="rebar", mesh_type="T3")

        xr = ((5*units.foot/2) - cover - ds - d/2, 0)


        return CompositeSection([
                    octagon,
                    interior,
                    *bar.linspace(xr, xr, nr, endpoint=False, center=(0,0))
                ])


    def create_girder(self):
        u = self.units

        return GirderSection(
            web_slope      = 0.5,
            thickness_top  = (7 + 1/2)  * u.inch,
            thickness_bot  = (5 + 1/2)  * u.inch,
            height         = 5*u.ft + 8*u.inch,
            width_top      = 2*26 * u.ft,
            width_webs     = [12*u.inch]*7,
            web_spacing    = 7*u.ft + 9*u.inch,
            overhang       = 2*u.ft + 6*u.inch,
            mesh_scale     = 1,
            poisson=self.poisson,
        )


        
    def create_model(self, 
                    elastic:bool,
                    multisupport:bool=False,
                    separate_deck_ends:bool = True,
                    verbose = False):

        units = self.units

        model = xara.Model(ndm=3, ndf=6)

        # Geometry
        deck_height = 24*units.ft + (5*units.ft + 8*units.inch)/2

        skew_angle = np.deg2rad(38.9)
        deck_width = 52*units.ft
        skew_x = deck_width*np.tan(skew_angle)/2

        span1_length = 146*units.ft
        span2_length = 119*units.ft

        deck_length = 2*skew_x + span1_length + span2_length

        density = 150.0 * (units.lbf/units.ft**3)/units.gravity  # mass density of concrete in lb/in^3

        # Nodes: (tag, (x, y, z))
        model.node(0, (0.0,                                   0.0,    deck_height)) # abutment 1 (west)
        model.node(1, (deck_length,                           0.0,    deck_height)) # abutment 2 (east)
        model.node(2, (skew_x+span1_length,                   0.0,    deck_height)) # mid-bent
        model.node(3, (span1_length,                 deck_width/3,    deck_height)) # top of column 1 (north)
        model.node(4, (span1_length,                 deck_width/3,            0.0)) # bottom of column 1 (north)
        model.node(5, (2*skew_x+span1_length,       -deck_width/3,    deck_height)) # top of column 2 (south)
        model.node(6, (2*skew_x+span1_length,       -deck_width/3,            0.0)) # bottom of column 2 (south)
        if separate_deck_ends:
            model.node(9,  (skew_x,                           0.0,    deck_height)) # deck-abut interface (west)
            model.node(10, (deck_length-skew_x,               0.0,    deck_height)) # deck-abut interface (east) 



        # Boundary conditions, fully fixed at 0, 1, 4, 6
        model.fix(0, (1, 1, 1, 1, 1, 1))
        model.fix(1, (1, 1, 1, 1, 1, 1))
        model.fix(4, (1, 1, 1, 1, 1, 1))
        model.fix(6, (1, 1, 1, 1, 1, 1))

        #
        # Sections
        #

        # tags
        column_tag = 1
        girder_tag = 2

        # Create section *shape* objects
        column = self.create_column()
        girder = self.create_girder()
        # if verbose:
        #     print(girder.summary())
        self.add_section(model, column_tag, column, elastic=elastic)
        self.add_section(model, girder_tag, girder, elastic=True, fiber=False)  # Girders always elastic



        # Transformations and elements
        colTransf  = 1
        beamTransf = 2
        model.geomTransf("Linear", colTransf,  (1.0, 0.0, 0.0))
        model.geomTransf("Linear", beamTransf, (0.0, 0.0, 1.0))


        # columns as elasticBeamColumn with elastic section
        col_type = "forceBeamColumn"
        column_element = {
            "section": column_tag,
            "transform": colTransf,
            "mass": column.area * density,
            "shear": 0
        }

        model.element(col_type, 2, (4, 3), **column_element)
        model.element(col_type, 3, (6, 5), **column_element)
    

        # beams always elastic
        beam_type = "PrismFrame"
        girder_element = {
            "mass": girder.area * density,
            "section": girder_tag,
            "transform": beamTransf,
            "shear": 0
        }

        # Girders
        model.element(beam_type, 101, ( 0,  9), **girder_element)
        
        model.element(beam_type, 102, ( 9, 2),  **girder_element)

        model.element(beam_type, 103, ( 2, 10), **girder_element)
        model.element(beam_type, 104, (10,  1), **girder_element)

        # Bent
        model.element(beam_type, 105, ( 3,  2), **girder_element)
        model.element(beam_type, 106, ( 5,  2), **girder_element)


        return model

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
        # P_col_cap = fc_conf * A_col   #  kips

        # # 10% 
        # P_grav_total = 0.05 * P_col_cap         # kips
        # P_per_col    = P_grav_total / 2.0      # kips

        #
        g = units.gravity   # in/s^2
        m_per_node = P_per_col / g             # kip / (in/s^2) 

        # Plain + Constant
        model.pattern("Plain", 1, "Constant")
        # for nd in [2, 3, 5]:
        #     model.load(nd, (0.0, 0.0, -P_per_col/2.0, 0.0, 0.0, 0.0), pattern=1)

        # for nd in [2, 9, 10, 3, 5]:
        #     model.load(nd, (0,0,-P_grav_total,0,0,0), pattern=1)


        # rayleigh(alphaM, betaK, betaKinit, betaKcomm)
        model.rayleigh(0.0319, 0.0, 0.0125, 0.0)
        return model


    def load_event(self, event, scale=1):
        inputs, dt = extract_channels(event, self.input_channels)
        inputs = scale*inputs
        return inputs, dt


    def analyze(self, model, event, 
                scale=1,
                output_nodes=[5,10,15],
                n_modes=3,
                verbose=False,
                ):

        inputs, dt = self.load_event(event, scale=scale)
        nin,nt = inputs.shape

        model = apply_load_bridge(model,
                                inputx=inputs[0],
                                inputy=inputs[1],
                                dt=dt
                                )

        # ----------------------------
        # 1. Configure the analysis
        # ----------------------------

        # create the system of equation
        model.system("BandGen")

        # create the DOF numberer
        model.numberer("RCM")

        # create the constraint handler
        model.constraints("Transformation")

        # Configure the analysis such that iterations are performed until either:
        # 1. the energy increment is less than 1.0e-14 (success)
        # 2. the number of iterations surpasses 20 (failure)
        model.test("EnergyIncr", 1.0e-16, 20)

        # Perform iterations with the Newton-Raphson algorithm
        model.algorithm("Newton")

        # define the integration scheme, the Newmark with gamma=0.5 and beta=0.25
        model.integrator("Newmark", 0.5, 0.25)

        # Define the analysis
        model.analysis("Transient")

        # -----------------------
        # 3. Perform the analysis
        # -----------------------

        # record once at time 0
        displacements = {
            node: [model.nodeDisp(node)] for node in output_nodes
        }
        strains = {
            # element: [get_material_response(model, element, 1, yFiber, zFiber)[0]] for element in output_elements
        }
        stresses = {
            # element: [get_material_response(model, element, 1, yFiber, zFiber)[1]] for element in output_elements
        }

        # get modes
        lambdas = model.eigen(n_modes, "fullGenLapack")  
        omega = np.sqrt(np.abs(lambdas))
        freqs_before = omega/(2*np.pi) 

        # Perform nt analysis steps with a time step of dt
        if verbose:
            print(f"Analysis Progress ({nt} timesteps)")
            timesteps = tqdm.tqdm(range(nt))
        else:
            timesteps = range(nt)

        for i in timesteps:
            status = model.analyze(1, dt) 
            if status != 0:
                raise RuntimeError(f"analysis failed at time {model.getTime()}")
            
            # Save displacements at the current time
            for node in output_nodes:
                displacements[node].append(model.nodeDisp(node))

            
        return displacements, stresses, strains, freqs_before


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


def create_bridge(elastic=True, 
                  multisupport=False,
                  separate_deck_ends=True, 
                  verbose=False):
    assert multisupport == False
    painter = Painter(units)
    model = painter.create_model(
        elastic=elastic, 
        separate_deck_ends=separate_deck_ends, 
        verbose=verbose)
    return model





if __name__ == "__main__":
    import sys
    import veux
    import quakeio
    import matplotlib.pyplot as plt

    painter = Painter(units)

    model = painter.create_model(elastic=True, separate_deck_ends=True, verbose=True)

    artist = veux.create_artist(model, vertical=3)
    artist.draw_axes(extrude=True)
    artist.draw_outlines()
    ev = 1
    model.eigen(3)
    model.modalProperties(print=True)
    artist.draw_outlines(state=lambda n: model.nodeEigenvector(n, ev), scale=100)
    artist.draw_nodes(state=lambda n: model.nodeEigenvector(n, ev), scale=100)
    veux.serve(artist)

    event = quakeio.read(sys.argv[1])


    disp, stresses, strains, freqs_before = painter.analyze(model, event,
                                                    output_nodes=painter.output_nodes,
                                                    scale=units.cmps2,
                                                    verbose=VERBOSE
                                                )
