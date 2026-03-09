import numpy as np
import tqdm

class ModalAnalysis:
    def __init__(self, model, n):
        self.model = model
        self.n = n
    
        self.lambdas = model.eigen(n)  
        print(f"[Debug] eigen(n={n}) returned {len(lambdas)} values: {lambdas}")
        lambdas = np.asarray(lambdas, dtype=float)
        omega = np.sqrt(np.abs(lambdas))                    # rad/s
        self.frequencies = omega / (2*np.pi)

    
def apply_damping(model, zeta, verbose=False):
    """
    Apply mass and stiffness proportional Rayleigh damping coefficients
    """
    lambdas = model.eigen(2, "fullGenLapack")  
    omegas = np.sqrt(np.abs(lambdas))
    A = np.array([[1/(2*omegas[0]), omegas[0]/2], [1/(2*omegas[1]), omegas[1]/2]])
    b = np.array(zeta)
    alpha, beta = np.linalg.solve(A, b)
    if verbose >= 2:
        print(f"Rayleigh damping coefficients: alpha={alpha:.4e}, beta={beta:.4e}")
    # rayleigh(alphaM, betaK, betaKinit, betaKcomm)
    model.rayleigh(alpha, 0.0, beta, 0.0)

def get_material_response(model, element, sec_tag, y, z):
    try:
        strain =  model.eleResponse(element, "section", sec_tag, "fiber", y, z, "strain")
        stress =  model.eleResponse(element, "section", sec_tag, "fiber", y, z, "stress")
        return strain, stress
    except Exception as e:
        print(e)
        return None

def analyze(model, nt, dt, 
            output_nodes=[5,10,15],
            output_elements=[1,5,9],
            n_modes=3,
            yFiber=9.0,
            zFiber=0.0,
            verbose=False,
            ):

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
    #model.test("EnergyIncr", 1.0e-14, 40)
    model.test("NormDispIncr", 1.0e-6, 40)

    # Perform iterations with the Newton-Raphson algorithm
    #model.algorithm("Newton")
    model.algorithm("NewtonLineSearch")

    # define the integration scheme, the Newmark with gamma=0.5 and beta=0.25
    model.integrator("Newmark", 0.5, 0.25)

    # Define the analysis
    model.analysis("Transient")

    # -----------------------
    # 3. Perform the analysis
    # -----------------------

    # record once at time 0
    record_nodes = set(output_nodes)

    displacements = {
        node: [model.nodeDisp(node)] for node in record_nodes
    }
    accelerations = {
        node: [model.nodeAccel(node)] for node in record_nodes
    }
    strains = {
        element: [get_material_response(model, element, 1, yFiber, zFiber)[0]] for element in output_elements
    }
    stresses = {
        element: [get_material_response(model, element, 1, yFiber, zFiber)[1]] for element in output_elements
    }

    # get modes
    lambdas = model.eigen(n_modes, "fullGenLapack")  
    omegas = np.sqrt(np.abs(lambdas))
    freqs_before = omegas/(2*np.pi) 

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
        for node in record_nodes:
            displacements[node].append(model.nodeDisp(node))
            accelerations[node].append(model.nodeAccel(node))
        
        for element in output_elements:
            strains[element].append(get_material_response(model, element, 1, yFiber, zFiber)[0])
            stresses[element].append(get_material_response(model, element, 1, yFiber, zFiber)[1])

    lambdas_after = model.eigen(n_modes, "fullGenLapack") 
    omega_after = np.sqrt(np.abs(lambdas_after))   
    freqs_after = omega_after/(2*np.pi)
           
    return displacements, accelerations, stresses, strains, freqs_before, freqs_after