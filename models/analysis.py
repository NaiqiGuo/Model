import numpy as np
import tqdm

class ModalAnalysis:
    def __init__(self, model, n):
        self.model = model
        self.n = n
    
        self.lambdas = model.eigen(n)  
        lambdas = np.asarray(lambdas, dtype=float)
        omega = np.sqrt(np.abs(lambdas))                    # rad/s
        self.frequencies = omega / (2*np.pi)

    
def apply_damping(model, zeta, verbose=False):
    """
    Apply mass and stiffness proportional Rayleigh damping coefficients
    """
    lambdas = model.eigen(2, "fullGenLapack")  
    if verbose >= 2:
        model.modalProperties(print=True)
    omegas = np.sqrt(np.abs(lambdas))
    A = np.array([[1/(2*omegas[0]), omegas[0]/2], [1/(2*omegas[1]), omegas[1]/2]])
    b = np.array(zeta)
    alpha, beta = np.linalg.solve(A, b)
    if verbose >= 2:
        print(f"Rayleigh damping coefficients: alpha={alpha:.4e}, beta={beta:.4e}")
    # rayleigh(alphaM, betaK, betaKinit, betaKcomm)
    model.rayleigh(alpha, 0.0, beta, 0.0)

def get_fiber_response(model, element, sec_tag, y, z):
    try:
        # Original fiber-response query kept here for section-based elements.
        strain = model.eleResponse(element, "section", sec_tag, "fiber", y, z, "strain") # 1DOF
        stress = model.eleResponse(element, "section", sec_tag, "fiber", y, z, "stress") # 1DOF
        return strain, stress
    except Exception as e:
        print(e)
        return None
    
def get_material_response(model, element):
    try:
        deformation = model.eleResponse(element, "deformation") #
        force = model.eleResponse(element, "force")
   
        return deformation, force
    except Exception as e:
        print(e)
        return None


def select_fiber_response_component(strain, stress, fiber_response_dof=None):
    strain = np.atleast_1d(np.asarray(strain, dtype=float)).reshape(-1)
    stress = np.atleast_1d(np.asarray(stress, dtype=float)).reshape(-1)

    if fiber_response_dof is None:
        idx = int(np.argmax(np.abs(stress)))
    else:
        idx = fiber_response_dof - 1
    if idx < 0 or idx >= len(strain) or idx >= len(stress):
        raise IndexError(
            f"Requested fiber_response_dof={fiber_response_dof}, "
            f"but available response size is strain={len(strain)}, stress={len(stress)}"
        )

    return float(strain[idx]), float(stress[idx])


def select_material_response_components(
    deformation,
    force,
    deformation_dof,
    force_dof,
):
    deformation = np.atleast_1d(np.asarray(deformation, dtype=float)).reshape(-1)
    force = np.atleast_1d(np.asarray(force, dtype=float)).reshape(-1)

    deformation_idx = deformation_dof - 1
    force_idx = force_dof - 1

    if deformation_idx < 0 or deformation_idx >= len(deformation):
        raise IndexError(
            f"Requested deformation_dof={deformation_dof}, "
            f"but available deformation response size is {len(deformation)}"
        )
    if force_idx < 0 or force_idx >= len(force):
        raise IndexError(
            f"Requested force_dof={force_dof}, "
            f"but available force response size is {len(force)}"
        )

    return float(deformation[deformation_idx]), float(force[force_idx])


def get_element_response(
    model,
    element,
    sec_tag,
    y,
    z,
    response_mode="fiber",
    fiber_response_dof=None,
    material_deformation_dof=None,
    material_force_dof=None,
):
    if response_mode == "fiber":
        fiber_response = get_fiber_response(model, element, sec_tag, y, z)
        if fiber_response is not None:
            return select_fiber_response_component(
                fiber_response[0],
                fiber_response[1],
                fiber_response_dof,
            )
        return None

    if response_mode != "material":
        raise ValueError(
            f"Unsupported response_mode={response_mode!r}. "
            "Expected 'fiber' or 'material'."
        )

    material_response = get_material_response(model, element)
    if material_response is None:
        return None

    if material_deformation_dof is None or material_force_dof is None:
        raise ValueError(
            "Material response requires both material_deformation_dof "
            "and material_force_dof to be specified."
        )

    return select_material_response_components(
        material_response[0],
        material_response[1],
        deformation_dof=material_deformation_dof,
        force_dof=material_force_dof,
    )

def analyze(model, nt, dt, 
            output_nodes=[5,10,15],
            output_elements=[1,5,9],
            n_modes=3,
            yFiber=9.0,
            zFiber=0.0,
            response_mode="fiber",
            fiber_response_dof=None,
            material_deformation_dof=None,
            material_force_dof=None,
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
    response_x = {
        element: [get_element_response(
            model,
            element,
            1,
            yFiber,
            zFiber,
            response_mode,
            fiber_response_dof,
            material_deformation_dof,
            material_force_dof,
        )[0]] for element in output_elements
    }
    response_y = {
        element: [get_element_response(
            model,
            element,
            1,
            yFiber,
            zFiber,
            response_mode,
            fiber_response_dof,
            material_deformation_dof,
            material_force_dof,
        )[1]] for element in output_elements
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
            response = get_element_response(
                model,
                element,
                1,
                yFiber,
                zFiber,
                response_mode,
                fiber_response_dof,
                material_deformation_dof,
                material_force_dof,
            )
            response_x[element].append(response[0])
            response_y[element].append(response[1])

    lambdas_after = model.eigen(n_modes, "fullGenLapack") 
    omega_after = np.sqrt(np.abs(lambdas_after))   
    freqs_after = omega_after/(2*np.pi)
           
    return displacements, accelerations, response_x, response_y, freqs_before, freqs_after
