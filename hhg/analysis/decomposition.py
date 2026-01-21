import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse import eye
from ..models.base import Hamiltonian

def get_eigenstates(model: Hamiltonian, k: int = None):
    """
    Get sorted eigenstates of the static Hamiltonian.
    
    Args:
        model (Hamiltonian): The model instance.
        k (int): Number of states to compute. Defaults to N_occ + 2.
    
    Returns:
        tuple: (eigvals, eigvecs) sorted by energy.
    """
    if k is None:
        k = (model.N // 2) + 2
        
    H0 = model.build_time_dependent_hamiltonian(0, lambda t: 0)
    eigvals, eigvecs = eigsh(H0, k=k, which='SA')
    
    order = np.argsort(eigvals)
    return eigvals[order], eigvecs[:, order]



from ..solvers import TimeEvolver
from ..analysis import compute_dipole_acceleration

def run_current_decomposition(model, field_func, dt=0.1, ncyc=5, omega=0.0075):
    """
    Run time evolution with Current Decomposition (CD).
    Splits the wavefunction into 'occ' (Full), 'VB' (Valence Band), and 'ES' (Edge States).
    
    Args:
        model: The Hamiltonian model (SSHModel).
        field_func: The electric field function.
        dt (float): Time step.
        ncyc (float): Number of cycles (for time range calculation).
        omega (float): Frequency (for time range calculation).
        
    Returns:
        tuple: (times, dipole_dict)
            times: Array of time points.
            dipole_dict: Dictionary with keys 'occ', 'VB', 'ES' 
                         containing dipole acceleration arrays.
    """
    t_max = 2 * np.pi * ncyc / omega
    
    # 1. Get Initial States
    # Full occupied state (matches 'occ')
    psi_occ = model.get_ground_state()
    
    # Determine split (SSH Logic)
    N_occ = model.N // 2
    psi_dict = {}
    psi_dict['occ'] = psi_occ
    
    # VB: All except last column
    psi_dict['VB'] = psi_occ[:, :-1]
    # ES: Last column only (keep as 2D column vector)
    psi_dict['ES'] = psi_occ[:, -1:]
    labels = ['occ', 'VB', 'ES']

    # 2. Time Evolution (Independent for each component)
    evolver = TimeEvolver(model)
    
    results = {}
    
    times = None
    
    for label in labels:
        psi_init = psi_dict[label]
        
        # Generator
        iterator = evolver.evolve(field_func, t_max, dt, verbose=(label=='occ'), initial_state=psi_init)
        
        X_t = []
        curr_times = []
        
        first = True
        for step, t, psi in iterator:
            if first:
                # Skip t=0 initial state to align with user's original logic (recording loop starts after evolution)
                first = False
                continue
            
            if psi.ndim == 1:
                psi = psi[:, np.newaxis]
                
            x_val = np.einsum('ij,i,ij->', np.conj(psi), model.positions, psi).real
            X_t.append(x_val)
            curr_times.append(t)
            
        if times is None:
            times = np.array(curr_times)
            
        results[label] = np.array(X_t)
        
    # 3. Compute Accelerations
    dipole_dict = {}
    for label in labels:
        dipole_dict[label] = compute_dipole_acceleration(results[label], dt)
        
    return times, dipole_dict

