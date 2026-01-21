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
        
    H0 = model.build_time_dependent_hamiltonian(0, lambda t: 0) # Static H basically
    eigvals, eigvecs = eigsh(H0, k=k, which='SA')
    
    order = np.argsort(eigvals)
    return eigvals[order], eigvecs[:, order]



from ..solvers import TimeEvolver
from ..analysis import compute_dipole_acceleration
from tqdm import tqdm

def run_current_decomposition(model, field_func, dt=0.1, ncyc=5, omega=0.0075):
    """
    Run time evolution with Current Decomposition (CD).
    Splits the wavefunction into 'occ' (Full), 'VB' (Valence Band), and 'ES'/'ESs' (Edge States).
    
    Args:
        model: The Hamiltonian model (SSHModel).
        field_func: The electric field function.
        dt (float): Time step.
        ncyc (float): Number of cycles (for time range calculation).
        omega (float): Frequency (for time range calculation).
        
    Returns:
        tuple: (times, dipole_dict)
            times: Array of time points.
            dipole_dict: Dictionary with keys 'occ', 'VB', 'ES' (or 'ESs') 
                         containing dipole acceleration arrays.
    """
    t_max = 2 * np.pi * ncyc / omega
    
    # 1. Get Initial States
    # Full occupied state (matches 'occ')
    psi_occ = model.get_ground_state()
    
    # Determine split based on model type or shape
    # SSH has 1 Edge State (N/2 - 1 is last VB, N/2 is ES) -> actually 0-indexed: 0..N_occ-1.
    # N_occ = N // 2
    # SSH: VB = 0..N_occ-2, ES = N_occ-1
    # ESSH: VB = 0..N_occ-3, ESs = N_occ-2, N_occ-1
    
    N_occ = model.N // 2
    psi_dict = {}
    psi_dict['occ'] = psi_occ
    
    # Standard SSH Decomposition
    # VB: All except last column
    psi_dict['VB'] = psi_occ[:, :-1]
    # ES: Last column only (keep as 2D column vector)
    psi_dict['ES'] = psi_occ[:, -1:]
    labels = ['occ', 'VB', 'ES']

    # 2. Time Evolution
    # We need to run evolution for EACH component independently
    # using the SAME TimeEvolver logic (Crank-Nicolson).
    # Since TimeEvolver yields states step-by-step, we can interleave or run sequentially.
    # Interleaving is better to match the structure of the user's loop and avoid re-calculating H(t) if possible,
    # but our TimeEvolver class encapsulates the loop.
    # So strictly using TimeEvolver would mean 3 separate runs.
    # However, H(t) is efficient to build.
    
    # Actually, we can just instantiate TimeEvolver once and call evolve 3 times.
    evolver = TimeEvolver(model)
    
    # Prepare results containers
    # We need to pre-calculate time array to know length? 
    # TimeEvolver.evolve creates the times array. 
    # Let's just run one first to get times.
    
    results = {}
    
    # We want to enable progress bar only for the main one or show combined?
    # Let's run sequentially.
    
    times = None
    
    for label in labels:
        psi_init = psi_dict[label]
        
        # Generator
        iterator = evolver.evolve(field_func, t_max, dt, verbose=(label=='occ'), initial_state=psi_init)
        
        X_t = []
        
        # Skip t=0? User's original code stored index 0 after t+dt/2 step.
        # Implies they store result of step 1 at index 0?
        # "X_t_values[idx] = ... compute ... " inside loop enumerate(times).
        # Times are 0, dt, 2dt...
        # Loop t: H(t+dt/2).
        # Yields psi(t+dt).
        # User stores psi(t+dt) at index corresp to t.
        # This effectively shifts time by dt.
        # But for 'standardized' TimeEvolver, we return (step, t, psi).
        # Let's collect all X expectations and then compute Acc.
        
        curr_times = []
        
        first = True
        for step, t, psi in iterator:
            if first:
                # TimeEvolver yields t=0 state first.
                # User's loop does NOT calculate X for t=0 initial state?
                # User code:
                #   X_t_values = np.zeros(len(times))
                #   for idx, t in enumerate(times):
                #       evolve...
                #       X_t[idx] = expected value
                # So user records state at t=dt, 2dt... into indices 0, 1... associated with times 0, dt...
                # This is a bit shifted.
                # Let's just collect ALL states including t=0, compute X, and then compute Acc.
                # compute_dipole_acceleration handles gradients properly.
                first = False
                continue
            
            # Position Expectation
            # Note: For multi-column psi, we want Sum of expectations of each column
            # X = sum_n <phi_n | x | phi_n>
            # Einstein sum: 'ij,i,ij->' (conjugate(psi) * pos * psi)
            # i=site, j=state
            
            # psi might be (N,) or (N, M)
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

