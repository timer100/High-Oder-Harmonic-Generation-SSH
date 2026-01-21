import numpy as np
from scipy.sparse import eye, csc_matrix
from scipy.sparse.linalg import spsolve
from tqdm import tqdm
from ..models.base import Hamiltonian

class TimeEvolver:
    """
    Handles time evolution of the wavefunction using Crank-Nicolson method.
    """
    
    def __init__(self, model: Hamiltonian):
        self.model = model

    def evolve(self, field_func, t_max: float, dt: float = 0.1, verbose: bool = True, initial_state: np.ndarray = None):
        """
        Run the time evolution.

        Args:
            field_func (callable): Function E(t).
            t_max (float): Total simulation time.
            dt (float): Time step.
            verbose (bool): Show progress bar.
            initial_state (np.ndarray, optional): Custom initial wavefunction. 
                                                  If None, computes ground state at t=0.

        Yields:
            tuple: (step_index, time, current_wavefunction_matrix)
            
        Returns:
            None
        """
        times = np.arange(0, t_max, dt)
        N = self.model.N
        
        psi = initial_state
        
        if psi is None:
            # 1. Initial State (Ground State)
            # Delegate to model to handle any parity/topological state logic
            psi = self.model.get_ground_state()
        
        # 2. Time Propagation
        I = eye(N, format='csr')
        
        # Yield initial state
        yield 0, 0.0, psi
        
        args = (times, ) if verbose else (times, )
        iterator = tqdm(enumerate(times), total=len(times), desc="Evolution") if verbose else enumerate(times)
        
        for i, t in iterator:
            # Crank-Nicolson at t + dt/2
            t_mid = t + dt / 2
            H_mid = self.model.build_time_dependent_hamiltonian(t_mid, field_func)
            
            # (I + i*dt/2 * H) psi(t+dt) = (I - i*dt/2 * H) psi(t)
            # A x = B b
            # A = I + 1j * (dt/2) * H
            # B = I - 1j * (dt/2) * H
            
            factor = 1j * (dt / 2)
            A = (I + factor * H_mid).tocsc()
            B = (I - factor * H_mid).tocsc()
            
            # Solve for next step
            # spsolve can handle multiple RHS (psi has N_occ columns)
            psi = spsolve(A, B @ psi)
            
            yield i + 1, t + dt, psi
