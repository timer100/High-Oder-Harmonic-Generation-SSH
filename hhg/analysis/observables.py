import numpy as np

def compute_position_expectation(psi: np.ndarray, positions: np.ndarray) -> float:
    """
    Compute <X(t)> = sum_occupied <psi_i | x | psi_i>
    
    Args:
        psi (np.ndarray): Wavefunction matrix (N, N_occ).
        positions (np.ndarray): Site positions (N,).
        
    Returns:
        float: Total dipole moment (real part).
    """
    return np.einsum('ji,j,ji->', np.conj(psi), positions, psi).real

def compute_dipole_acceleration(X_t_values: np.ndarray, dt: float) -> np.ndarray:
    """
    Compute dipole acceleration from position expectation values X(t).
    d^2X/dt^2.
    
    Args:
        X_t_values (np.ndarray): Time trace of position expectation.
        dt (float): Time step.
        
    Returns:
        np.ndarray: Acceleration array.
    """
    velocity = np.gradient(X_t_values, dt)
    acceleration = np.gradient(velocity, dt)
    return acceleration
