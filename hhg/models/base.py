from abc import ABC, abstractmethod
import numpy as np
from scipy.sparse import csr_matrix

class Hamiltonian(ABC):
    """
    Abstract base class for tight-binding Hamiltonians.
    """

    def __init__(self, N: int):
        """
        Initialize the Hamiltonian model.

        Args:
            N (int): Number of sites (must be even).
        """
        if N % 2 != 0:
            raise ValueError("Number of sites N must be even.")
        self.N = N
        self.static_H = None

    @abstractmethod
    def build_static_hamiltonian(self) -> csr_matrix:
        """
        Construct the static part of the Hamiltonian.
        
        Returns:
            scipy.sparse.csr_matrix: The static Hamiltonian matrix.
        """
        pass

    @abstractmethod
    def build_time_dependent_hamiltonian(self, t: float, electric_field_func) -> csr_matrix:
        """
        Construct the full Hamiltonian at time t.

        Args:
            t (float): Time.
            electric_field_func (callable): Function E(t) that returns the electric field amplitude.

        Returns:
            scipy.sparse.csr_matrix: The Hamiltonian H(t).
        """
        pass

    @abstractmethod
    def get_ground_state(self) -> np.ndarray:
        """
        Calculate the ground state of the system, applying any necessary topological state logic.
        
        Returns:
            np.ndarray: The ground state wavefunction (or occupied states).
        """
        pass
