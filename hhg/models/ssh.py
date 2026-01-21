import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, diags, eye
from scipy.sparse.linalg import eigsh
from .base import Hamiltonian

class SSHModel(Hamiltonian):
    """
    Su-Schrieffer-Heeger (SSH) Model with Nearest-Neighbor (NN) hopping.
    Based on logic from `HHG_SSH.ipynb`.
    """

    def __init__(self, N: int = 100, a: float = 2.0, delta: float = -0.15, 
                 V_A: float = 0.0):
        """
        Initialize the SSH Model.

        Args:
            N (int): Number of sites.
            a (float): Lattice constant.
            delta (float): Dimerization parameter.
            V_A (float): On-site potential strength (staggered V_A, -V_A).
        """
        super().__init__(N)
        
        self.a = a
        self.delta = delta
        self.V_A = V_A
        self.V_B = -V_A

        # Hopping amplitudes
        self.v = -np.exp(-(a - 2 * delta))
        self.w = -np.exp(-(a + 2 * delta))

        # Site positions
        j_indices = np.arange(1, N + 1)
        self.positions = (j_indices - (N + 1) / 2) * a - ((-1)**j_indices * delta)

        # Build static Hamiltonian once
        self.static_H = self.build_static_hamiltonian()

    def build_static_hamiltonian(self) -> csr_matrix:
        H = lil_matrix((self.N, self.N), dtype=float)

        # Intracell hopping (v)
        for i in range(self.N // 2):
            a_idx, b_idx = 2 * i, 2 * i + 1
            H[a_idx, b_idx] = H[b_idx, a_idx] = self.v

        # Intercell hopping (w)
        for i in range(self.N // 2 - 1):
            b_idx, a_next = 2 * i + 1, 2 * (i + 1)
            H[b_idx, a_next] = H[a_next, b_idx] = self.w
        
        return H.tocsr()

    def build_time_dependent_hamiltonian(self, t: float, electric_field_func) -> csr_matrix:
        """
        Construct H(t) = H_static + V(t).
        V(t) includes the staggered potential and the electric field potential.
        
        Args:
            t (float): Time.
            electric_field_func (callable): Function E(t) or E(x, t).
        """
        # Try calling with (x, t) first if it's spatially dependent, or checking signature?
        # Simpler: Try 't' only. If fails or we know it's spatial...
        # Actually SpatiallyDependentField.__call__ requires (x, t).
        # Standard Pulse.__call__ requires (t).
        
        # We can try calling with t. If TypeError, try with (self.positions, t).
        # Or check if it accepts 2 args.
        
        try:
            val = electric_field_func(t)
        except TypeError:
            val = electric_field_func(self.positions, t)
            
        diag = np.array([self.V_A if i % 2 == 0 else self.V_B for i in range(self.N)], dtype=np.float64)
        
        if np.isscalar(val) or val.shape == ():
            # Homogeneous field E(t) -> Potential E(t)*x
            diag += val * self.positions
        else:
            # Spatially dependent field E(x, t) -> Potential integral E(x) dx
            # Logic from user: 
            # dx = np.diff(x)
            # dx = np.append(dx, self.a + 2*self.delta) 
            # return np.cumsum(E*dx)
            
            x = self.positions
            dx = np.diff(x)
            dx = np.append(dx, self.a + 2*self.delta)
            
            # val is E(x, t)
            potential = np.cumsum(val * dx)
            diag += potential
            
        return self.static_H + diags(diag, 0, format='csr')

    def _get_odd_parity_edge_state(self):
        H0 = self.build_time_dependent_hamiltonian(0, lambda t: 0.0)
        
        N_occ = self.N // 2
        # Need k=N_occ+1 to get the edge states
        eigvals, eigvecs = eigsh(H0, k=N_occ + 1, which='SA')
        idx = np.argsort(eigvals)
        eigvecs = eigvecs[:, idx]

        psi1 = eigvecs[:, N_occ - 1]
        psi2 = eigvecs[:, N_occ]

        P = np.eye(self.N)[::-1]
        subspace = np.stack([psi1, psi2], axis=1)
        P_sub = subspace.T.conj() @ (P @ subspace)
        _, Vp = np.linalg.eigh(P_sub)
        
        # Select the odd parity state
        psi_odd = subspace @ Vp[:, 0]
        psi_odd /= np.linalg.norm(psi_odd)

        return psi_odd

    def get_ground_state(self) -> np.ndarray:
        # 1. Build H0
        H0 = self.build_time_dependent_hamiltonian(0, lambda t: 0.0)
        N_occ = self.N // 2
        
        vals, vecs = eigsh(H0, k=N_occ, which='SA')
        order = np.argsort(vals)
        psi = vecs[:, order]
        
        # 2. Check Topological Condition
        # If V_A == 0 and delta < 0, we are in Topological Phase with Inversion Symmetry
        if abs(self.V_A) < 1e-9 and self.delta < 0:
            # Enforce Parity on the last occupied state (Edge State)
            psi[:, -1] = self._get_odd_parity_edge_state()
            
        return psi
