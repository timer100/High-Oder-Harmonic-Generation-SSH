from abc import ABC, abstractmethod
import numpy as np

class Field(ABC):
    """Abstract base class for time-dependent fields."""
    
    @abstractmethod
    def __call__(self, t: float):
        """Return the field value at time t."""
        pass

class SinSquaredPulse(Field):
    """
    Electric field pulse with a sin^2 envelope.
    E(t) = -A0 * [ d(env)/dt * sin(wt) + w * env * cos(wt) ]
    Matches `electric_field` logic in HHG notebooks.
    """
    
    def __init__(self, A0: float = 0.2, omega: float = 0.0075, ncyc: float = 5):
        """
        Args:
            A0 (float): Amplitude.
            omega (float): Frequency.
            ncyc (float): Number of optical cycles.
        """
        self.A0 = A0
        self.omega = omega
        self.ncyc = ncyc

    def __call__(self, t: float) -> float:
        """
        Calculate electric field at time t.
        """
        omega_t = self.omega * t
        
        # Envelope: sin^2( wt / 2N )
        # Note: The notebook uses `omega_t / (2 * self.ncyc)` argument for envelope
        
        if t < 0 or t > (2 * np.pi * self.ncyc / self.omega): 
            pass

        arg = omega_t / (2 * self.ncyc)
        envelope = np.sin(arg) ** 2
        
        # Derivative of envelope wrt t
        d_envelope = (self.omega / self.ncyc) * np.sin(arg) * np.cos(arg)
        
        # E(t) formula from notebook:
        E = -self.A0 * (d_envelope * np.sin(omega_t) + self.omega * envelope * np.cos(omega_t))
        return E

class SpatiallyDependentField:
    """
    Electric field with spatial dependence (Local Illumination).
    E(x, t) = E(t) * mask(x)
    """
    
    def __init__(self, field_t, x0: float, x_l: float):
        """
        Args:
            field_t (callable): Time-dependent field function E(t).
            x0 (float): Center of illumination.
            x_l (float): Width parameter of illumination.
        """
        self.field_t = field_t
        self.x0 = x0
        self.x_l = x_l

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        """
        Calculate electric field at positions x and time t.
        
        Args:
            x (np.ndarray): Array of site positions.
            t (float): Time.
            
        Returns:
            np.ndarray: Field array E(x, t).
        """
        E_t = self.field_t(t)
        
        # Matches HHG_ESSH_LL.ipynb logic:
        # arg = (x - self.x0) * (np.pi / (2*self.x_l))
        # mask = np.abs(x - self.x0) < self.x_l
        # field[mask] = E_t * np.cos(arg[mask])**2
        
        arg = (x - self.x0) * (np.pi / (2 * self.x_l))
        mask = np.abs(x - self.x0) < self.x_l
        
        field_val = np.zeros_like(x, dtype=float)
        field_val[mask] = E_t * np.cos(arg[mask])**2
        
        return field_val
