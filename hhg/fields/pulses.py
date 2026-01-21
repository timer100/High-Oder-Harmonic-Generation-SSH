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
        
        if t < 0 or t > (2 * np.pi * self.ncyc / self.omega): 
            pass

        arg = omega_t / (2 * self.ncyc)
        envelope = np.sin(arg) ** 2
        
        # Derivative of envelope wrt t
        d_envelope = (self.omega / self.ncyc) * np.sin(arg) * np.cos(arg)
        
        E = -self.A0 * (d_envelope * np.sin(omega_t) + self.omega * envelope * np.cos(omega_t))
        return E