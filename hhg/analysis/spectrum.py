import numpy as np
from scipy.fftpack import fft, fftfreq

def compute_hhg_spectrum(acceleration: np.ndarray, dt: float, omega_drive: float):
    """
    Compute the HHG spectrum (harmonic order vs intensity).
    
    Args:
        acceleration (np.ndarray): Dipole acceleration time signal.
        dt (float): Time step.
        omega_drive (float): Driving frequency (to normalize harmonic order).
        
    Returns:
        tuple: (harmonic_orders, spectrum_intensity)
    """
    N_t = len(acceleration)
    
    # Use Window function to denoise
    window = np.hanning(N_t)
    acc_windowed = acceleration * window
    
    # Compute Power spectrum
    spectrum = np.abs(fft(acc_windowed))**2
    
    # Compute Frequencies
    freqs = fftfreq(N_t, d=dt)
    
    # Convert to harmonic order
    harmonic_orders = (2 * np.pi * freqs) / omega_drive
    
    # Return positive half
    return harmonic_orders[:N_t//2], spectrum[:N_t//2]
