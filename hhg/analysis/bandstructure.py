
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from ..models.base import Hamiltonian

def plot_band_structure(model: Hamiltonian, label: str = None, ax=None, show: bool = True):
    """
    Calculate and plot the band structure of a finite model using FFT of eigenvectors.
    
    Args:
        model (Hamiltonian): Instance of SSHModel or ESSHModel.
        label (str): Title/Label for the plot.
        ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, creates new figure.
        show (bool): Whether to show the plot immediately.
        
    Returns:
        ax: The matplotlib axes with the plot.
    """
    # 1. Get Static Hamiltonian
    # Ensure raw format (dense might be needed for full diagonalization)
    # The user used dense `eigh` in example.
    H_sparse = model.build_time_dependent_hamiltonian(0, lambda t: 0)
    H_dense = H_sparse.toarray()
    N = model.N
    
    # 2. Diagonalize
    eigvals, eigvecs = eigh(H_dense)
    
    # 3. FFT to get k-space weight
    # eigvecs shape (N, N) where columns are states
    # fft along axis=0 (site index) to transform spatial dependence to k-dependence
    fft_eigenvectors = np.fft.fft(eigvecs, axis=0)
    fft_magnitude = np.abs(fft_eigenvectors)**2
    ln_fft_magnitude = np.log1p(fft_magnitude)
    
    # 4. Define k-values
    # N points from -pi/2 to pi/2 (as per user code)
    k_values = np.linspace(-np.pi/2, np.pi/2, N)
    
    # 5. Plotting
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
        
    for i in range(N):
        # Scatter calculation for each band (eigenstate i)
        # We plot Energy vs k. But we have N eigenstates.
        # For each eigenstate 'i', it has energy E_i. 
        # It is distributed in k-space according to ln_fft_magnitude[:, i].
        # So we plot (k_values, E_i * ones) colored by magnitude.
        
        # Current method plots N * N points. Can be heavy for large N.
        # User defined s=3.
        ax.scatter(k_values, eigvals[i] * np.ones_like(k_values),
                   c=ln_fft_magnitude[:, i], cmap='gray_r', s=3, marker='.')
        
    if label:
        ax.set_title(label, fontsize=14)
            
    ax.set_xlim(-np.pi/2, np.pi/2)
    ax.set_xlabel(r"$k\ (\mathrm{a.u.})$", fontsize=14)
    ax.set_ylabel("Energy (a.u.)", fontsize=14)
    ax.grid(True, linestyle='--', linewidth=0.5)
        
    if show and ax is None:
        plt.show()
        
    return ax
