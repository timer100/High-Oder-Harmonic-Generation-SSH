import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from ..models.base import Hamiltonian

def plot_wavefunction(model: Hamiltonian, states, kind: str = "wavefunction", ax=None, show: bool = True, title: str = None):
    """
    Plot the wavefunction or probability density of specified eigenstates.
    
    Args:
        model (Hamiltonian): Instance of the model.
        states (int, list, slice, range): Index or indices of states to plot.
        kind (str): "wavefunction" (default) to plot amplitude (real part),
                    or "density" to plot probability density |psi|^2.
        ax (matplotlib.axes.Axes, optional): Axes to plot on.
        show (bool): Whether to show the plot.
        title (str): Optional title.
    """
    # 1. Diagonalize Static Hamiltonian
    H_sparse = model.build_time_dependent_hamiltonian(0, lambda t: 0)
    H_dense = H_sparse.toarray()
    
    # Full diagonalization
    eigvals, eigvecs = eigh(H_dense)
    
    # 2. Parse 'states' argument
    indices = []
    if isinstance(states, int):
        indices = [states]
    elif isinstance(states, (list, tuple, np.ndarray)):
        indices = states
    elif isinstance(states, (slice, range)):
        start = states.start if states.start is not None else 0
        stop = states.stop if states.stop is not None else model.N
        step = states.step if states.step is not None else 1
        indices = list(range(start, stop, step))
    else:
        raise ValueError("states must be int, list, slice, or range.")
        
    # Validate indices
    indices = [i for i in indices if 0 <= i < model.N]
    
    if not indices:
        print("No valid states selected.")
        return None

    # 3. Plotting
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        
    x = model.positions
    
    for idx, i in enumerate(indices):
        psi = eigvecs[:, i]
        energy = eigvals[i]
        
        if kind == "density":
            y_data = np.abs(psi)**2
            ylabel = r"$|\psi(x)|^2$"
            label = f"Density: State {i} (E={energy:.4f})"
        else:
            y_data = np.real(psi)
            ylabel = r"$\psi(x)$"
            label = f"WF: State {i} (E={energy:.4f})"
            
        ax.plot(x, y_data, label=label, alpha=0.8, linewidth=1.5)
        
    ax.set_xlabel("Position (x)")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Wavefunction ({kind})")
        
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    if show and ax is None:
        plt.show()
        
    return ax
