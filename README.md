# HHG Project

This package provides tools for simulating High Harmonic Generation (HHG) in 1D topological insulators, specifically focusing on the Su-Schrieffer-Heeger (SSH) model.

Key features include:
- **Standard Time Evolution**: Length gauge simulation of electron dynamics.
- **Current Decomposition (CD)**: Analysis of contributions from Bulk vs. Edge states.
- **Static Analysis**: Tools for plotting Band Structures and Wavefunctions.


## 1. Installation

```bash
git clone https://github.com/timer100/High-Oder-Harmonic-Generation-SSH.git
cd High-Oder-Harmonic-Generation-SSH
pip install -r requirements.txt
```

## 2. Setup

Ensure the `hhg` package is in your Python path.
```python
import sys
import os
sys.path.append(os.getcwd()) 

import numpy as np
import matplotlib.pyplot as plt
from hhg.models import SSHModel
from hhg.fields import SinSquaredPulse
from hhg.solvers import TimeEvolver
from hhg.analysis import compute_dipole_acceleration, compute_hhg_spectrum
from hhg.analysis.decomposition import run_current_decomposition
from hhg.analysis import plot_band_structure, plot_wavefunction
```

## 3. Models

### Su-Schrieffer-Heeger (SSH) Model
Load the SSH model from the `hhg.models` module.

```python
# Topological Phase (delta < 0)
ssh_topo = SSHModel(N=100, delta=-0.15, V_A=0.0)

# Trivial Phase (delta > 0)
ssh_trivial = SSHModel(N=100, delta=0.15, V_A=0.0)
```

## 4. Time Evolution (HHG)

To simulate HHG, define an electric field pulse and evolve the system.

```python
# 1. Define Pulse
pulse = SinSquaredPulse(A0=0.2, omega=0.0075, ncyc=5)

# 2. Define Time Evolver
evolver = TimeEvolver(ssh_topo)

# 3. Evolve
t_max = 2 * np.pi * pulse.ncyc / pulse.omega
dt = 0.1

# 4. Get initial state
psi0 = ssh_topo.get_ground_state()
iterator = evolver.evolve(pulse, t_max, dt, initial_state=psi0)

# 5. Collect Data (Dipole Acceleration)
X_t = []

for step, t, psi in iterator:
    x_val = np.einsum('ij,i,ij->', np.conj(psi), ssh_topo.positions, psi).real
    X_t.append(x_val)

acc = compute_dipole_acceleration(np.array(X_t), dt)
harmonics, spectrum = compute_hhg_spectrum(acc, dt, pulse.omega)
```

## 5. Current Decomposition (CD)

Separates the HHG spectrum into contributions from the **Valence Band (VB)** and **Edge States (ES)**.

```python
from hhg.analysis.decomposition import run_current_decomposition

# Run decomposition 
times, dipoles = run_current_decomposition(
    ssh_topo, 
    pulse, 
    dt=0.1, 
    ncyc=5, 
    omega=0.0075
)

# Access components
acc_total = dipoles['occ']
acc_bulk  = dipoles['VB']
acc_edge  = dipoles['ES']

# Compute Spectrum
w, I_total = compute_hhg_spectrum(acc_total, dt, 0.0075)
w, I_edge  = compute_hhg_spectrum(acc_edge, dt, 0.0075)
```

## 6. Static Analysis

Tools for analyzing the system's static properties: Band Structure and Wavefunctions.

```python
from hhg.analysis import plot_band_structure, plot_wavefunction

# Take N=100 for example (100 states)
# 1. Band Structure
plot_band_structure(ssh_topo, label="SSH Topological")

# 2. Plotting Wavefunction 
plot_wavefunction(ssh_topo, states=50, title="State 50")

# 3. Plotting Probability Density |psi|^2
plot_wavefunction(ssh_topo, states=50, kind="density", title="State 50 Density")

# 4. Plotting range of states
plot_wavefunction(ssh_topo, states=range(48, 52), title="Edge States")
```

## 7. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.