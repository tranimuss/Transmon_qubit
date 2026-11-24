import numpy as np
from scipy.linalg import eigh
from scipy.integrate import solve_ivp
from scipy.constants import e, hbar, h

# The transmon circuit Hamiltonian

def transmon_hamiltonian(EJ, EC, ng=0, N=20):

    n = np.arange(-N, N+1)

    H = np.diag(4 * EC * (n-ng)**2)

    for i in range(len(n)-1):
        H[i, i+1] = -EJ/2
        H[i+1, i] = -EJ/2

    return H

# Get the parameters for solving the time-dependent SHrodinger equation - energies and the charge operator in the energy basis.

def get_energy_levels(H, n_levels=6):
    # Finding the cutoff value for the charge states
    N = int((len(H)-1)/2)

    # Charge basis
    n = np.arange(-N, N + 1)

    # Charge operator
    n_operator = np.diag(n)

    # Diagonalize the static Hamiltonian
    energies, states = eigh(H)

    # Retain the lowest energy levels
    V = states[:, :n_levels]

    # Transform charge operator into energy basis
    n_energy = V.conj().T @ n_operator @ V

    return energies, n_energy

# A gaussian envelope function, truncated at 5 sigma for optimization

def gaussian_envelope(t, t0, sigma):

    if abs(t-t0)<5*sigma:
        return np.exp( -(t-t0)**2 / (2*sigma**2) )
    else:
        return 0

# The transmon Hamiltonian with an addition of the drive microwave energy with a gaussian pulse envelope

def driven_hamiltonian(t, energies, n_energy,
                       EC, ng_ac, omega_d,
                       t0, sigma):

    H0 = np.diag(energies)

    A = gaussian_envelope(t, t0, sigma)

    H_drive = (
        -8 * EC * ng_ac
        * A * np.cos(omega_d * t)
        * n_energy
    )

    return H0 + H_drive

# The schrodinger equation for the driven transmon hamiltonian

def schrodinger_equation(t, psi, energies,
                        n_energy, EC, ng_ac, omega_d,
                        t0, sigma):

    H = driven_hamiltonian(
        t, energies, n_energy,
        EC, ng_ac, omega_d,
        t0, sigma
    )

    return -1j / hbar * H @ psi