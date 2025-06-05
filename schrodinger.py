import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Constants
h_bar = 1.0545718e-34  # Planck's constant / (2*pi) in J*s
m = 9.10938356e-31     # Electron mass in kg
L = 1e-9               # Width of the potential well in meters
N = 1000               # Number of grid points
dx = L / N             # Grid spacing
x = np.linspace(0, L, N)

# Potential function (modify as needed)
def potential(x):
    return 0

# Diagonal elements of the Hamiltonian
def diagonal_elements():
    return h_bar**2 / (2 * m * dx**2) + potential(x)

# Off-diagonal elements of the Hamiltonian
def off_diagonal_elements():
    return -h_bar**2 / (2 * m * dx**2)

# Constructing the Hamiltonian matrix
def construct_hamiltonian():
    diagonal = np.diagflat(diagonal_elements())
    off_diagonal = np.diagflat([off_diagonal_elements()] * (N-1), 1) + np.diagflat([off_diagonal_elements()] * (N-1), -1)
    return diagonal + off_diagonal

# Solve the eigenvalue problem
def solve_schrodinger():
    H = construct_hamiltonian()
    energies, wavefunctions = np.linalg.eigh(H)
    return energies, wavefunctions

# Main function to run the solver
def main():
    energies, wavefunctions = solve_schrodinger()
    print("Eigenenergies:")
    for i, energy in enumerate(energies):
        print(f"Energy level {i + 1}: {energy:.2e} J")
    
    # Calculate the probability density function
    probability_density = np.abs(wavefunctions[:, 0])**2

    # Convert colormap values to RGBA
    cmap = cm.magma(probability_density / np.max(probability_density))
    rgba_colors = cmap[:, :3]  # Exclude alpha channel

    # Plot the ground state wavefunction
    plt.figure(figsize=(10, 5))
    for i in range(N):
        plt.plot([x[i], x[i]], [0, wavefunctions[i, 0]], color=rgba_colors[i])

    plt.title("Ground State Wavefunction")
    plt.xlabel("Position (m)")
    plt.ylabel("Wavefunction Amplitude")

    # Plot the probability density function
    plt.figure(figsize=(10, 5))
    for i in range(N):
        plt.plot([x[i], x[i]], [0, probability_density[i]], color=rgba_colors[i])

    plt.title("Probability Density Function")
    plt.xlabel("Position (m)")
    plt.ylabel("Probability Density")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()