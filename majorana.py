import numpy as np
import matplotlib.pyplot as plt
import argparse

def build_hamiltonian(N, t, mu, delta):
    """
    Build the 2N×2N BdG Hamiltonian for a 1D Kitaev chain.
    """
    H = np.zeros((2*N, 2*N), dtype=complex)

    # on‐site (chemical potential) terms
    for i in range(N):
        H[2*i,   2*i  ] = -mu
        H[2*i+1, 2*i+1] =  mu

    # hopping + pairing terms
    for i in range(N-1):
        H[2*i,   2*i+1] = -t + delta
        H[2*i+1, 2*i  ] = -t - delta
        H[2*i+1, 2*i+2] = -t - delta
        H[2*i+2, 2*i+1] = -t + delta

    return H

def simulate_majorana(N, t, mu, delta):
    """
    Compute eigenvalues/vectors, extract the zero‐mode probability.
    Returns:
      energies: array of BdG energies
      prob_particles: |ψ|^2 on particle sublattice (every other component)
    """
    H = build_hamiltonian(N, t, mu, delta)
    energies, modes = np.linalg.eigh(H)

    # pick the eigenvector whose energy is closest to zero
    idx_zero = np.argmin(np.abs(energies))
    zero_mode = modes[:, idx_zero]

    # probability density
    prob = np.abs(zero_mode)**2
    # take only the particle part (even indices)
    prob_particles = prob[::2]

    return energies, prob_particles

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulate Majorana zero‐mode probability in a Kitaev chain"
    )
    parser.add_argument("--N", type=int, default=50, help="number of lattice sites")
    parser.add_argument("--t", type=float, default=1.0, help="hopping amplitude")
    parser.add_argument("--mu", type=float, default=0.5, help="chemical potential")
    parser.add_argument("--delta", type=float, default=1.0, help="pairing strength")
    parser.add_argument("--interactive", action="store_true", help="Enable interactive parameter input")
    args = parser.parse_args()

    if args.interactive:
        try:
            user_input = input("Enter number of lattice sites (default 50): ").strip()
            N = int(user_input) if user_input else 50
        except ValueError:
            N = 50

        try:
            user_input = input("Enter hopping amplitude (default 1.0): ").strip()
            t = float(user_input) if user_input else 1.0
        except ValueError:
            t = 1.0

        try:
            user_input = input("Enter chemical potential (default 0.5): ").strip()
            mu = float(user_input) if user_input else 0.5
        except ValueError:
            mu = 0.5

        try:
            user_input = input("Enter pairing strength (default 1.0): ").strip()
            delta = float(user_input) if user_input else 1.0
        except ValueError:
            delta = 1.0
    else:
        N = args.N
        t = args.t
        mu = args.mu
        delta = args.delta

    energies, prob = simulate_majorana(N, t, mu, delta)

    # report zero‐mode energy
    E0 = energies[np.argmin(np.abs(energies))]
    print(f"Closest‐to‐zero eigenvalue: {E0:.3e}")

    # Create a figure with two subplots:
    # Left: full BdG energy spectrum.
    # Right: Majorana zero‐mode probability distribution.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(np.arange(len(energies)), energies, "o-", label="Eigenvalues")
    ax1.axhline(0, color="black", linestyle="--")
    ax1.set_xlabel("Eigenvalue Index")
    ax1.set_ylabel("Energy")
    ax1.set_title("BdG Energy Spectrum")
    ax1.legend()

    ax2.plot(np.arange(N), prob, "o-", label="Zero‐Mode")
    ax2.set_xlabel("Lattice Site Index")
    ax2.set_ylabel("Probability")
    ax2.set_title("Majorana Zero‐Mode Distribution")
    ax2.legend()

    plt.suptitle(f"Simulation Results (N={N}, t={t}, μ={mu}, Δ={delta})")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
