import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.cluster import KMeans

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

    # picking the eigenvector whose energy is closest to zero
    idx_zero = np.argmin(np.abs(energies))
    zero_mode = modes[:, idx_zero]

    # probability density
    prob = np.abs(zero_mode)**2
    # taking only the particle part (even indices)
    prob_particles = prob[::2]

    return energies, prob_particles

def simulate_particle_in_box(L, n_points, cluster_count=5):
    """
    Constructs the Hamiltonian for a particle in a 1D box (Dirichlet boundaries) using finite differences.
    Computes eigenstates for 1000+ states and applies K-means clustering on the probability densities of eigenstates.
    
    Returns:
      energies: computed eigenvalues
      eigvecs: computed eigenvectors (each column corresponds to an eigenstate)
      cluster_labels: cluster label for each eigenstate after K-means clustering
      x: spatial grid points corresponding to eigenstates
    """
    # Discretize the box: interior points only (Dirichlet BC)
    dx = L / (n_points + 1)
    x = np.linspace(dx, L - dx, n_points)

    # Building Hamiltonian using central finite difference for the second derivative
    H = np.zeros((n_points, n_points))
    diag = 1.0 / (dx**2)
    off_diag = -1.0 / (2*dx**2)
    for i in range(n_points):
        H[i, i] = diag
        if i > 0:
            H[i, i-1] = off_diag
        if i < n_points - 1:
            H[i, i+1] = off_diag

    # Solving eigenvalue problem
    energies, eigvecs = np.linalg.eigh(H)

    # Building feature vectors using probability density (eigenstate squared amplitudes)
    # Normalizing each eigenstate's probability density.
    features = np.abs(eigvecs)**2
    features = (features.T / np.linalg.norm(features, axis=0)).T

    # Applying K-means clustering to group eigenstates into distinct symmetry clusters
    kmeans = KMeans(n_clusters=cluster_count, random_state=0)
    cluster_labels = kmeans.fit_predict(features.T)

    return energies, eigvecs, cluster_labels, x

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulate Majorana zero‐mode or quantum particle in a box with eigenvalue solvers and clustering"
    )
    parser.add_argument("--system", type=str, default="majorana", choices=["majorana", "particle"],
                        help="Choose the simulation system: 'majorana' (default) or 'particle'")
    # Majorana arguments
    parser.add_argument("--N", type=int, default=50, help="number of lattice sites for Majorana simulation")
    parser.add_argument("--t", type=float, default=1.0, help="hopping amplitude")
    parser.add_argument("--mu", type=float, default=0.5, help="chemical potential")
    parser.add_argument("--delta", type=float, default=1.0, help="pairing strength")
    # Particle in a box arguments
    parser.add_argument("--L", type=float, default=1.0, help="Length of the box")
    parser.add_argument("--n_points", type=int, default=1024, help="Number of spatial discretization points")
    parser.add_argument("--clusters", type=int, default=5, help="Number of clusters for K-means clustering")
    parser.add_argument("--interactive", action="store_true", help="Enable interactive parameter input")
    args = parser.parse_args()

    if args.interactive:
        if args.system == "majorana":
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

            energies, prob = simulate_majorana(N, t, mu, delta)

            # report zero‐mode energy
            E0 = energies[np.argmin(np.abs(energies))]
            print(f"Closest‐to‐zero eigenvalue: {E0:.3e}")

            # Creating a figure with two subplots:
            # Left: full BdG energy spectrum.
            # Right: Majorana zero‐mode probability distribution.
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            ax1.plot(np.arange(len(energies)), energies, "o-", label="Eigenvalues")
            ax1.axhline(0, color="black", linestyle="--")
            ax1.set_xlabel("Eigenvalue Index")
            ax1.set_ylabel("Energy")
            ax1.set_title("BdG Energy Spectrum")
            ax1.legend()

            ax2.plot(np.arange(len(prob)), prob, "o-", label="Zero‐Mode")
            ax2.set_xlabel("Lattice Site Index")
            ax2.set_ylabel("Probability")
            ax2.set_title("Majorana Zero‐Mode Distribution")
            ax2.legend()

            plt.suptitle(f"Majorana Simulation (N={N}, t={t}, μ={mu}, Δ={delta})")
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

        elif args.system == "particle":
            try:
                user_input = input(f"Enter length of box (default {args.L}): ").strip()
                L = float(user_input) if user_input else args.L
            except ValueError:
                L = args.L

            try:
                user_input = input(f"Enter number of grid points (default {args.n_points}): ").strip()
                n_points = int(user_input) if user_input else args.n_points
            except ValueError:
                n_points = args.n_points

            energies, eigvecs, cluster_labels, x = simulate_particle_in_box(L, n_points, cluster_count=args.clusters)
            print("Eigenstate clustering counts:")
            for cluster_id in range(args.clusters):
                count = np.sum(cluster_labels == cluster_id)
                print(f"  Cluster {cluster_id}: {count} eigenstates")

            # Plot: energy spectrum with cluster labels indicated on a scatter plot.
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            ax1.scatter(np.arange(len(energies)), energies, c=cluster_labels, cmap="viridis", s=5)
            ax1.set_xlabel("Eigenstate Index")
            ax1.set_ylabel("Energy")
            ax1.set_title("Particle in a Box: Energy Spectrum with Cluster Labels")

            # For visualization, plot one representative probability density per cluster.
            for cluster_id in range(args.clusters):
                idx = np.where(cluster_labels == cluster_id)[0][0]  # select first eigenstate in this cluster
                ax2.plot(x, np.abs(eigvecs[:, idx])**2, label=f"Cluster {cluster_id}")
            ax2.set_xlabel("Position x")
            ax2.set_ylabel("Probability Density")
            ax2.set_title("Representative Eigenstates per Cluster")
            ax2.legend()

            plt.suptitle(f"Particle in a Box Simulation (L={L}, n_points={n_points}, clusters={args.clusters})")
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

    else:
        if args.system == "majorana":
            N = args.N
            t = args.t
            mu = args.mu
            delta = args.delta
            energies, prob = simulate_majorana(N, t, mu, delta)
            E0 = energies[np.argmin(np.abs(energies))]
            print(f"Closest‐to‐zero eigenvalue: {E0:.3e}")

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            ax1.plot(np.arange(len(energies)), energies, "o-", label="Eigenvalues")
            ax1.axhline(0, color="black", linestyle="--")
            ax1.set_xlabel("Eigenvalue Index")
            ax1.set_ylabel("Energy")
            ax1.set_title("BdG Energy Spectrum")
            ax1.legend()

            ax2.plot(np.arange(len(prob)), prob, "o-", label="Zero‐Mode")
            ax2.set_xlabel("Lattice Site Index")
            ax2.set_ylabel("Probability")
            ax2.set_title("Majorana Zero‐Mode Distribution")
            ax2.legend()

            plt.suptitle(f"Majorana Simulation (N={N}, t={t}, μ={mu}, Δ={delta})")
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

        elif args.system == "particle":
            L = args.L
            n_points = args.n_points
            energies, eigvecs, cluster_labels, x = simulate_particle_in_box(L, n_points, cluster_count=args.clusters)
            print("Eigenstate clustering counts:")
            for cluster_id in range(args.clusters):
                count = np.sum(cluster_labels == cluster_id)
                print(f"  Cluster {cluster_id}: {count} eigenstates")

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            ax1.scatter(np.arange(len(energies)), energies, c=cluster_labels, cmap="viridis", s=5)
            ax1.set_xlabel("Eigenstate Index")
            ax1.set_ylabel("Energy")
            ax1.set_title("Particle in a Box: Energy Spectrum with Cluster Labels")

            for cluster_id in range(args.clusters):
                idx = np.where(cluster_labels == cluster_id)[0][0]
                ax2.plot(x, np.abs(eigvecs[:, idx])**2, label=f"Cluster {cluster_id}")
            ax2.set_xlabel("Position x")
            ax2.set_ylabel("Probability Density")
            ax2.set_title("Representative Eigenstates per Cluster")
            ax2.legend()

            plt.suptitle(f"Particle in a Box Simulation (L={L}, n_points={n_points}, clusters={args.clusters})")
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()
