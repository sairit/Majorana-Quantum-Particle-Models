# Quantum Particle Simulator: Majorana Zero-Modes and Particle-in-a-Box

**Author:** Sai Yadavalli  
**Version:** 3.0

A sophisticated quantum mechanics simulation package implementing two fundamental quantum systems: Majorana zero-modes in 1D Kitaev chains and quantum particle-in-a-box with eigenstate clustering analysis, built from scratch using numerical linear algebra and machine learning techniques.

## Overview

This project implements advanced quantum mechanical simulations without relying on specialized quantum computing libraries, demonstrating deep understanding of condensed matter physics, quantum mechanics, numerical methods, and machine learning applications in physics. The simulator features two distinct quantum systems with comprehensive visualization and analysis capabilities.

## Theoretical Foundation

### Majorana Zero-Modes in Kitaev Chain

#### Bogoliubov-de Gennes (BdG) Hamiltonian
The 1D Kitaev chain is described by the BdG Hamiltonian in Nambu space:

```
H_BdG = Σᵢ [-μ(c†ᵢcᵢ - ½) - t(c†ᵢcᵢ₊₁ + h.c.) + Δ(cᵢcᵢ₊₁ + h.c.)]
```

Where:
- `μ` is the chemical potential
- `t` is the hopping amplitude
- `Δ` is the superconducting pairing strength
- The Hamiltonian is constructed in the basis `[c₁, c†₁, c₂, c†₂, ...]`

#### Matrix Representation
The 2N×2N BdG Hamiltonian matrix elements are:
- **On-site terms**: Chemical potential contributions `±μ`
- **Hopping terms**: Nearest-neighbor coupling `±t`
- **Pairing terms**: Superconducting correlations `±Δ`

#### Majorana Physics
Majorana zero-modes emerge at the boundaries when the system is in the topological phase:
- **Topological condition**: `|μ| < 2t` and `Δ ≠ 0`
- **Zero-energy states**: Eigenstates with energies exponentially close to zero
- **Exponential localization**: Probability density decays as `e^(-x/ξ)` from boundaries

### Quantum Particle-in-a-Box

#### Time-Independent Schrödinger Equation
The particle-in-a-box problem solves:

```
[-ℏ²/(2m) ∇² + V(x)]ψ(x) = Eψ(x)
```

With infinite potential walls: `V(x) = 0` for `0 < x < L`, `V(x) = ∞` elsewhere.

#### Finite Difference Discretization
The second derivative is approximated using central differences:

```
d²ψ/dx² ≈ [ψ(x+dx) - 2ψ(x) + ψ(x-dx)]/dx²
```

Creating a tridiagonal Hamiltonian matrix with eigenvalues proportional to `n²π²/(2mL²)`.

#### Machine Learning Integration
K-means clustering analyzes eigenstate symmetries by:
- **Feature vectors**: Normalized probability densities `|ψₙ(x)|²`
- **Clustering**: Groups eigenstates with similar spatial patterns
- **Symmetry classification**: Identifies even/odd parity and nodal structures

## Features

### Majorana Zero-Mode Simulation
- **BdG Hamiltonian Construction**: Automated matrix assembly for arbitrary system sizes
- **Eigenvalue Analysis**: Complete energy spectrum computation with zero-mode identification
- **Spatial Localization**: Probability density analysis of Majorana states
- **Parameter Sweeps**: Interactive exploration of topological phase transitions
- **Visualization**: Dual-panel plots showing spectrum and zero-mode distribution

### Particle-in-a-Box Analysis
- **Finite Difference Method**: High-accuracy discretization with customizable grid resolution
- **Complete Eigenspectrum**: Computation of 1000+ eigenstates for statistical analysis
- **K-Means Clustering**: Automated eigenstate classification by symmetry properties
- **Pattern Recognition**: Identification of nodal structures and spatial correlations
- **Comparative Visualization**: Energy spectrum with cluster-coded states

### Technical Capabilities
- **Command-Line Interface**: Comprehensive argument parsing with default parameters
- **Interactive Mode**: User-friendly parameter input with error handling
- **Numerical Stability**: Robust eigenvalue solvers using `numpy.linalg.eigh`
- **Memory Efficiency**: Optimized matrix operations for large-scale simulations
- **Publication-Quality Plots**: Professional matplotlib visualizations

## Key Components

### Majorana Simulation Core

#### `build_hamiltonian(N, t, mu, delta)` - BdG Matrix Construction
Assembles the complete Bogoliubov-de Gennes Hamiltonian:
- **Matrix Dimensions**: 2N×2N for N-site chain
- **Particle-Hole Structure**: Proper Nambu space representation
- **Boundary Conditions**: Open boundaries for Majorana localization
- **Parameter Encoding**: Physical parameters mapped to matrix elements

#### `simulate_majorana(N, t, mu, delta)` - Quantum State Analysis
Performs complete eigenanalysis of the Kitaev chain:
- **Eigenvalue Computation**: Full spectrum diagonalization
- **Zero-Mode Identification**: Automatic detection of near-zero energy states
- **Probability Extraction**: Spatial distribution of Majorana modes
- **Physical Interpretation**: Connection between mathematics and physics

### Particle-in-a-Box Engine

#### `simulate_particle_in_box(L, n_points, cluster_count)` - Quantum Solver
Implements comprehensive quantum mechanical analysis:
- **Hamiltonian Assembly**: Tridiagonal matrix construction via finite differences
- **Boundary Implementation**: Dirichlet conditions through grid truncation
- **Eigenstate Computation**: Complete spectrum calculation
- **Feature Engineering**: Probability density normalization for clustering

#### Machine Learning Integration
- **K-Means Clustering**: Unsupervised eigenstate classification
- **Feature Vectors**: Normalized probability densities as input features
- **Cluster Analysis**: Statistical breakdown of eigenstate categories
- **Pattern Discovery**: Automatic identification of quantum symmetries

## Mathematical Implementation

### Numerical Linear Algebra
The implementation employs advanced computational techniques:

#### Eigenvalue Problems
- **Hermitian Matrices**: Exploitation of physical symmetries for efficiency
- **LAPACK Integration**: NumPy's optimized eigenvalue routines
- **Memory Management**: Efficient storage of large matrices
- **Numerical Precision**: Double-precision arithmetic for quantum accuracy

#### Matrix Construction Algorithms
- **Sparse Structure**: Recognition of tridiagonal/banded patterns
- **Vectorized Operations**: NumPy broadcasting for performance
- **Boundary Handling**: Proper implementation of quantum boundary conditions

### Quantum Mechanical Principles

#### Normalization and Probability
- **Wavefunction Normalization**: Proper quantum mechanical interpretation
- **Probability Density**: `|ψ(x)|²` calculation and visualization
- **Conservation Laws**: Verification of probability conservation

#### Physical Parameter Mapping
- **Dimensionless Units**: Consistent parameter scaling throughout simulation
- **Phase Diagram Navigation**: Systematic parameter space exploration
- **Topological Transitions**: Identification of critical points

## Usage

### Basic Majorana Simulation
```bash
# Default parameters
python quantum_simulator.py --system majorana

# Custom parameters
python quantum_simulator.py --system majorana --N 100 --mu 0.3 --delta 1.5

# Interactive mode
python quantum_simulator.py --system majorana --interactive
```

### Particle-in-a-Box Analysis
```bash
# Default simulation
python quantum_simulator.py --system particle

# High-resolution clustering
python quantum_simulator.py --system particle --n_points 2048 --clusters 8

# Interactive parameter selection
python quantum_simulator.py --system particle --interactive
```

### Command-Line Arguments
```
--system        Choose simulation: 'majorana' or 'particle'
--N             Number of lattice sites (Majorana)
--t             Hopping amplitude (Majorana)
--mu            Chemical potential (Majorana)
--delta         Pairing strength (Majorana)
--L             Box length (Particle)
--n_points      Grid resolution (Particle)
--clusters      Number of K-means clusters (Particle)
--interactive   Enable interactive parameter input
```

## Scientific Applications

### Condensed Matter Physics
- **Topological Superconductivity**: Study of Majorana fermions in quantum wires
- **Phase Transitions**: Investigation of topological phase boundaries
- **Quantum Computing**: Analysis of topologically protected qubits
- **Experimental Guidance**: Parameter optimization for laboratory realizations

### Quantum Mechanics Education
- **Eigenvalue Problems**: Visualization of fundamental quantum concepts
- **Boundary Conditions**: Impact of confinement on quantum states
- **Symmetry Analysis**: Machine learning approach to quantum symmetries
- **Numerical Methods**: Bridge between analytical and computational physics

## Visualization Capabilities

### Majorana Analysis Plots
- **Energy Spectrum**: Complete BdG eigenvalue distribution
- **Zero-Mode Localization**: Spatial probability density of Majorana states
- **Parameter Dependence**: Real-time visualization during parameter sweeps
- **Phase Diagrams**: Topological phase boundary identification

### Particle-in-a-Box Visualizations
- **Clustered Spectrum**: Energy eigenvalues color-coded by symmetry class
- **Representative Eigenstates**: Probability densities for each cluster
- **Pattern Recognition**: Visual identification of nodal structures
- **Statistical Analysis**: Cluster population distributions

## Requirements

```
numpy>=1.20.0
matplotlib>=3.3.0
scikit-learn>=1.0.0
argparse>=1.4.0
```

## Educational Value

This implementation demonstrates expertise in:

### Quantum Mechanics
- **Advanced Topics**: Majorana fermions, topological superconductivity, BdG formalism
- **Numerical Methods**: Finite difference schemes, eigenvalue algorithms
- **Physical Interpretation**: Connection between mathematical formalism and physics
- **Boundary Conditions**: Proper implementation of quantum mechanical constraints

### Computational Physics
- **Linear Algebra**: Large-scale eigenvalue problems and matrix diagonalization
- **Numerical Stability**: Robust algorithms for challenging quantum problems
- **Performance Optimization**: Efficient memory usage and computational speed
- **Visualization**: Scientific plotting and data presentation

### Machine Learning in Physics
- **Unsupervised Learning**: K-means clustering for quantum state classification
- **Feature Engineering**: Transformation of quantum states into ML-compatible formats
- **Pattern Recognition**: Automatic discovery of physical symmetries
- **Interdisciplinary Methods**: Integration of ML with fundamental physics

### Software Engineering
- **Command-Line Tools**: Professional argument parsing and user interfaces
- **Code Organization**: Modular design with clear separation of concerns
- **Error Handling**: Robust input validation and graceful failure modes
- **Documentation**: Comprehensive inline comments and docstrings

## Physical Insights

### Majorana Zero-Modes
- **Topological Protection**: Understanding of topologically protected quantum states
- **Experimental Relevance**: Connection to ongoing quantum computing research
- **Many-Body Physics**: Implementation of sophisticated quantum many-body theories
- **Phase Transitions**: Identification of topological phase boundaries

### Quantum Confinement
- **Eigenstate Structure**: Systematic analysis of confined quantum states
- **Symmetry Properties**: Machine learning discovery of quantum symmetries
- **Statistical Physics**: Large-scale eigenstate analysis and pattern recognition
- **Computational Scaling**: Handling of high-dimensional quantum problems

## Future Enhancements

- [ ] 2D Kitaev model implementation
- [ ] Time evolution capabilities (TDSE solver)
- [ ] Disorder and impurity effects
- [ ] Transport property calculations
- [ ] GPU acceleration for large systems
- [ ] Interactive 3D visualizations
- [ ] Export capabilities for experimental comparison

---

This implementation represents a sophisticated fusion of quantum mechanics, condensed matter physics, numerical methods, and machine learning, showcasing advanced computational physics skills and deep understanding of quantum many-body systems.
