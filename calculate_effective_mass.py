# Required imports
import numpy as np
from scipy.linalg import eigh



def calculate_effective_mass(V0, a, hbar = 1.0, m = 1.0, G_max = 20, k_range = 0.1, n_k = 51):
    """
    Calculate the effective mass of an electron in a periodic potential 
    using the nearly-free electron approximation.
    
    Parameters:
    - V0      : potential strength (Fourier component of the periodic potential)
    - a       : lattice constant
    - hbar    : reduced Planck constant (default = 1.0)
    - m       : bare electron mass (default = 1.0)
    - G_max   : maximum reciprocal lattice index for plane-wave expansion
    - k_range : fraction of the Brillouin zone around k=0 to sample
    - n_k     : number of k-points for dispersion calculation
    
    Returns:
    - m_eff   : effective mass extracted from the curvature of the band
    - E0      : band edge energy at k = 0
    - k_points: array of sampled k values
    - E_k     : corresponding band energies
    """
    
    # Reciprocal lattice vectors (plane-wave basis)
    G_values = np.arange(-G_max, G_max+1) * (2*np.pi/a)
    nG       = len(G_values)
    
    # Initialize potential matrix in reciprocal space
    V_matrix = np.zeros((nG, nG), dtype = complex)
    for i, Gi in enumerate(G_values):
        for j, Gj in enumerate(G_values):
            
            delta_G = Gi - Gj
            # Only Fourier components with ΔG = ± 2π/a contribute
            if np.isclose(delta_G, 2*np.pi/a) or np.isclose(delta_G, -2*np.pi/a):
                V_matrix[i, j] = V0 / 2.0
    
    # Define k-points in the reduced Brillouin zone around k = 0
    k_points = np.linspace(-k_range*np.pi/a, k_range*np.pi/a, n_k)
    E_k      = []
    
    # Loop over k-points to construct Hamiltonian and compute eigenvalues
    for k in k_points:

        # Start with potential term
        H_k = V_matrix.copy()
        
        # Add kinetic energy contribution for each plane-wave state
        for i, Gi in enumerate(G_values):
            H_k[i, i] += (hbar**2 * (k + Gi)**2) / (2*m)
        
        # Diagonalize Hamiltonian -> energy eigenvalues
        eigenvalues, _ = eigh(H_k)
        
        # Take the lowest band (ground state energy for each k)
        E_k.append(eigenvalues[0]) 
    
    E_k = np.array(E_k)
    
    # Fit dispersion E(k) near k = 0 with a quadratic polynomial
    coeffs  = np.polyfit(k_points, E_k, 2)
    
    # Extract effective mass from curvature: m* = ħ² / (d²E/dk²)
    d2E_dk2 = 2*coeffs[0]
    m_eff   = hbar**2 / d2E_dk2
    
    # Band edge energy at k = 0
    E0      = coeffs[2] 
    
    return m_eff, E0, k_points, E_k