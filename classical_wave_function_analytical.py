# Required imports
import numpy as np
import matplotlib.pyplot as plt



def classica_wave_function_analytical(L, nx, T, c, f, N_modes = 50, bc = "Dirichlet", n_plots = 6):
    """
    Analytical solution for the classical wave equation using Fourier expansion.
    
    Parameters:
    - L        : length of the spatial domain
    - nx       : number of spatial discretization points
    - T        : maximum time for the simulation
    - c        : wave propagation speed
    - f        : initial function f(x) defining the wave profile
    - N_modes  : number of Fourier modes used in the expansion (default = 50)
    - bc       : boundary condition type ("Dirichlet" or "Neumann")
    - n_plots  : number of time snapshots to plot (default = 6)
    """
    
    # Spatial discretization
    x = np.linspace(0, L, nx)          # grid points in space
    function_values = f(x)             # evaluate initial condition at spatial grid
    
    # Fourier expansion coefficients depending on boundary conditions
    if bc == "Dirichlet":
        A = np.zeros(N_modes)
        for n in range(1, N_modes + 1):
            # Sine series expansion coefficients
            A[n-1] = 2/L * np.trapezoid(function_values * np.sin(n * np.pi * x / L), x)

    elif bc == "Neumann":
        A = np.zeros(N_modes + 1)
        # First term (constant mode)
        A[0] = 1/L * np.trapezoid(function_values, x) 
        for n in range(1, N_modes + 1):
            # Cosine series expansion coefficients
            A[n] = 2/L * np.trapezoid(function_values * np.cos(n * np.pi * x / L), x)
       
    # Times at which the wave function will be plotted
    t_plots = np.linspace(0.0, T, n_plots)
       
    # Create the figure for plotting
    plt.figure(figsize = (9, 6))
    
    # Loop over selected times and compute wave evolution
    for t in t_plots:
        u = np.zeros_like(x)   # initialize wave profile at time t
        
        if bc == "Dirichlet":
            # Expansion in sine functions
            for n in range(1, N_modes + 1):
                omega_n = n * np.pi * c / L
                u += A[n-1] * np.cos(omega_n * t) * np.sin(n * np.pi * x / L)

        elif bc == "Neumann":
            # Expansion in cosine functions
            u += A[0]   # constant mode contribution
            for n in range(1, N_modes + 1):
                omega_n = n * np.pi * c / L
                u += A[n] * np.cos(omega_n * t) * np.cos(n * np.pi * x / L)
            
        # Plot wave profile at this time
        plt.plot(x, u, label = f't = {t:.2f}')
     
    # Final plot configuration   
    plt.xlabel("x")
    plt.ylabel("u(x,t)")
    plt.title(f"Analytical wave function evolution with {bc} boundary conditions")
    plt.legend()
    plt.grid(True, alpha = 0.3)
    
    plt.show()