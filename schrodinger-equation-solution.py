# Required imports
import numpy as np
import matplotlib.pyplot as plt
import ufl
from mpi4py                   import MPI
from dolfinx                  import mesh, fem
from dolfinx.fem              import petsc as fem_petsc
from petsc4py                 import PETSc
from calculate_effective_mass import calculate_effective_mass



def schrodinger_equation_solution(L, nx, x0, sigma, k0, T, dt, hbar = 1.0, m = 1.0, bc = "Dirichlet", potential_type = "Free-particle", V0 = 1.0, a = 1.0, F = 1.0, n_plots = 6):
    """
    Time evolution of a 1D quantum wavepacket using finite elements and a
    Crank-Nicolson-like splitting for real/imaginary parts.

    The function evolves a Gaussian wavepacket (center x0, width sigma, mean
    momentum k0) under different potential types (free, periodic, linear).
    Observables (probability density, mean position, variance) are printed
    and plotted at selected times.

    Notes:
    - The real and imaginary parts of the wavefunction are stored in two FEM
      functions (u_present, v_present).
    - A block linear system for the update is assembled using PETSc nested
      matrices and solved with LU (preonly + lu).
    """

    # ------------------------------------------------------------------
    # 1) Domain and function space
    # ------------------------------------------------------------------
    # Create a 1D interval mesh on [-L/2, L/2] and a first-order Lagrange
    # function space W. This defines the degrees of freedom where the wave
    #function real/imag parts are represented.
    domain = mesh.create_interval(MPI.COMM_WORLD, nx, [-L/2, L/2]) 
    W      = fem.functionspace(domain, ("Lagrange", 1))
    
    # ------------------------------------------------------------------
    # 2) Boundary conditions: Dirichlet, Neumann, or Absorbing
    # ------------------------------------------------------------------
    # - Dirichlet: enforce u = 0 on selected boundary facets (homogeneous).
    # - Neumann: nothing to impose explicitly here (natural BC).
    # - Absorbing: treated later by adding an imaginary absorbing potential.
    if bc == "Dirichlet":
        
        u_bc            = fem.Function(W)
        u_bc.x.array[:] = 0.0
        
        facets = mesh.locate_entities_boundary(
            domain,
            dim    = 0,
            marker = lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], L)
        )
        
        dofs    = fem.locate_dofs_topological(W, entity_dim = 0, entities = facets)
        bc_list = [fem.dirichletbc(u_bc, dofs)]
        
    elif bc == "Neumann":
        
        bc_list = []
        
    elif bc == "Absorbing":
        
        bc_list = []
        
    # ------------------------------------------------------------------
    # 3) Initial condition: Gaussian wavepacket (real and imaginary parts)
    # ------------------------------------------------------------------
    # The initial complex wavefunction psi0 is a normalized Gaussian multiplied
    # by a plane-wave factor exp(i k0 x). We store the real and imaginary parts
    # separately in two FEM functions for time evolution.
    u_present = fem.Function(W)
    v_present = fem.Function(W)
    
    x_coords             = W.tabulate_dof_coordinates()[:, 0]
    psi0                 = 1 / ((np.pi * (sigma**2)) ** (1/4)) * np.exp(-(x_coords - x0)**2 / (2 * sigma**2)) * np.exp(1j * k0 * x_coords)
    u_present.x.array[:] = np.real(psi0)
    v_present.x.array[:] = np.imag(psi0)
    
    # ------------------------------------------------------------------
    # 4) Variational trial/test functions and basic forms
    # ------------------------------------------------------------------
    # Build Trial and Test functions for assembling mass and stiffness matrices.
    u = ufl.TrialFunction(W)
    w = ufl.TestFunction(W)
    
    # Mass matrix M = ∫ u w dx  (discretization of identity in weak form)
    M_form = u * w * ufl.dx
    M      = fem_petsc.assemble_matrix(fem.form(M_form), bcs = bc_list)
    M.assemble()
    
    # Kinetic (stiffness) matrix K = ∫ ∇u · ∇w dx
    K_form = ufl.inner(ufl.grad(u), ufl.grad(w)) * ufl.dx
    K      = fem_petsc.assemble_matrix(fem.form(K_form), bcs = bc_list)
    K.assemble()
    
    # ------------------------------------------------------------------
    # 5) Potential definition and assembly
    # ------------------------------------------------------------------
    # Three supported potential types:
    # - "Free-particle": V(x) = 0
    # - "Periodic"     : V(x) = V0 * cos(2π x / a)
    # - "Linear"       : V(x) = V0 * cos(2π x / a) + F * x  (tilted periodic)
    if potential_type == "Free-particle":
        V_expr = fem.Constant(domain, 0.0)
        
    elif potential_type == "Periodic":
        x      = ufl.SpatialCoordinate(domain)[0]
        V_expr = V0 * ufl.cos(2 * np.pi * x / a)
        
    elif potential_type == "Linear":
        x      = ufl.SpatialCoordinate(domain)[0]
        V_expr = (V0 * ufl.cos(2 * np.pi * x / a)) + (F * x)
    
    V_form   = V_expr * u * w * ufl.dx
    V_matrix = fem_petsc.assemble_matrix(fem.form(V_form), bcs = bc_list)
    V_matrix.assemble()
    
    # ------------------------------------------------------------------
    # 6) Absorbing boundary treatment (complex absorbing potential)
    # ------------------------------------------------------------------
    # If user requested absorbing boundaries, define an imaginary potential eta(x)
    # that smoothly increases near the domain edges and assemble its matrix.
    if bc == "Absorbing":
        x      = ufl.SpatialCoordinate(domain)[0]
        width  = 0.25 * L
        eta    = V0 * ufl.conditional(ufl.ge(ufl.sqrt(x**2), L/2 - width),
                                    ((ufl.sqrt(x**2) - (L/2 - width)) / width)**4,
                                    0.0)

        Abs_form   = eta * u * w * ufl.dx
        Abs_matrix = fem_petsc.assemble_matrix(fem.form(Abs_form), bcs = bc_list)
        Abs_matrix.assemble()
    
    # ------------------------------------------------------------------
    # 7) Time-stepping coefficients and Hamiltonian assembly
    # ------------------------------------------------------------------
    # Using a splitting that results in matrices A and B for the (implicit)
    # linear solve: alpha = dt/(2 ħ), beta = ħ/(2 m).
    alpha = dt / (2.0*hbar)
    beta  = hbar / (2.0*m)
    
    # Hamiltonian operator H ≈ - (ħ² / 2m) ∇² + V
    H = K.copy()
    H.scale(-beta)
    H.axpy(1.0, V_matrix)
    
    # Matrices entering the linear solve (Crank–Nicolson style for real/imag parts)
    A = M.copy()
    A.aypx(alpha, H) 
     
    B = M.copy()
    B.axpy(-alpha, H)
    
    # ------------------------------------------------------------------
    # 8) Initial state vector (concatenate real and imaginary parts)
    # ------------------------------------------------------------------
    # psi = [Re(ψ); Im(ψ)] as a single vector for PETSc block system operations
    psi = np.concatenate([u_present.x.array, v_present.x.array])
        
    # ------------------------------------------------------------------
    # 9) Nested PETSc block matrices for the coupled real/imag system
    # ------------------------------------------------------------------
    # Build 2x2 block matrices for the system A_block and B_block. When an
    # absorbing potential is present, it enters the diagonal blocks.
    if bc == "Absorbing":
        A_block = PETSc.Mat().createNest([[M + alpha*Abs_matrix, -alpha*H],
                                         [alpha*H, M + alpha*Abs_matrix]], comm = domain.comm)
        B_block = PETSc.Mat().createNest([[M - alpha*Abs_matrix, alpha*H],
                                         [-alpha*H, M - alpha*Abs_matrix]], comm = domain.comm)
    else:
        A_block = PETSc.Mat().createNest([[M, -alpha*H],
                                         [alpha*H, M]], comm = domain.comm)
        B_block = PETSc.Mat().createNest([[M, alpha*H],
                                         [-alpha*H, M]], comm = domain.comm)

    A_block.assemble()
    B_block.assemble()
    
    # ------------------------------------------------------------------
    # 10) Linear solver setup (PETSc KSP with LU)
    # ------------------------------------------------------------------
    ksp = PETSc.KSP().create(domain.comm)
    ksp.setOperators(A_block)
    ksp.setType('preonly')
    ksp.getPC().setType('lu')
    
    # ------------------------------------------------------------------
    # 11) PETSc vectors for RHS and solution, and a temporary vector
    # ------------------------------------------------------------------
    b = PETSc.Vec().create(comm = domain.comm)
    b.setSizes(2 * len(u_present.x.array))
    b.setUp()
    
    x = PETSc.Vec().create(comm = domain.comm)
    x.setSizes(2 * len(u_present.x.array))
    x.setUp()
    
    temp_vec = PETSc.Vec().createWithArray(psi.copy(), comm = domain.comm)
    
    # ------------------------------------------------------------------
    # 12) Time discretization and plotting setup
    # ------------------------------------------------------------------
    nt      = int(T / dt) + 1
    t_vals  = np.linspace(0.0, T, nt)
    t_plots = np.linspace(0.0, T, n_plots)
    
    # If periodic potential is used, compute an effective mass and band edge
    # (useful to compare spreading rate with effective-mass approximation)
    if potential_type == "Periodic":
        m_eff, E0, _, _ = calculate_effective_mass(V0, a, hbar, m, G_max = L/2)
        print(f"Effective mass:     {m_eff:.3f}")
        print(f"Fundamental energy: {E0:.3f}\n")
    
    plt.figure(figsize = (12,6))
    
    # ------------------------------------------------------------------
    # 13) Time loop: update, solve, extract observables, and plot snapshots
    # ------------------------------------------------------------------
    # For each time step:
    #  - compute b = B_block * psi_old
    #  - solve A_block * psi_new = b
    #  - unpack psi_new into real/imag parts
    #  - compute probability density, normalization, mean and variance
    #  - at selected times, print and plot observables
    for t_current in t_vals:

        B_block.mult(temp_vec, b)
        
        ksp.solve(b, x)
        
        temp_vec.setArray(x.getArray())
        psi[:] = x.getArray()
        
        u_present.x.array[:] = psi[:len(u_present.x.array)]
        v_present.x.array[:] = psi[len(u_present.x.array):]
        
        prob_density = u_present.x.array**2 + v_present.x.array**2
        norm         = np.sum(prob_density)

        x_mean  = np.sum(x_coords * prob_density) / norm
        
        x2_mean = np.sum((x_coords**2) * prob_density) / norm
        sigma_t = np.sqrt(x2_mean - x_mean**2)

        if np.any(np.isclose(t_current, t_plots, atol = dt / 2)):
            
            # Print average position for tilted potentials
            if potential_type == "Linear":
                print(f"Average: {x_mean:.3f}")
            
            # Expected spreading formulas:
            # - Free-particle / Linear: analytical expected sigma(t) from free
            #   evolution (showing comparison).
            # - Periodic: use effective mass m_eff in the spreading estimate.
            if (potential_type == "Free-particle") | (potential_type == "Linear"):
                sigma_expected = np.sqrt((sigma**2) / 2 + ((hbar * t_current) / (np.sqrt(2) * m * sigma))**2)
                
            elif potential_type == "Periodic":
                sigma_expected = (np.sqrt((sigma**2) / 2) + ((hbar * t_current) / (np.sqrt(2) * m_eff * sigma))**2)
                
            print(f"Variance at time {t_current:.3f}: {sigma_t:.3f} (Expected: {sigma_expected:.3f})")
            
            plt.plot(x_coords, prob_density, label = f"t = {t_current:.3f}")
            
    # ------------------------------------------------------------------
    # 14) Mark periodic lattice sites on the plot (if applicable)
    # ------------------------------------------------------------------
    if potential_type == "Periodic":

        n_max = int(L/(2*a))
        positions = []
        
        for n in range(-n_max - 1, n_max + 1):
            pos = n*a + a/2 
            if -L/2 <= pos <= L/2:
                positions.append(pos)
        
        for pos in positions:
            plt.axvline(x = pos, color = 'red', alpha = 0.65, linewidth = 0.75)
            
    # ------------------------------------------------------------------
    # 15) Final plotting details
    # ------------------------------------------------------------------
    plt.xlabel("x")
    plt.ylabel(r"$|\psi(x,t)|^2$")
    plt.title(f"Wave function evolution with {bc} boundary conditions")
    plt.legend()
    plt.grid(True)

    plt.show()