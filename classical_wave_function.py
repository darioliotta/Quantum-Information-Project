# Required imports
import numpy as np
import matplotlib.pyplot as plt
import ufl
from mpi4py      import MPI
from dolfinx     import mesh, fem
from dolfinx.fem import petsc as fem_petsc
from petsc4py    import PETSc



def classical_wave_function(L, nx, T, dt, c, f, bc = "Dirichlet", n_plots = 6):
    """
    Finite Element Method (FEM) simulation of the 1D classical wave equation.

    Parameters:
    -----------
    L        : float
        Length of the spatial domain.
    nx       : int
        Number of elements for mesh discretization.
    T        : float
        Final simulation time.
    dt       : float
        Time step size.
    c        : float, callable, or array
        Wave propagation velocity. Can be constant or space-dependent.
    f        : callable
        Function defining the initial condition u(x,0).
    bc       : str, optional
        Type of boundary condition: "Dirichlet" (default) or "Neumann".
    n_plots  : int, optional
        Number of solution snapshots to be plotted.
    """

    # ----------------------------------------------------------------------
    # 1. Domain and function space definition
    # ----------------------------------------------------------------------
    domain = mesh.create_interval(MPI.COMM_WORLD, nx, [0.0, L])
    V      = fem.functionspace(domain, ("Lagrange", 1))
    
    # ----------------------------------------------------------------------
    # 2. Boundary conditions
    # ----------------------------------------------------------------------
    bc_list = []
    
    if bc == "Neumann":
        # Homogeneous Neumann: no boundary conditions are explicitly imposed
        bc_list = []
    
    elif bc == "Dirichlet":
        # Homogeneous Dirichlet: enforce u = 0 at x = 0 and x = L
        u_bc            = fem.Function(V)
        u_bc.x.array[:] = 0.0
        
        facets = mesh.locate_entities_boundary(
            domain,
            dim    = 0,
            marker = lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], L)
        )
        
        dofs    = fem.locate_dofs_topological(V, entity_dim = 0, entities = facets)
        bc_list = [fem.dirichletbc(u_bc, dofs)]
        
    # ----------------------------------------------------------------------
    # 3. Initial conditions
    # ----------------------------------------------------------------------
    u_past    = fem.Function(V)   # solution at t_{n-1}
    u_present = fem.Function(V)   # solution at t_{n}
    u_future  = fem.Function(V)   # solution at t_{n+1}
    
    # Apply initial displacement f(x) and set initial velocity = 0
    x_coords             = V.tabulate_dof_coordinates()[:, 0]
    u_past.x.array[:]    = f(x_coords)
    u_present.x.array[:] = u_past.x.array
    
    # ----------------------------------------------------------------------
    # 4. Wave velocity handling (constant or space-dependent)
    # ----------------------------------------------------------------------
    if np.isscalar(c):
        # Case A: constant velocity
        c2   = fem.Constant(domain, PETSc.ScalarType(c**2))
        wall = False
        v_max = c
       
    else:
        # Case B: spatially varying velocity
        c_fun = fem.Function(V)
        wall  = True
        
        if callable(c):
            # velocity given as a function of space
            c_fun.interpolate(lambda x: c(x[0])**2)
        else:
            # velocity provided as array values
            c_fun.x.array[:] = np.asarray(c(x_coords))**2
            
        c2 = c_fun
        v_max = np.sqrt(np.max([c_fun.x.array[0], c_fun.x.array[-1]]))
        
    # ----------------------------------------------------------------------
    # 5. CFL stability check
    # ----------------------------------------------------------------------
    CFL = v_max*dt / (L/nx)
    print(f"CFL = {CFL:.3f}\n")
    
    # ----------------------------------------------------------------------
    # 6. Weak formulation and matrix assembly
    # ----------------------------------------------------------------------
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Bilinear form for the system matrix
    a_form = (u*v / dt**2) * ufl.dx + c2 * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    A      = fem_petsc.assemble_matrix(fem.form(a_form), bcs = bc_list)
    A.assemble()
    
    # Solver setup (LU factorization)
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType("preonly")
    solver.getPC().setType("lu")
    
    # ----------------------------------------------------------------------
    # 7. Time discretization
    # ----------------------------------------------------------------------
    nt      = int(T / dt) + 1
    t_vals  = np.linspace(0.0, T, nt)         # all time steps
    t_plots = np.linspace(0.0, T, n_plots)    # selected plot times
    
    # ----------------------------------------------------------------------
    # 8. Transmission/reflection setup (for heterogeneous medium)
    # ----------------------------------------------------------------------
    if wall == True:
        i_mid   = np.argmin(np.abs(x_coords - L/2))
        i_left  = np.argmin(np.abs(x_coords - L/4))       
        i_right = np.argmin(np.abs(x_coords - 3*L/4))
        
        t_cross       = None
        t_transmitted = None
        t_reflected   = None
        
        initial_amplitude     = None
        transmitted_amplitude = None
        reflected_amplitude   = None
        
        threshold = 0.05
    else:
        t_cross = None
        
    # ----------------------------------------------------------------------
    # 9. Energy storage initialization
    # ----------------------------------------------------------------------
    energies = []
    
    # Prepare plot
    plt.figure(figsize = (10, 6))
    
    # ----------------------------------------------------------------------
    # 10. Time-stepping loop
    # ----------------------------------------------------------------------
    for t_current in t_vals:

        # --- Plot and compute energy at selected time steps ---
        if np.any(np.isclose(t_current, t_plots, atol = dt / 2)):
            
            # Plot current solution snapshot
            plt.plot(x_coords, u_present.x.array, label = f"t = {t_current:.2f}")
            
            # Compute discrete energy
            u_t            = fem.Function(V)
            u_t.x.array[:] = (u_present.x.array - u_past.x.array) / dt
            
            energy_form = 0.5 * ((u_t**2) / c2 + ufl.inner(ufl.grad(u_present), ufl.grad(u_present))) * ufl.dx
            E           = fem.assemble_scalar(fem.form(energy_form))
            
            if t_current == 0.0:
                E0 = E  # reference energy at t=0
                
            print(f"Energy at time {t_current:.3f}: {E/E0:.3f}")
            energies.append(E/E0)

        # Skip final step to avoid redundancy
        if np.isclose(t_current, T):
            break

        # --- Assemble right-hand side ---
        L_form = (2.0 * u_present * v / dt**2 - u_past * v / dt**2) * ufl.dx
        b      = fem_petsc.assemble_vector(fem.form(L_form))
        fem_petsc.apply_lifting(b, [fem.form(a_form)], bcs = [bc_list])
        b.ghostUpdate(addv = PETSc.InsertMode.ADD, mode = PETSc.ScatterMode.REVERSE)
        fem_petsc.set_bc(b, bc_list)

        # --- Solve linear system ---
        solver.solve(b, u_future.x.petsc_vec)
        u_future.x.scatter_forward()

        # --- Update time levels ---
        u_past.x.array[:]    = u_present.x.array
        u_present.x.array[:] = u_future.x.array
        
        # --- Transmission/reflection analysis ---
        if wall == True:
            umax = np.max(np.abs(u_present.x.array))
            
            # Detect wave crossing midpoint
            if t_cross is None and np.abs(u_present.x.array[i_mid]) > threshold * umax:
                t_cross           = t_current
                initial_amplitude = umax
                
            # Detect transmission
            if t_cross is not None and transmitted_amplitude is None:
                u_mid   = u_present.x.array[i_mid]
                u_right = u_present.x.array[i_right]
                
                if np.isclose(u_mid, u_right, rtol = 1e-2):
                    transmitted_amplitude = np.max(np.abs(u_present.x.array[i_mid:i_right+1]))
                    t_transmitted         = t_current
            
            # Detect reflection
            if t_cross is not None and reflected_amplitude is None:
                u_mid  = u_present.x.array[i_mid]
                u_left = u_present.x.array[i_left]
                
                if np.isclose(u_mid, u_left, rtol = 1e-2):
                    reflected_amplitude = np.max(np.abs(u_present.x.array[i_left:i_mid+1]))
                    t_reflected         = t_current
                    
    # ----------------------------------------------------------------------
    # 11. Post-processing for transmission/reflection
    # ----------------------------------------------------------------------
    if t_cross is None:
        wall = False
        if np.isscalar(c) == False:
            print("\nThe time was not enough for the wave to cross the wall")

    # Fresnel coefficient estimation and comparison with theory
    if wall == True and initial_amplitude is not None:
        
        print("\nTimes:")
        print(f"   Crossing the wall: t = {t_cross:.3f}")
        
        if transmitted_amplitude is not None:
            print(f"   Transmission:      t = {t_transmitted:.3f}")
        else:
            print("   Not enough time for transmission")
        
        if reflected_amplitude is not None:
            print(f"   Reflection:        t = {t_reflected:.3f}")
        else:
            print("   Not enough time for reflection")
        
        print("\nAmplitudes:")
        print(f"   Initial:     A = {initial_amplitude:.3f}")
        
        if transmitted_amplitude is not None:
            print(f"   Transmitted: A = {transmitted_amplitude:.3f}")
        else:
            print("   Not enough time for transmission")
            
        if reflected_amplitude is not None:
            print(f"   Reflected:   A = {reflected_amplitude:.3f}")
        else:
            print("   Not enough time for reflection")
        
        # Numerical Fresnel coefficients
        if transmitted_amplitude is not None:
            t_coeff = transmitted_amplitude / initial_amplitude
        
        if reflected_amplitude is not None:
            r_coeff = reflected_amplitude / initial_amplitude
            # Correct sign convention depending on velocity profile
            if c_fun.x.array[0] < c_fun.x.array[-1]:
                r_coeff *= -1
                
        if transmitted_amplitude is not None and reflected_amplitude is not None:
            print(f"\nEstimated Fresnel coefficients: T = {t_coeff}, R = {r_coeff}")
        
        # Analytical Fresnel coefficients
        t_coeff_correct = 2*np.sqrt(c_fun.x.array[0]) / (np.sqrt(c_fun.x.array[0]) + np.sqrt(c_fun.x.array[-1]))
        r_coeff_correct = (np.sqrt(c_fun.x.array[0]) - np.sqrt(c_fun.x.array[-1])) / (np.sqrt(c_fun.x.array[0]) + np.sqrt(c_fun.x.array[-1]))
        print(f"Correct Fresnel Coefficients:   T = {t_coeff_correct}, R = {r_coeff_correct:}\n")
        
        # Mark interface position on the plot
        plt.axvline(L/2, color = "k", linestyle = "--", alpha = 0.5, label = "Interface")
        
    # ----------------------------------------------------------------------
    # 12. Final plot settings
    # ----------------------------------------------------------------------
    plt.xlabel("x")
    plt.ylabel("u(x,t)")
    plt.title(f"FEM reconstruction of the wave function evolution with {bc} boundary conditions")
    plt.legend()
    plt.grid(True)
    plt.show()