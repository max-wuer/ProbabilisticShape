import dolfin as df

def SolvePoisson(MyMesh, inside_domains, outside_domains, interface_id, bounding_box, smooth, degree=1):
    mesh = MyMesh.mesh
    dx = MyMesh.dx
    ds = MyMesh.ds
    #define function spaces
    CG1 = df.FunctionSpace(mesh, "CG", degree)
    v = df.TestFunction(CG1)
    u = df.TrialFunction(CG1)
    #set BCs to measure distance from inner interface
    bc = []
    for i in interface_id:
        bc.append(df.DirichletBC(CG1, 0.0, MyMesh.boundaries, i))
    #bc = df.DirichletBC(CG1, 0.0, "on_boundary")
    #start with Laplace
    f1 = df.Constant(smooth)*df.inner(df.grad(v), df.grad(u))*dx
    l1 = v*df.Constant(1.0)*dx(outside_domains)
    l1 += v*df.Constant(1.0)*dx(inside_domains)
    d = df.Function(CG1)
    d.rename("Laplace", "")
    
    #p1 = df.LinearVariationalProblem(f1, l1, d, bc)
    #solver1 = df.LinearVariationalSolver(p1)
    #prm = solver1.parameters
    #prm["linear_solver"] = "cg"
    #prm["preconditioner"] = "sor"
    
    #make linear solver less picky...
    #prm["krylov_solver"]["maximum_iterations"] = 30000
    #prm["krylov_solver"]["error_on_nonconvergence"] = True
    #prm["krylov_solver"]["absolute_tolerance"] = 1e-10
    #prm["krylov_solver"]["relative_tolerance"] = 1e-10
    #prm["krylov_solver"]["monitor_convergence"] = True
    print("Start linear solver")
    #solver1.solve()
    df.solve(f1==l1, d, bc)
    print("Linear solver finished")
    return d
    
def SolvePoissonInteriorOnly(MyMesh, inside_domains, outside_domains, interface_id, bounding_box, smooth):
    mesh = MyMesh.mesh
    dx = MyMesh.dx
    ds = MyMesh.ds
    #define function spaces
    CG1 = df.FunctionSpace(mesh, "CG", 1)
    v = df.TestFunction(CG1)
    u = df.TrialFunction(CG1)
    #make initial guess
    #set BCs to measure distance from inner interface
    bc = []
    for i in interface_id:
        bc.append(df.DirichletBC(CG1, 0.0, MyMesh.boundaries, i))
    #quick and dirty weak soltion for solving Laplace in the interior only
    f1 = df.Constant(smooth)*df.inner(df.grad(v), df.grad(u))*dx(inside_domains)
    f1 += v*u*dx(outside_domains)
    l1 = v*df.Constant(1.0)*dx(inside_domains)
    l1 += v*df.Constant(0.0)*dx(outside_domains)
    d = df.Function(CG1)
    d.rename("Laplace", "")
    df.solve(f1==l1, d, bc)
    print("Linear solver finished")
    return d
