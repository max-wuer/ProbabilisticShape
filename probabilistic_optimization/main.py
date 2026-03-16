import dolfin as df
import numpy as np
#for making folders
import os

def SaveMeshH5(mesh, filename , closeFile=True):
    orientation = np.asarray(mesh.cell_orientations(), dtype = 'float')
    if os.path.splitext(filename)[1] == ".h5":
        s=filename
    else:
        s = filename + '.h5'
    hdf = df.HDF5File(df.MPI.comm_world, s,'w')
    hdf.write(mesh,'mesh')
    vectorDG = df.VectorFunctionSpace(mesh, "DG", 0)
    DGnormal = df.project(df.CellNormal(mesh), vectorDG)
    hdf.write(DGnormal,'normal')
    if closeFile:
        hdf.close()
        return
    else:
        return hdf
        
def ReadMeshH5(filename, closeFile=True):
    if os.path.splitext(filename)[1] == ".h5":
        s=filename
    else:
        s = filename + '.h5'
    hdf = df.HDF5File(df.MPI.comm_world, s,'r')
    mesh = df.Mesh()
    hdf.read(mesh, 'mesh', False)
    mesh.init()
    if hdf.has_dataset('normal'):
        vectorDG = df.VectorFunctionSpace(mesh, "DG", 0)
        DGnormal = df.Function(vectorDG)
        hdf.read(DGnormal,'normal')
        #Orientate(mesh, DGnormal)
    else:
        print("mesh could not be orientated, no dataset \"normal\" included in H5")

    if closeFile:
        hdf.close()
        return mesh
    else:
        return mesh, hdf

#for computing the exit expectation part of the rhs of the shape derivative
# Evaluate the i-th basis function at point x:
def basis_func(i, x_list, cell_indices, grad_u_list, Space):
    out = 0.0
    for x, cell_index, grad_u in zip(x_list, cell_indices, grad_u_list):
        cell_global_dofs = Space.dofmap().cell_dofs(cell_index)
        for local_dof in range(0,len(cell_global_dofs)):
            if(i==cell_global_dofs[local_dof]):
                cell = df.Cell(Space.mesh(), cell_index)
                n = cell.cell_normal()
                MyNx = n[0]
                MyNy = n[1]
                grad_u_n = grad_u[0]*MyNx + grad_u[1]*MyNy #!?
                out += grad_u_n*Space.element().evaluate_basis(local_dof,x, cell.get_vertex_coordinates(), cell.orientation())[0] #vector if space is VectorFunctionSpace
        # If none of this cell's shape functions map to the i-th basis function,
        # then the i-th basis function is zero at x.
    return out

from MyMesh import MyMesh, ExtractInterface
from FenicsPDEs import SolvePoisson, SolvePoissonInteriorOnly
from monte_carlo_shapederivative import monte_carlo_shapederivative

#MeshNameState = "circle1_2D_unit"
MeshNameState = "circle2_2D_unit"
#MeshNameState = "circle_unit_2_only"
MyMeshState = MyMesh()
MyMeshState.open(MeshNameState)

inside_domains = (1,)
outside_domains = (2,)
interface_id = [1]
interface_id = [1]
interface_id = [1]
interface_id = [1]
interface_id = [1]
bounding_box = [2]

Folder = "results_Optimization"

#compute tracking type data
MeshNameData = "ellipsoid2D_unit_3_tracking"
#MeshNameData = "ellipsoid2D_unit_3_tracking_narrow"
MyMeshData = MyMesh()
MyMeshData.open(MeshNameData)
inside_domains = (1,)
outside_domains = (2,)
interface_id = [1]
bounding_box = [2]

#solve the Laplacian in the interior and extend with zero into the hold-all
fem_tracking = SolvePoissonInteriorOnly(MyMeshData, inside_domains, outside_domains, interface_id, bounding_box, 1.0)
#fem_tracking = SolvePoisson(MyMeshData, inside_domains, outside_domains, interface_id, bounding_box, 1.0)
fem_tracking.rename("u_d", "")
df.File(Folder+"/Tracking_orig.pvd") << fem_tracking
#fem_tracking = df.project(fem_tracking, u_fem_state.function_space())
#fem_tracking.rename("u_d", "")
#df.File(Folder+"/Tracking_projected.pvd") << fem_tracking

#Laplace state from PINN
from TorchModels import *
from torch.utils.data import DataLoader, TensorDataset

#seeds for debugging:
#np.random.seed(42)
#torch.manual_seed(42)

DATA_TYPE = torch.float64
u_pinn_state = TorchPoisson()
u_pinn_state.double()

Restart = 0
if Restart <= 0:
    u_pinn_state.load("LaplacePinnUnitCircle/model_2x20_poly_inside_2.pt")
    (MyBoundaryMesh, _) = ExtractInterface(MyMeshState.mesh, MyMeshState.subdomains, inside_domains[0], outside_domains[0])
    fOptHistory = open(Folder+"/OptHistory.txt", "w")
    iOpt = 0
else:
    iOpt = Restart
    MyBoundaryMesh = ReadMeshH5(Folder+f"/{iOpt}/mesh.h5")
    u_pinn_state.load(Folder+f"/{iOpt-1}/PINN_final.pt")
    fOptHistory = open(Folder+"/OptHistory.txt", "a")

#create the monte_carlo optimizer object and set its objective evaluation to the PINN
mcsd = monte_carlo_shapederivative()

Step = df.Constant(1e-1)
u_pinn_state.SaveEvery = -1
u_pinn_state.FolderName = ""

#Change this if you want to solve the state equation with FEniCS
#or a Torch PINN
#use fem state
#MyState = u_fem_state
#use pinn state
MyState = u_pinn_state
nn_indicator = MyState

MyBoundaryMesh.init_cell_orientations(df.Expression(("x[0]-0.5", "x[1]-0.5"), degree=1))

#test orientation
x = df.SpatialCoordinate(MyBoundaryMesh)
orientationtest = df.assemble(df.inner(x, df.CellNormal(MyBoundaryMesh))*df.dx)/2
print(f"Test Orientation: {orientationtest}")

#begin optimization loop
UseLBFGS = False

if UseLBFGS:
    from LBFGS import MyLBFGS
    MyOptimizer = MyLBFGS()
    #LBFGSFile = df.File(Folder+"/LBFGSGradient.pvd")
    LBFGSFile = df.XDMFFile(df.MPI.comm_world, Folder+"/LBFGSGradient.xdmf")
    LBFGSFile.parameters["flush_output"] = True
    LBFGSFile.parameters["rewrite_function_mesh"] = True

#ShapeGradFile = df.File(Folder+"/ShapeGradient.pvd")
ShapeGradFile = df.XDMFFile(df.MPI.comm_world, Folder+"/ShapeGradient.xdmf")
ShapeGradFile.parameters["flush_output"] = True
ShapeGradFile.parameters["rewrite_function_mesh"] = True

#Some files with debugging info
#NormalFile = df.File(Folder+"/Normal.pvd")
NormalFile = df.XDMFFile(df.MPI.comm_world, Folder+"/Normal.xdmf")
NormalFile.parameters["flush_output"] = True
NormalFile.parameters["rewrite_function_mesh"] = True
#DirDerivFile = df.File(Folder+"/DirDeriv.pvd")
DirDerivFile = df.XDMFFile(df.MPI.comm_world, Folder+"/DirDeriv.xdmf")
DirDerivFile.parameters["flush_output"] = True
DirDerivFile.parameters["rewrite_function_mesh"] = True
#ExitExpectationFile = df.File(Folder+"/ExitExpectation.pvd")
ExitExpectationFile = df.XDMFFile(df.MPI.comm_world, Folder+"/ExitExpectation.xdmf")
ExitExpectationFile.parameters["flush_output"] = True
ExitExpectationFile.parameters["rewrite_function_mesh"] = True
#StokesFile = df.File(Folder+"/StokesTerm.pvd")
StokesFile = df.XDMFFile(df.MPI.comm_world, Folder+"/StokesTerm.xdmf")
StokesFile.parameters["flush_output"] = True
StokesFile.parameters["rewrite_function_mesh"] = True
#end debugging files

GradNorm = 1e9
while(GradNorm > 1e-10):
    iOpt = iOpt + 1
    print(f"Optimization loop {iOpt}")
    #make new folder for results
    if not os.path.exists(Folder+f"/{iOpt}"):
        os.makedirs(Folder+f"/{iOpt}")
        
    MyState.FolderName = Folder+f"/{iOpt}"

    print(f"Monte Carlo Measure")
    #mcsd.monte_carlo_measure_domain_partition(MyState, fem_tracking, nn_indicator, Folder+f"/{iOpt}")
    mcsd.monte_carlo_measure_domain_partition(MyState, fem_tracking, nn_indicator)
    
    #compute objective based on rejection sampling integrals
    Obj = mcsd.monte_carlo_shape_functional(MyState, fem_tracking, nn_indicator)
    print(f"The objective is {Obj}")

    #compute m^+ and m^- based on rejection sampling
    print(f"Monte Carlo Constants")
    mcsd.monte_carlo_constants(MyState, fem_tracking, nn_indicator)
    print(f"Exit Points")
    mcsd.ExitPointBatch(1000, MyState, fem_tracking, nn_indicator, MyBoundaryMesh)
    print(f"Start Gradient")
    #Compute Shape Derivative
    #dolfin test function for deformation/smoothing
    Space = df.FunctionSpace(MyBoundaryMesh, "CG", 1)
    Vn = df.TestFunction(Space)
    
    #prepare shape gradient and do smoothing
    MyRHS = df.assemble(df.inner(df.TestFunction(Space),df.Constant((0.0)))*df.dx)
    Data = MyRHS.get_local()
    #integrate with custom point measures
    if mcsd.ConstantPlus > 0:
        TorchExitPlus = torch.tensor(mcsd.ExitPoints["+"], dtype=DATA_TYPE)
        TorchExitPlus.requires_grad = True
        state_grad_plus = spatial_grad(u_pinn_state.u(TorchExitPlus), TorchExitPlus)
        state_grad_plus = state_grad_plus.detach().numpy()

        for i in range(len(MyRHS.get_local())):
            Data[i] += mcsd.ConstantPlus*basis_func(i, mcsd.ExitPoints["+"], mcsd.ExitPoints["CellIndicesPlus"], state_grad_plus, Space)/mcsd.batch_size
    if mcsd.ConstantMinus > 0:
        TorchExitMinus = torch.tensor(mcsd.ExitPoints["-"], dtype=DATA_TYPE)
        TorchExitMinus.requires_grad = True
        state_grad_minus = spatial_grad(u_pinn_state.u(TorchExitMinus), TorchExitMinus)
        state_grad_minus = state_grad_minus.detach().numpy()

        for i in range(len(MyRHS.get_local())):
            Data[i] -= mcsd.ConstantMinus*basis_func(i, mcsd.ExitPoints["-"], mcsd.ExitPoints["CellIndicesMinus"], state_grad_minus, Space)/mcsd.batch_size
    MyRHS.set_local(Data)
    MyRHS.apply("")
    
    fem_tracking_projected = df.interpolate(fem_tracking, Space)
    StokesComponent = df.assemble(Vn*fem_tracking_projected**2*df.dx(domain=MyBoundaryMesh))

    #h1 gradient representation
    Wn = df.TrialFunction(Space)

    a = df.Constant(0.5)*df.inner(df.grad(Wn), df.grad(Vn))*df.dx + df.inner(Vn, Wn)*df.dx
    A = df.assemble(a)
    ShapeGrad = df.Function(Space)
    ShapeGrad.rename("ShapeGradient", "")
    df.solve(A, ShapeGrad.vector(), -(MyRHS-StokesComponent))
        
    #plot RHS before smoothing for debugging
    PltRhs = df.Function(Space)
    PltRhs.rename("DirDeriv", "")
    PltRhs.vector().set_local(-(MyRHS-StokesComponent))
    PltRhs.vector().apply("")
    #DirDerivFile << PltRhs
    DirDerivFile.write(PltRhs, float(iOpt))
    #plot stokes
    PltStokes = df.Function(Space)
    PltStokes.rename("StokesTerm", "")
    PltStokes.vector().set_local(StokesComponent)
    PltStokes.vector().apply("")
    #StokesFile << PltStokes
    StokesFile.write(PltStokes, float(iOpt))
    #plot ExitExpectation
    PltEE = df.Function(Space)
    PltEE.rename("ExitExpectationTerm", "")
    PltEE.vector().set_local(-MyRHS)
    PltEE.vector().apply("")
    #ExitExpectationFile << PltEE
    ExitExpectationFile.write(PltEE, float(iOpt))
    
    PlotNormal = df.project(df.CellNormal(MyBoundaryMesh), df.VectorFunctionSpace(MyBoundaryMesh, "DG", 0, 2))
    #NormalFile << PlotNormal
    NormalFile.write(PlotNormal, float(iOpt))

    W = df.project(Step*ShapeGrad*df.CellNormal(MyBoundaryMesh), df.VectorFunctionSpace(MyBoundaryMesh, "CG", 1))
    W.rename("W", "")
    
    ShapeGradFile.write(W, float(iOpt))
    GradNorm = float(df.norm(W, "L2")/Step)
    
    if UseLBFGS:
        MyOptimizer.Store(W)
        W = MyOptimizer.ApplyStep(W)
        W.rename("W", "")

        LBFGSFile.write(W, float(iOpt))
        LBFGSNorm = df.norm(W, "L2")/Step
        
        fOptHistory.write("%d %le %le %le\n"%(iOpt, Obj, GradNorm, LBFGSNorm))
    else:
        fOptHistory.write("%d %le %le\n"%(iOpt, Obj, GradNorm))

    fOptHistory.flush()
    
    df.ALE.move(MyBoundaryMesh, W)
    SaveMeshH5(MyBoundaryMesh, Folder+f"/{iOpt}/mesh.h5")
    
    #retrain neural net!
    n_points_interior = 45000
    n_points_boundary = n_points_interior//2
    
    #model.SampleCircleDomain(n_points_interior)
    #n_points_interior = 5*45000
    MyState.SampleSquareDomain(n_points_interior)
    MyState.SampleFEniCSBoundary(MyBoundaryMesh, n_points_boundary)
    MaxBatch = 400000
    RelativeBatch = 1#1.0/60

    batch_size_pde = int(np.min([RelativeBatch*MyState.x_domain.shape[0], MaxBatch]))
    print(f"Training Batch Size PDE {batch_size_pde} out of {MyState.x_domain.shape[0]}")
    dataset_pde = TensorDataset(MyState.x_domain, torch.ones(n_points_interior, 1))
    dataloader_pde = DataLoader(dataset_pde, batch_size = batch_size_pde, shuffle=True)
    #boundaries
    batch_size_boundary = int(np.min([RelativeBatch*MyState.x_boundary.shape[0], MaxBatch]))
    print(f"Training Batch Size Boundary {batch_size_boundary} out of {MyState.x_boundary.shape[0]}")
    dataset_bc = TensorDataset(MyState.x_boundary, torch.zeros(MyState.x_boundary.shape[0], 1))
    dataloader_bc = DataLoader(dataset_bc, batch_size=batch_size_boundary, shuffle=True)
    
    MyState.train_with_dataloader(dataloader_pde = dataloader_pde, dataloader_bc = dataloader_bc, N_EPOCHS = 1000, lr=1e-4)
    
    MyState.train_with_dataloader(dataloader_pde = dataloader_pde, dataloader_bc = dataloader_bc, N_EPOCHS = 1200, lr=1e-5)
    MyState.save(Folder+f"/{iOpt}/PINN_final.pt")
    print("PINN created")

print("DONE!")
