import torch
import torch.nn as nn
import numpy as np

class TorchNeuralNet(nn.Module):
    def __init__(self, NumLayers, NumNeuronsPerLayer):
        #initialize parent
        super().__init__()
        self.NumLayer = NumLayers
        self.NeuronsPerLayer = NumNeuronsPerLayer
        UseBias = True
        self.Layers = []
        self.Layers.append(nn.Linear(2, self.NeuronsPerLayer, bias=True))
        self.Layers.append(nn.Tanh())
        for i in range(self.NumLayer):
            self.Layers.append(nn.Linear(self.NeuronsPerLayer, self.NeuronsPerLayer, bias=UseBias))
            self.Layers.append(nn.Tanh())
            #self.Layers.append(nn.ReLU())
            #self.Layers.append(nn.Sigmoid())
        self.Layers.append(nn.Linear(self.NeuronsPerLayer, 1, bias=UseBias))
        self.call = nn.Sequential(*self.Layers)
    
    #forward
    def forward(self, x):
        #identity = x.clone()
        return self.call(x)

class TorchPinn(nn.Module):
    def __init__(self):
        #initialize parent
        super().__init__()
        #mimics PDE solution
        self.u = None
        self.NumLayer = 0
        self.NumNeuronsPerLayer = 0
        self.parameters = None
        self.TrainHistory = []
        
        #for restarting training
        self.EpochsTrained = 0
        #how many training iterations to save
        self.SaveEvery = -1
        #training sample points
        self.x_domain = []
        self.x_boundary = []
        self.x_interpolation = []
        self.DATA_TYPE = torch.float64
        torch.set_default_dtype(self.DATA_TYPE)
        
    def save(self, path):
        data = {"layers":self.NumLayer, "NeuronsPerLayer":self.NumNeuronsPerLayer, "TrainHistory":self.TrainHistory, "EpochsTrained":self.EpochsTrained, "SaveEvery":self.SaveEvery, "x_domain":self.x_domain, "x_boundary":self.x_boundary, "x_interpolation":self.x_interpolation, "DataType":self.DATA_TYPE,"NeuralNet":self.u}
        torch.save(data, path)
        
    def load(self, path):
        loaded = torch.load(path, weights_only=False)
        self.NumLayer = loaded["layers"]
        self.NumNeuronsPerLayer = loaded["NeuronsPerLayer"]
        self.TrainHistory = loaded["TrainHistory"]
        self.x_domain = loaded["x_domain"]
        self.x_boundary = loaded["x_boundary"]
        self.x_interpolation = loaded["x_interpolation"]
        self.DATA_TYPE = loaded["DataType"]
        self.u = loaded["NeuralNet"]
        self.SaveEvery = loaded["SaveEvery"]
        
    def InitNeuralNet(self, NumLayer, NumNeuronsPerLayer):
        self.NumLayer = NumLayer
        self.NumNeuronsPerLayer = NumNeuronsPerLayer
        self.u = TorchNeuralNet(NumLayer, NumNeuronsPerLayer)
        self.parameters = [*self.u.parameters()]
        
    def obs_loss(self, x_data, obs):
        x_data.requires_grad = True
        loss = ((self.u(x_data) - obs)**2).sum()
        return loss
        
    def train(self, x_data = None):
        #optimizer to train the neural net:
        import torch.optim as optim
        #iterations for the optimizer to train the torch neural net
        maxiter = 1500
        #store convergence history
        # old fenics: optimizer = optim.LBFGS(self.u.parameters(), max_iter=maxiter, callback=opt_cb, line_search_fn="strong_wolfe", history_size=1000, tolerance_change=0)
        optimizer = optim.LBFGS(self.u.parameters(), max_iter=maxiter, line_search_fn="strong_wolfe", history_size=1000, tolerance_change=0)
        #optimizer = optim.Adam(self.u.parameters(), lr = 1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', patience=100, min_lr=1e-5)
        #self.TrainHistory = []
        #construct objective for machine learning
        #weights for each part of the loss function
        w_pde = 0.0
        w_bc = 0.0
        w_data = 0.0
        if self.x_domain != None:
            w_pde = 1.0/self.x_domain.tensors[0].shape[0]
        if self.x_boundary != None:
            w_bc = 4.0/self.x_boundary.tensors[0].shape[0]
        if x_data != None:
            w_data = 1.0/x_data.tensors[0].shape[0]
            
        def closure():
            if self.EpochsTrained == 0:
                fOut = open(f"{self.FolderName}/Train_History.txt", 'w')
            else:
                fOut = open(f"{self.FolderName}/Train_History.txt", 'a')
            opt_loss_history = []
            optimizer.zero_grad()
            loss = 0.0
            if w_pde > 0:
                L_pde = w_pde*self.pde_loss(self.x_domain.tensors[0], self.x_domain.tensors[1])
                loss += L_pde
                opt_loss_history.append(float(L_pde))
            if w_bc > 0:
                L_bc = w_bc*self.bc_loss(self.x_boundary.tensors[0], self.x_boundary.tensors[1])
                loss += L_bc
                opt_loss_history.append(float(L_bc))
            if w_data > 0:
                L_data = w_data*self.obs_loss(x_data.tensors[0], x_data.tensors[1])
                loss += L_data
                opt_loss_history.append(float(L_data))
            loss.backward()
            #self.TrainHistory.append([float(loss), *opt_loss_history])
            """
            for i in [float(loss), *opt_loss_history]:
                fOut.write("%le "%i)
            fOut.write("\n")
            """
            fOut.write("%le\n "%loss)
            fOut.flush()
            self.TrainHistory.append([float(floss), float(L_pde), float(L_bc), float(L_data)])
            if self.SaveEvery > 0:
                if epoch%self.SaveEvery == 0:
                    name = f"{self.FolderName}/TempLearn_{epoch}.pt"
                    self.save(name)
            print("Closure: Loss = %le"%loss)
            return loss
        #start training:
        try:
            epochs = 1#1500
            for epoch in range(self.EpochsTrained, self.EpochsTrained+epochs):
                Loss = closure
                optimizer.step(Loss)
                self.EpochsTrained += 1
                #scheduler.step(Loss())
                #print(scheduler._last_lr)
                
        except KeyboardInterrupt:
            pass
            
    def train_with_dataloader(self, dataloader_pde = None, dataloader_obs = None, dataloader_bc = None, N_EPOCHS = 100000, lr = 1e-3):
        import torch.optim as optim
        optimizer = optim.Adam(self.u.parameters(), lr)
        if self.EpochsTrained == 0:
            fOut = open(f"{self.FolderName}/Train_History.txt", 'w')
        else:
            fOut = open(f"{self.FolderName}/Train_History.txt", 'a')
        #from tqdm import tqdm
        for epoch in range(self.EpochsTrained, self.EpochsTrained+N_EPOCHS):
            # Loop over batches in an epoch using DataLoader
            optimizer.zero_grad()
            
            #compute equal weights independent of batch size
            total_batch = 0
            if dataloader_pde is not None:
                total_batch += dataloader_pde.batch_size
            if dataloader_obs is not None:
                total_batch += dataloader_obs.batch_size
            if dataloader_bc is not None:
                total_batch += dataloader_bc.batch_size
                
            loss = 0.0
            loss_pde = 0.0
            loss_bc = 0.0
            loss_obs = 0.0
            if dataloader_obs is not None:
                weight = dataloader_obs.batch_size
                for id_batch, (x_batch, y_batch) in enumerate(dataloader_obs):
                    loss_obs += weight*self.obs_loss(x_batch, y_batch)
                loss += loss_obs
                    #optimizer.zero_grad()
                    #loss.backward()
                #optimizer.step()
            if dataloader_pde is not None:
                weight = dataloader_pde.batch_size
                for id_batch_bc, (x_bc, y_bc) in enumerate(dataloader_pde):
                    loss_pde += weight*self.pde_loss(x_bc, y_bc)
                loss += loss_pde
                    #optimizer.zero_grad()
                    #loss.backward()
                #optimizer.step()
            if dataloader_bc is not None:
                weight = 4.0*dataloader_bc.batch_size
                for id_batch_bc, (x_bc, y_bc) in enumerate(dataloader_bc):
                    loss_bc += weight*self.bc_loss(x_bc, y_bc)
                loss += loss_bc
                    #optimizer.zero_grad()
                    #loss.backward()
                #optimizer.step()
            loss /= total_batch
            loss_pde /= total_batch
            loss_bc /= total_batch
            loss_obs /= total_batch
            #loss = torch.tensor(loss, requires_grad=True)
            
            fOut.write(f"{loss:.5e}, {loss_pde:.5e}, {loss_bc:.5e}, {loss_obs:.5e}\n")
            fOut.flush()
            
            loss.backward()
            optimizer.step()
            #print(self.TrainHistory)
            self.TrainHistory.append([float(loss), float(loss_pde), float(loss_bc), float(loss_obs)])

            if self.SaveEvery > 0:
                if epoch%self.SaveEvery == 0:
                    print(f"Epoch {epoch}: Total loss: {loss:.4e},\tPDE loss: {loss_pde:.4e},\tBC loss: {loss_bc:.4e},\tData loss: {loss_obs:.4e}")
                    name = f"{self.FolderName}/TempLearn_{epoch}.pt"
                    #torch.save(self.state_dict(), name)
                    self.save(name)
        self.EpochsTrained += N_EPOCHS
                
    def ConvertToFEM(self, MyMesh):
        import dolfin as df
        
        DATA_TYPE = torch.float64
        torch.set_default_dtype(DATA_TYPE)

        #convert neural net into function and plot
        #print model values at some point to test
        PlotSpace = df.FunctionSpace(MyMesh.mesh, "CG", 1)
        d_plot = df.Function(PlotSpace)
        d_plot.rename("NeuralNetSolution", "")
        d2v = df.dof_to_vertex_map(d_plot.function_space())
        for i in range(len(d_plot.vector())):
            v = df.Vertex(MyMesh.mesh, d2v[i])
            GeoDim = MyMesh.mesh.geometry().dim()
            Coordinate = []
            for j in range(GeoDim):
                Coordinate.append(v.x(j))
            tx = torch.tensor(Coordinate, dtype=DATA_TYPE)
            u = float(self.u(tx))
            d_plot.vector()[i] = u
        d_plot.vector().apply("")
        return d_plot
        
    def GetInterpolationDataFromFEM(self, data_function):
        """
        obs_point = []
        obs_data = []
        from dolfin import dof_to_vertex_map, Vertex
        #from IPython import embed; embed()
        mesh = data_function.function_space().mesh()
        d2v = dof_to_vertex_map(data_function.function_space())
        for i in range(len(data_function.vector())):
            v = Vertex(mesh, d2v[i])
            GeoDim = mesh.geometry().dim()
            Coordinate = []
            for j in range(GeoDim):
                Coordinate.append(v.x(j))
            #boundary belongs to data loss
            obs_point.append(Coordinate)
            #data for Laplace
            obs_data.append(data_function.vector()[i])
        """
        Space = data_function.function_space()
        obs_data = data_function.vector().get_local()
        obs_point = Space.tabulate_dof_coordinates()
        return torch.tensor(obs_point), torch.tensor(obs_data).unsqueeze(1)
        
    def SampleCircleDomain(self, n_points_interior):
        #sample only inside sphere
        def InsideOutside(x):
            shift = np.array([0.5, 0.5])
            radius = 0.25
            return np.linalg.norm(x-shift, axis=1, keepdims=True) < radius
        #circle only
        self.x_domain, x_domain_shape = None, 0
        while x_domain_shape < n_points_interior:
            samples = np.random.uniform([0, 0], [1, 1], (n_points_interior * 10, 2))
            samples_in_domain = samples[np.where(InsideOutside(samples))[0]]
            if self.x_domain is None:
                self.x_domain = samples_in_domain
                x_domain_shape = self.x_domain.shape[0]
            else:
                self.x_domain = np.concatenate([self.x_domain, samples_in_domain], axis=0)
                x_domain_shape = self.x_domain.shape[0]
        self.x_domain = self.x_domain[:n_points_interior]
        self.x_domain = torch.tensor(self.x_domain, dtype=self.DATA_TYPE)
        
    def SampleSquareDomain(self, n_points_interior):
        self.x_domain = np.random.uniform([0, 0], [1,1], [n_points_interior, 2])
        self.x_domain = torch.tensor(self.x_domain, dtype=self.DATA_TYPE)
        
    def SampleCircleBoundary(self, n_points_boundary):
        w_bc = np.random.uniform(0.0, 2.0*np.pi, n_points_boundary)
        a = 0.25
        b = 0.25
        self.x_boundary = np.column_stack((a*np.sin(w_bc)+0.5, b*np.cos(w_bc)+0.5))
        self.x_boundary = torch.tensor(self.x_boundary, dtype=self.DATA_TYPE)
        
    def SampleFEniCSBoundary(self, MyBoundaryMesh, n_points_boundary):
        TopDim = MyBoundaryMesh.topology().dim()
        GeoDim = MyBoundaryMesh.geometry().dim()
        if TopDim != 1:
            print("WARNING: ONLY INTENTED FOR 1D MESHES")
        NumCells = MyBoundaryMesh.num_entities(TopDim)
        SamplesPerCell = n_points_boundary//NumCells
        boundary_samples = np.zeros([SamplesPerCell*NumCells, 2])
        from dolfin import cells, vertices
        sample_counter = 0
        #TODO: This is not globally uniform, but only uniform per edge (not identical)
        #Fine for training point cloud, but not for Stokes-Integral
        for c in cells(MyBoundaryMesh):
            MyCoords = np.zeros([TopDim+1, GeoDim])
            counter = 0
            for v in vertices(c):
                for j in range(GeoDim):
                    MyCoords[counter, j] = v.x(j)
                counter += 1
            e = MyCoords[1] - MyCoords[0]
            for i in range(SamplesPerCell):
                p = (MyCoords[0] + np.random.uniform(0,1)*e)
                boundary_samples[sample_counter, :] = p
                sample_counter += 1
        np.random.shuffle(boundary_samples)
        self.x_boundary = boundary_samples
        self.x_boundary = torch.tensor(self.x_boundary, dtype=self.DATA_TYPE)
        
class TorchInterpolate(TorchPinn):
    def __init__(self):
        TorchPinn.__init__(self)
        UserExpression.__init__(self)
        self.FolderName = "results_Interpolate"

from dolfin import UserExpression

import torch.autograd as autograd
def spatial_grad(u, x, retain_graph=True, create_graph=True, device=None):
    adj_inp = torch.ones(u.shape, device=device)
    r, *_ = autograd.grad(u, x, adj_inp, retain_graph=retain_graph, create_graph=create_graph)
    return r

def spatial_div(u, x, retain_graph=True, create_graph=True, device=None):
    """
        u: Shape [BATCH_SIZE, d, ...]
    """
    shape = u.shape
    dim = shape[1]
    r = 0.
    for i in range(dim):
        adj_inp = torch.zeros(shape, device=device)
        adj_inp[:, i, ...] = 1
        r_, *_ = autograd.grad(u, x, adj_inp, retain_graph=retain_graph, create_graph=create_graph)
        r += r_[:, i].unsqueeze(1)
    return r

def spatial_laplace(u, x):
    return spatial_div(spatial_grad(u, x), x)

class TorchPoisson(TorchPinn, UserExpression):
    def __init__(self):
        TorchPinn.__init__(self)
        UserExpression.__init__(self)
        self.FolderName = "results_Laplace"
        self.smooth=1.0
        
    def pde_loss(self, x, rhs):
        x.requires_grad = True
        #PDE
        loss = ((-self.smooth*spatial_laplace(self.u(x), x) - rhs)**2).sum()
        return loss
        
    def bc_loss(self, x, rhs):
        x.requires_grad = True
        return ((self.u(x) - rhs)**2).sum()
        
    #to do: Move to paranet class?!
    def eval(self, values, x):
        #calculate y,z depending on x
        input = torch.tensor(x)
        out = self.u.call(input)
        values[0] = float(out)
        
    def forward(self, x):
        input = torch.tensor(x)
        out = self.u.call(input)
        return out
        
    def value_shape(self):
        return ()

class TorchPoissonSubdomain(nn.Module):
    def __init__(self, NeuralNet, InsideIndicator):
        #initialize parent
        super().__init__()
        #mimics PDE solution
        self.u = NeuralNet
        self.InsideIndicator = InsideIndicator
        #for regulaization / hybrid NNs
        #self.NN = PDETerm()
        
    def forward_inside(self, x):
        x.requires_grad = True
        #PDE
        loss = ((-spatial_laplace(self.u(x), x) + 1.0)**2).mean()
        return loss
        
    def forward_outside(self, x):
        x.requires_grad = True
        #PDE
        loss = ((-spatial_laplace(self.u(x), x) - 1.0)**2).mean()
        return loss
        
    def bc_loss(self, x):
        return ((self.u(x))**2).mean()
        
    def train(self, x_subdomain_inside, x_subdomain_outside):
        #optimizer to train the neural net:
        import torch.optim as optim
        #iterations for the optimizer to train the torch neural net
        maxiter = 250
        #optimizer = optim.LBFGS(self.parameters(), max_iter=maxiter, callback=opt_cb, line_search_fn="strong_wolfe", history_size=1000, tolerance_change=0)
        optimizer = optim.LBFGS(self.parameters(), max_iter=maxiter, line_search_fn="strong_wolfe", history_size=1000, tolerance_change=0)
        #construct objective for machine learning
        torch_optim = []
        #opt_loss_history = []
        #weights for each part of the loss function
        w_pde_inside = 1e0
        w_pde_outside = 1e0
        w_bc = 1e0
        #w_ic = 1.0
        #w_data = 1.0
        def closure():
            global torch_optim
            optimizer.zero_grad()
            L_pde_inside = self.forward_inside(x_subdomain_inside)
            L_pde_outside = self.forward_outside(x_subdomain_outside)
            L_bc = self.bc_loss(self.x_boundary)
            #regularization by NN from Sebastian's example
            #L_reg = 1e-8 * (self.NN(self.x_domain) ** 2).mean()
            #data from FEM-solution
            #L_data = self.obs_loss(x_data, t_data, obs).mean()
            #initial condition
            #L_ic = self.obs_loss(x_ic, t_ic, obs_ic).mean()

            loss = w_bc*L_bc + w_pde_inside*L_pde_inside + w_pde_outside*L_pde_outside
            loss.backward()
            print("Closure: Loss = %le"%float(loss))
            #torch_optim = [float(L_bc), float(L_pde_inside), float(L_pde_outside)]#, float(L_reg)]#, float(L_data), float(L_ic)]
            return loss
        #start training:
        try:
            epochs = 1
            for epoch in range(epochs):
                print(f"Epoch = {epoch}")
                optimizer.step(closure)
        except KeyboardInterrupt:
            pass

class TorchEikonal(nn.Module):
    def __init__(self, NeuralNet):
        #initialize parent
        super().__init__()
        #mimics PDE solution
        self.u = NeuralNet
        #for regulaization / hybrid NNs
        #self.NN = PDETerm()
        
    def forward_inside(self, x):
        x.requires_grad = True
        #PDE
        from torch import inner as tinner
        g = spatial_grad(self.u(x), x)
        loss_eikonal = tinner(g,g) - 1.0
        loss_smooth = -0.015625*spatial_laplace(self.u(x), x)+1
        return ((loss_smooth + 0.0*loss_eikonal)**2).mean()
        
    def forward_outside(self, x):
        x.requires_grad = True
        #PDE
        from torch import inner as tinner
        g = spatial_grad(self.u(x), x)
        loss_eikonal = tinner(g,g) - 1.0
        loss_smooth = -0.015625*spatial_laplace(self.u(x), x)+1
        return ((loss_smooth + 0.0*loss_eikonal)**2).mean()
        
    def bc_loss(self, x):
        return ((self.u(x))**2).mean()
        
    def obs_loss(self, x_data, obs):
        return ((self.u(x_data) - obs)**2).mean()
        
    def train(self, x_subdomain_inside, x_subdomain_outside, x_data, obs):
        #optimizer to train the neural net:
        import torch.optim as optim
        #iterations for the optimizer to train the torch neural net
        maxiter = 1000
        #optimizer = optim.LBFGS(self.parameters(), max_iter=maxiter, callback=opt_cb, line_search_fn="strong_wolfe", history_size=1000, tolerance_change=0)
        #optimizer = optim.LBFGS(self.parameters(), max_iter=maxiter, line_search_fn="strong_wolfe", history_size=1000, tolerance_change=0)
        optimizer = optim.Adam(self.parameters(), lr = 1e-2)
        #construct objective for machine learning
        #torch_optim = []
        #opt_loss_history = []
        #weights for each part of the loss function
        w_pde_inside = 1e0
        w_pde_outside = 1e0
        w_data = 0e1
        w_bc = 1e1
        #w_ic = 1.0
        #w_data = 1.0
        #counter = 0
        def closure():
            global torch_optim
            optimizer.zero_grad()
            L_pde_inside = self.forward_inside(x_subdomain_inside)
            L_pde_outside = self.forward_outside(x_subdomain_outside)
            L_bc = self.bc_loss(self.x_boundary)
            L_data = self.obs_loss(x_data, obs)
            #regularization by NN from Sebastian's example
            #L_reg = 1e-8 * (self.NN(self.x_domain) ** 2)
            #data from FEM-solution

            loss = w_bc*L_bc + w_pde_inside*L_pde_inside + w_pde_outside*L_pde_outside + w_data*L_data# + L_reg # + w_ic*L_ic
            loss.backward()
            print("Closure %d: Loss = %le"%(0.0, float(loss)))
            #torch_optim = [float(L_bc), float(L_pde), ]#, float(L_reg)]#, float(L_data), float(L_ic)]
            #counter = counter + 1
            return loss
        #start training:
        try:
            epochs = 1
            for epoch in range(epochs):
                print(f"Epoch = {epoch}")
                optimizer.step(closure)
        except KeyboardInterrupt:
            pass
