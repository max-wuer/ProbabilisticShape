import numpy as np
from TorchModels import *
from MyMesh import ComputeMeshDistance, ProjectToBoundary

#already vectorized
def IsInside(x, nn_indicator):
    #shift = np.array([0.5, 0.5])
    #radius = 0.25
    #return np.linalg.norm(x-shift, axis=1, keepdims=True) < radius
    return nn_indicator(x).detach().numpy() > 0

class monte_carlo_shapederivative():
    def __init__(self):
        self.time_step_size = 1.0/50.0
        #FIXME: Number of sample for random start, 100 too low
        self.batch_size = 100
        self.process_dimension = 2
        self.HoldAll_x = (0.0, 0.0)
        self.HoldAll_y = (1.0, 1.0)
        self.HoldAllVolume = 1.0
        self.ExitPoints = {"+": [], "-": [], "CellIndicesPuls":[], "CellIndicesMinus":[]}
        #self.ExitPointsPlusProjected = []
        #self.ExitPointsMinusProjected = []
        self.MeasureOmegaPlus = None
        self.MeasureOmegaMinus = None
        self.ConstantPlus = None
        self.ConstantMinus = None
        self.density_max_val = 0.0
        self.density_min_val = 0.0
        
    def monte_carlo_measure_domain_partition(self, state, u_tracking, nn_indicator, PlotFolder = ""):
        sample_number = int(1e7)

        measure_samples = np.random.uniform(self.HoldAll_x, self.HoldAll_y, (sample_number, self.process_dimension))

        samples_in_domain = measure_samples[np.where(IsInside(measure_samples, nn_indicator))[0]]
        #test if state solver is given by Torch PINN or something else
        #eval objective accordingly
        if isinstance(state, TorchPoisson):
            tracking_eval = np.expand_dims(np.array(list(map(u_tracking, samples_in_domain))), axis=1)
            density_evaluation = state(samples_in_domain).detach().numpy() - tracking_eval
        else:
            density_evaluation = np.array(list(map(state, samples_in_domain))) - np.array(list(map(u_tracking, samples_in_domain)))

        number_samples_in_domain = samples_in_domain.shape[0]
        number_samples_omega_plus = np.where(density_evaluation > 0)[0].shape[0]
        number_samples_omega_minus = number_samples_in_domain - number_samples_omega_plus

        if PlotFolder != "":
            import matplotlib.pyplot as plt
            omega_plus = samples_in_domain[np.where(density_evaluation > 0)[0]]
            omega_minus = samples_in_domain[np.where(density_evaluation < 0)[0]]

            # todo colors of this!!
            plt.scatter(omega_plus[:, 0], omega_plus[:, 1], s=0.01, label=r'$\Omega^+$', color='blue')
            plt.scatter(omega_minus[:, 0], omega_minus[:, 1], s=0.01, label=r'$\Omega^-$', color='red')
            lgnd = plt.legend()
            lgnd.legendHandles[0]._sizes = [30]
            lgnd.legendHandles[1]._sizes = [30]
            plt.axis('square')
            #plt.show()
            plt.savefig(PlotFolder+"/domain_partition_laplace")

        self.MeasureOmegaPlus = number_samples_omega_plus/sample_number * self.HoldAllVolume
        self.MeasureOmegaMinus = number_samples_omega_minus/sample_number * self.HoldAllVolume

        self.density_max_val = np.max([self.density_max_val, np.max(density_evaluation)])
        self.density_min_val = np.max([self.density_min_val, np.abs(np.min(density_evaluation))])

        print(f"Monte Carlo approximation - measure omega plus = {self.MeasureOmegaPlus} and measure omega minus = {self.MeasureOmegaMinus} (sum of both should be area. Here: {self.MeasureOmegaPlus + self.MeasureOmegaMinus})")
        pass

    #compute m^+ and m^- based on rejection sampling
    def monte_carlo_constants(self, state, u_tracking, nn_indicator):
        max_iter = int(1e6) #should be 1e6
        constant_plus, constant_minus = 0.0, 0.0
        sample_counter, sample_counter_plus, sample_counter_minus = 0, 0, 0
        break_counter = 0
        break_counter_stop = int(5e2)
        #one loop is 2-3 seconds
        while sample_counter < max_iter and break_counter < break_counter_stop:
            samples = np.random.uniform(self.HoldAll_x, self.HoldAll_y, (max_iter, self.process_dimension))
            samples_in_domain = samples[np.where(IsInside(samples, nn_indicator))[0]]
            tracking_eval = np.expand_dims(np.array(list(map(u_tracking, samples_in_domain))), axis=1)
            if isinstance(state, TorchPoisson):
                density_evaluation = state(samples_in_domain).detach().numpy() - tracking_eval
            else:
                density_evaluation = np.array(list(map(state, samples_in_domain))) - np.array(list(map(u_tracking, samples_in_domain)))

            density_eval_plus = density_evaluation[np.where(density_evaluation > 0)[0]]
            density_eval_minus = density_evaluation[np.where(density_evaluation < 0)[0]]

            # rejection test plus
            uniform_plus = np.random.uniform(0, 1, density_eval_plus.shape[0])
            acceptance_rejection_indices_plus = np.where(uniform_plus <= self.MeasureOmegaPlus / self.HoldAllVolume)[0]
            
            constant_plus += np.sum(density_eval_plus[acceptance_rejection_indices_plus])
            sample_counter_plus += acceptance_rejection_indices_plus.shape[0]

            uniform_minus = np.random.uniform(0, 1, density_eval_minus.shape[0])
            acceptance_rejection_indices_minus = np.where(uniform_minus <= self.MeasureOmegaMinus / self.HoldAllVolume)[0]
            constant_minus += np.sum(density_eval_minus[acceptance_rejection_indices_minus])
            sample_counter_minus += acceptance_rejection_indices_minus.shape[0]

            if self.MeasureOmegaPlus*self.MeasureOmegaMinus > 0:
                sample_counter = np.min([sample_counter_plus, sample_counter_minus])
            else:
                sample_counter = np.max([sample_counter_plus, sample_counter_minus])

            break_counter += 1
            if break_counter > break_counter_stop//2:
                #TODO: sanity check for small omega^+- then all acc_rej samples will miss the 'inside omega^+-' // special problem if the domain shrinks
                print(f"Warning: Monte_Carlo_constants: Break Counter becomes large: {break_counter}")
                print(f"Sample_counter_plus: {sample_counter_plus}, Sample_counter_minus: {sample_counter_minus}")

        if sample_counter_plus == 0:
            self.ConstantPlus = 0.0
        else:
            self.ConstantPlus = self.MeasureOmegaPlus * constant_plus / sample_counter_plus
        if sample_counter_minus == 0:
            self.ConstantMinus = 0.0
        else:
            self.ConstantMinus = -1 * self.MeasureOmegaMinus * constant_minus / sample_counter_minus
        print(f"Monte Carlo sampling for constants completed. - constant plus = {self.ConstantPlus} with {sample_counter_plus} samples and constant minus = {self.ConstantMinus} with {sample_counter_minus} samples.")
    
    #TODO: compute measure of domain directly
    #TODO: Can reuse union of omega_+ and omega_- samples from function: monte_carlo_measure_domain_partition
    #computes the objective function based on rejection sampling
    def monte_carlo_shape_functional(self, state, u_tracking, nn_indicator):
        max_iter = int(1e6) #should be 1e6
        
        integral_value = 0.0
        domain_measure = self.MeasureOmegaPlus + self.MeasureOmegaMinus
        sample_counter = 0
        
        while sample_counter < max_iter:
            samples = np.random.uniform(self.HoldAll_x, self.HoldAll_y, (max_iter, self.process_dimension))
            samples_in_domain = samples[np.where(IsInside(samples, nn_indicator))[0]]
            
            tracking_eval = np.expand_dims(np.array(list(map(u_tracking, samples_in_domain))), axis=1)
            if isinstance(state, TorchPoisson):
                density_evaluation = state(samples_in_domain).detach().numpy() - tracking_eval
            else:
                density_evaluation = np.array(list(map(state, samples_in_domain))) - np.array(list(map(u_tracking, samples_in_domain)))

            #density_eval_plus = density_evaluation[np.where(density_evaluation > 0)[0]]
            #density_eval_minus = density_evaluation[np.where(density_evaluation < 0)[0]]

            # rejection test plus
            uniform_plus = np.random.uniform(0, 1, density_evaluation.shape[0])
            acceptance_rejection_indices = np.where(uniform_plus <= domain_measure / self.HoldAllVolume)[0]
            
            integral_value += np.sum(density_evaluation[acceptance_rejection_indices]**2)/2
            #integral_value += acceptance_rejection_indices.shape[0]
            sample_counter += acceptance_rejection_indices.shape[0]

        integral_value = integral_value / sample_counter * domain_measure
        return integral_value

    def sample_start(self, state, u_tracking, nn_indicator, plus_minus: str):
        safety_factor = 1.1
        if plus_minus == 'plus':
            acceptance_rejection_kappa = (self.density_max_val / self.ConstantPlus) * self.HoldAllVolume * safety_factor
        else:
            acceptance_rejection_kappa = (self.density_min_val / self.ConstantMinus) * self.HoldAllVolume * safety_factor

        out, out_shape = None, 0
        while out_shape < self.batch_size:
            samples = np.random.uniform(self.HoldAll_x, self.HoldAll_y, (self.batch_size*10, self.process_dimension))
            samples_in_domain = samples[np.where(IsInside(samples, nn_indicator))[0]]
            tracking_eval = np.expand_dims(np.array(list(map(u_tracking, samples_in_domain))), axis=1)
            if isinstance(state, TorchPoisson):
                density_evaluation = state(samples_in_domain).detach().numpy() - tracking_eval
            else:
                density_evaluation = np.array(list(map(state, samples_in_domain))) - np.array(list(map(u_tracking, samples_in_domain)))

            if plus_minus == 'plus':
                tmp_density = (density_evaluation[np.where(density_evaluation > 0)[0]] / self.ConstantPlus) * (self.HoldAllVolume / acceptance_rejection_kappa)
                uniform_plus = np.random.uniform(0, 1, tmp_density.shape[0])
                acceptance_rejection_indices = np.where(uniform_plus <= tmp_density[:, 0])
            else:
                tmp_density = (np.abs(density_evaluation[np.where(density_evaluation < 0)[0]]) / self.ConstantMinus) * (self.HoldAllVolume / acceptance_rejection_kappa)
                uniform_minus = np.random.uniform(0, 1, tmp_density.shape[0])
                acceptance_rejection_indices = np.where(uniform_minus <= tmp_density[:, 0])
                    
            if out is None:
                #out = samples_in_domain[np.argwhere(np.abs(density_evaluation) > 0)[:, 0]][acceptance_rejection_indices]
                if plus_minus == 'plus':
                    out = samples_in_domain[np.where(density_evaluation > 0)[0]][acceptance_rejection_indices]
                else:
                    out = samples_in_domain[np.where(density_evaluation < 0)[0]][acceptance_rejection_indices]
            else:
                if plus_minus == 'plus':
                    out = np.concatenate((out, samples_in_domain[np.where(density_evaluation > 0)[0]][acceptance_rejection_indices]),axis=0)
                else:
                    out = np.concatenate((out, samples_in_domain[np.where(density_evaluation < 0)[0]][acceptance_rejection_indices]), axis=0)
            out_shape += acceptance_rejection_indices[0].shape[0]
        return out[:self.batch_size]
            
    def BoundaryEulerMayuramaBatch(self, nn_indicator, x0 = None):
        #directly create a vectorized version
        b0 = np.zeros((self.batch_size, self.process_dimension))
        if x0 is None:
            x0 = np.array([0.5]*self.process_dimension)
            x0 = np.tile(x0, (self.batch_size, 1))
        
        #stores exit points
        exit_list = []
        while(True):
            #b1 is brownian motion
            b1 = b0 + np.sqrt(self.time_step_size) * np.random.normal(size=(b0.shape[0], self.process_dimension))
            #x1 = x0 + self.example.forward_drift() * self.example.time_step_size + (b1 - b0) @ self.example.sigma()
            
            x1 = x0 + np.sqrt(2)*(b1 - b0)
            
            containing_bools = IsInside(x1, nn_indicator)
            containing_indices = np.where(containing_bools)[0]
            not_containing_indices = np.where(1-containing_bools)[0]
            
            #FIXME: Slow for large self.batch_size? Weiterverarbeitung output sample start
            if not_containing_indices.shape[0] > 0:
                #for i in ProjectToBoundary(x1[not_containing_indices]):
                for i in x1[not_containing_indices]:
                    exit_list.append(i)
            
            x0 = x1[containing_indices]
            b0 = b1[containing_indices]
            
            if x0.shape[0] == 0:
                break
        return np.array(exit_list)
            
    def ExitPointBatch(self, SampleNumber, state, tracking, nn_indicator, MyBoundaryMesh):
        ExitPointList = []
        if self.ConstantPlus > 0:
            for i in range(SampleNumber):
                #Start_Tmp = self.SampleStart(self.acc_rej_kappa_Plus, self.ConstantPlus, state, tracking)
                Start_Tmp = self.sample_start(state, tracking, nn_indicator, "plus")
                ExitPoints = self.BoundaryEulerMayuramaBatch(nn_indicator, Start_Tmp)
                ExitPointList.append(ExitPoints)
            self.ExitPoints["+"], self.ExitPoints["CellIndicesPlus"] = ProjectToBoundary(np.array(ExitPointList).reshape(SampleNumber * self.batch_size, self.process_dimension), MyBoundaryMesh)
        ExitPointList = []
        if self.ConstantMinus > 0:
            for i in range(SampleNumber):
                #Start_Tmp = self.SampleStart(self.acc_rej_kappa_Minus, self.ConstantMinus, state, tracking)
                Start_Tmp = self.sample_start(state, tracking, nn_indicator, "minus")
                ExitPoints = self.BoundaryEulerMayuramaBatch(nn_indicator, Start_Tmp)
                ExitPointList.append(ExitPoints)
            self.ExitPoints["-"], self.ExitPoints["CellIndicesMinus"] = ProjectToBoundary(np.array(ExitPointList).reshape(SampleNumber * self.batch_size, self.process_dimension), MyBoundaryMesh)
