# this file computes the probabilistic representation of the shape derivative without using a mesh
# example: tracking type functional, Laplace PDE, unit ball
import numpy as np
import torch
import matplotlib.pyplot as plt
import tikzplotlib

from torch_methods import load_torch_model


def is_inside_ball(x):
    radius = 1.0
    return np.linalg.norm(x, axis=1, keepdims=True) < radius


def project_to_ball(x):
    radius = 1.0
    factor = 1.0/radius
    x_tmp = x
    norm_x = np.linalg.norm(x_tmp, axis=1, keepdims=True)
    x_new = x_tmp + ((1.0 - factor*norm_x)/(factor*norm_x))*x_tmp
    return x_new


def tracking_data(x):
    return x[:, 0] * (1 - x[:, 0]) * x[:, 1] * (1 - x[:, 1])


def v_dir(x):
    return np.sin(x)


def outer_normal(x):
    return x


def u_exact(x):
    return np.array([[-(p[0]**2+p[1]**2)/4 + 1/4] for p in x])


def grad_u_exact(x):
    return np.array([[-p[0]/2, -p[1]/2] for p in x])


def v_dir_exact(x: np.array, v_str: str) -> np.array:
    if v_str == "('sin(x[0])', 'sin(x[1])')":
        return np.sin(x)
    elif v_str == "('-sin(x[0])', '-sin(x[1])')":
        return -np.sin(x)
    elif v_str == "('cos(x[0])', 'cos(x[1])')":
        return np.cos(x)
    elif v_str == "('cos(x[0])', 'sin(x[1])')":
        return np.array([[np.cos(p[0]), np.sin(p[1])] for p in x])
    elif v_str == "('x[0]*x[1]', 'x[1]')":
        # return np.array([x[0]*x[1], x[1]])
        return np.array([[p[0]*p[1], p[1]] for p in x])
    elif v_str == "('1.0', '0.0')":
        return np.array([1.0, 0.0]) + 0.0*x
    elif v_str == "('1.0', '1.0')":
        return np.array([1.0, 1.0]) + 0.0*x
    elif v_str == "('2*x[0]', '2*x[1]')":
        return 2*x
    elif v_str == "('x[0]-x[1]', 'x[1]-x[0]')":
        return np.array([[p[0]-p[1], p[1]-p[0]] for p in x])
    elif v_str == "('x[0]*x[0]', 'x[0]*x[1]')":
        return np.array([[p[0]*p[0], p[1]*p[0]] for p in x])
    elif v_str == "('1.5*x[0]', '1.5*x[1]')":
        return np.array([[1.5*p[0], 1.5*p[1]] for p in x])
    elif v_str == "('abs(x[0]-x[1])*x[0]', 'abs(x[1]-x[0])*x[1]')":
        return np.array([[np.abs(p[0]-p[1])*p[0], np.abs(p[0]-p[1])*p[1]] for p in x])
    elif v_str == "('sin(6*atan(x[0]/x[1]))*x[0]', 'sin(6*atan(x[0]/x[1]))*x[1]')":
        return np.array([[np.sin(6*np.arctan(p[0]/p[1]))*p[0], np.sin(6*np.arctan(p[0]/p[1]))*p[1]] for p in x])
    elif v_str == "('0.3-x[0]', '0.2-x[1]')":
        return np.array([[0.3-p[0], 0.2-p[1]] for p in x])
    elif v_str == "('x[0]', 'x[1]')":
        return x
    else:
        raise NotImplementedError


class PaperDiffusion(object):
    def __init__(self):
        self.stepsize = 1.0/500.0  # 1.0/50.0
        self.batch_size = 100
        self.process_dimension = 2

        self.ExitPointsPlus = []
        self.ExitPointsMinus = []

        self.MeasureOmegaPlus, self.MeasureOmegaMinus = 0.0, 0.0
        self.ConstantPlus, self.ConstantMinus = 0.0, 0.0
        self.density_max_val, self.density_minus_val = 0.0, 0.0

        # load pde solution
        self.u_pinn = load_torch_model('pinn_model_example_laplace_zero_unit_sphere')

    def monte_carlo_measure_domain_partition(self, generate_plot=False, use_pinn=False):
        sample_number = int(1e7)

        box_volume = 9
        measure_samples = np.random.uniform([-1.5, -1.5], [1.5, 1.5], (sample_number, 2))

        samples_in_domain = measure_samples[np.where(is_inside_ball(measure_samples))[0]]
        number_samples_in_domain = samples_in_domain.shape[0]

        # evaluate samples in unit circle to check u-z_d > 0
        tracking_eval = tracking_data(samples_in_domain).reshape(number_samples_in_domain, 1)
        torch_samples_in_domain = torch.Tensor(samples_in_domain)
        if use_pinn:
            u_eval = self.u_pinn(torch_samples_in_domain).detach().numpy()
        else:
            u_eval = u_exact(samples_in_domain)

        density_eval = u_eval - tracking_eval
        number_samples_omega_plus = np.where(density_eval > 0)[0].shape[0]
        number_samples_omega_minus = number_samples_in_domain - number_samples_omega_plus

        if generate_plot:
            omega_plus = samples_in_domain[np.where(density_eval > 0)[0]]
            omega_minus = samples_in_domain[np.where(density_eval < 0)[0]]

            plt.scatter(omega_plus[:, 0], omega_plus[:, 1], s=2.5, label='$\Omega^+$', color='blue')
            plt.scatter(omega_minus[:, 0], omega_minus[:, 1], s=2.5, label='$\Omega^-$', color='red')

            from matplotlib.lines import Line2D
            from matplotlib.legend import Legend
            # dashed lines not supported by tikzplotlib - solution by:
            # michi7x7 -  Dashed Lines not supported since mpl 3.6.2 #567
            Line2D._us_dashSeq = property(lambda self: self._dash_pattern[1])
            Line2D._us_dashOffset = property(lambda self: self._dash_pattern[0])
            Legend._ncol = property(lambda self: self._ncols)

            lgnd = plt.legend()
            lgnd.legendHandles[0]._sizes = [30]
            lgnd.legendHandles[1]._sizes = [30]

            plt.axis('square')
            plt.savefig("./domain_laplace.eps", bbox_inches='tight')
            plt.savefig("./plot_domain_laplace_unit_circle")
            #tikzplotlib.save("./tikz_domain_partition_laplace_unit_circle")

        self.MeasureOmegaPlus = number_samples_omega_plus/sample_number * box_volume
        self.MeasureOmegaMinus = number_samples_omega_minus/sample_number * box_volume

        self.density_max_val = np.max([self.density_max_val, np.max(density_eval)])
        self.density_minus_val = np.max([self.density_minus_val, np.abs(np.min(density_eval))])

        print(f"Monte Carlo approximation - measure omega plus = {self.MeasureOmegaPlus} and measure omega minus = {self.MeasureOmegaMinus} (sum of both should be pi = {self.MeasureOmegaPlus + self.MeasureOmegaMinus})")
        pass

    def monte_carlo_constants(self, use_pinn=False):
        box_volume = 9
        max_iter = int(1e6)

        constant_plus, constant_minus = 0.0, 0.0
        sample_counter, sample_counter_plus, sample_counter_minus = 0, 0, 0
        while sample_counter < max_iter:
            samples = np.random.uniform([-1.5, -1.5], [1.5, 1.5], (max_iter, 2))
            samples_in_domain = samples[np.where(is_inside_ball(samples))[0]]

            tracking_eval = tracking_data(samples_in_domain).reshape(samples_in_domain.shape[0], 1)
            torch_samples_in_domain = torch.Tensor(samples_in_domain)
            if use_pinn:
                u_eval = self.u_pinn(torch_samples_in_domain).detach().numpy()
            else:
                u_eval = u_exact(samples_in_domain)

            density_evaluation = u_eval - tracking_eval

            density_eval_plus = density_evaluation[np.where(density_evaluation > 0)[0]]
            density_eval_minus = density_evaluation[np.where(density_evaluation < 0)[0]]

            # rejection test plus
            uniform_plus = np.random.uniform(0, 1, density_eval_plus.shape[0])
            acceptance_rejection_indices_plus = np.where(uniform_plus <= self.MeasureOmegaPlus / box_volume)[0]
            constant_plus += np.sum(density_eval_plus[acceptance_rejection_indices_plus])
            sample_counter_plus += acceptance_rejection_indices_plus.shape[0]

            uniform_minus = np.random.uniform(0, 1, density_eval_minus.shape[0])
            acceptance_rejection_indices_minus = np.where(uniform_minus <= self.MeasureOmegaMinus / box_volume)[0]
            constant_minus += np.sum(density_eval_minus[acceptance_rejection_indices_minus])
            sample_counter_minus += acceptance_rejection_indices_minus.shape[0]

            if self.MeasureOmegaMinus * self.MeasureOmegaPlus > 0:
                sample_counter = np.min([sample_counter_plus, sample_counter_minus])
            else:
                sample_counter = np.max([sample_counter_plus, sample_counter_minus])

        if sample_counter_plus > 0:
            self.ConstantPlus = self.MeasureOmegaPlus * constant_plus / sample_counter_plus
        else:
            self.ConstantPlus = 0.0
        if sample_counter_minus > 0:
            self.ConstantMinus = -1 * self.MeasureOmegaMinus * constant_minus / sample_counter_minus
        else:
            self.ConstantMinus = 0.0

        print(f"Monte Carlo sampling for constants completed. - constant plus = {self.ConstantPlus} with {sample_counter_plus} samples and constant minus = {self.ConstantMinus} with {sample_counter_minus} samples.")
        pass

    def sample_start(self, plus_minus: str, use_pinn=False):

        box_volume = 4
        safety_factor = 1.1
        if plus_minus == 'plus':
            acceptance_rejection_kappa = (self.density_max_val / self.ConstantPlus) * box_volume * safety_factor
        else:
            acceptance_rejection_kappa = (self.density_minus_val / self.ConstantMinus) * box_volume * safety_factor

        out, out_shape = None, 0
        while out_shape < self.batch_size:
            samples = np.random.uniform([-1, -1], [1, 1], (self.batch_size * 10, 2))
            samples_in_domain = samples[np.where(is_inside_ball(samples))[0]]

            tracking_eval = tracking_data(samples_in_domain).reshape(samples_in_domain.shape[0], 1)
            torch_samples_in_domain = torch.Tensor(samples_in_domain)
            if use_pinn:
                u_eval = self.u_pinn(torch_samples_in_domain).detach().numpy()
            else:
                u_eval = u_exact(samples_in_domain)

            density_evaluation = u_eval - tracking_eval

            if plus_minus == 'plus':
                tmp_density = (density_evaluation[np.where(density_evaluation > 0)[0]] / self.ConstantPlus) * (box_volume / acceptance_rejection_kappa)
                uniform_plus = np.random.uniform(0, 1, tmp_density.shape[0])
                acceptance_rejection_indices = np.where(uniform_plus <= tmp_density[:, 0])
            else:
                tmp_density = (np.abs(density_evaluation[np.where(density_evaluation < 0)[0]]) / self.ConstantMinus) * (box_volume / acceptance_rejection_kappa)
                uniform_minus = np.random.uniform(0, 1, tmp_density.shape[0])
                acceptance_rejection_indices = np.where(uniform_minus <= tmp_density[:, 0])

            if out is None:
                if plus_minus == 'plus':
                    out = samples_in_domain[np.where(density_evaluation > 0)[0]][acceptance_rejection_indices]
                else:
                    out = samples_in_domain[np.where(density_evaluation < 0)[0]][acceptance_rejection_indices]
            else:
                if plus_minus == 'plus':
                    out = np.concatenate((out, samples_in_domain[np.where(density_evaluation > 0)[0]][acceptance_rejection_indices]),axis=0)
                else:
                    out = np.concatenate(
                        (out, samples_in_domain[np.where(density_evaluation < 0)[0]][acceptance_rejection_indices]),
                        axis=0)
            out_shape += acceptance_rejection_indices[0].shape[0]
        return out[:self.batch_size]

    def euler_maruyama_exit_points(self, x0=None):
        b0 = np.zeros((self.batch_size, self.process_dimension))
        if x0 is None:
            x0 = np.array([0.0] * self.process_dimension)
            x0 = np.tile(x0, (self.batch_size, 1))

        exit_list = []
        while True:
            b1 = b0 + np.sqrt(self.stepsize) * np.random.normal(size=(b0.shape[0], self.process_dimension))
            x1 = x0 + np.sqrt(2) * (b1 - b0)

            containing_bools = is_inside_ball(x1)
            containing_indices = np.where(containing_bools)[0]
            not_containing_indices = np.where(1 - containing_bools)[0]

            if not_containing_indices.shape[0] > 0:
                for i in project_to_ball(x1[not_containing_indices]):
                    exit_list.append(i)

            x0 = x1[containing_indices]
            b0 = b1[containing_indices]

            if x0.shape[0] == 0:
                break
        return np.array(exit_list)

    def exit_point_batch(self, sample_number=1):
        exit_list = []
        if self.ConstantPlus > 0:
            for i in range(sample_number):
                start_plus = self.sample_start('plus')
                exit_points = self.euler_maruyama_exit_points(start_plus)
                exit_list.append(exit_points)
            self.ExitPointsPlus = np.array(exit_list).reshape(sample_number * self.batch_size, self.process_dimension)

        exit_list = []
        if self.ConstantMinus > 0:
            for i in range(sample_number):
                start_minus = self.sample_start('minus')
                exit_points = self.euler_maruyama_exit_points(start_minus)
                exit_list.append(exit_points)
            self.ExitPointsMinus = np.array(exit_list).reshape(sample_number * self.batch_size, self.process_dimension)
        pass


class MeshFreeProbabilisticShapeDerivative(PaperDiffusion):
    def __init__(self, perturbation_list):
        super(MeshFreeProbabilisticShapeDerivative, self).__init__()

        self.pinn_results_dict = {'name': 'PINN'}
        self.exact_results_dict = {'name': 'exact'}
        self.perturbations = perturbation_list

        self.sample_number_exit_points = 10000
        self.iter_range = self.sample_number_exit_points * self.batch_size

        # number of computations of $\D\Phi [V]$ for variance estimator
        self.monte_carlo_variance_estimator_number = 100

    def init_diffusion(self):
        print('---> sample measure, constants and exit points')
        self.monte_carlo_measure_domain_partition(generate_plot=False, use_pinn=True)
        self.monte_carlo_constants(use_pinn=True)
        self.exit_point_batch(self.sample_number_exit_points)
        print('---> exit points generated')
        pass

    def evaluate_shape_derivative(self):
        self.init_diffusion()
        print('evaluate expectations of probabilistic representation')

        # evaluate gradient of u -  once!
        if self.ConstantPlus > 0.0:
            pinn_u_grad_eval_plus = np.zeros(shape=(self.iter_range, self.process_dimension))
            for k in range(self.iter_range):
                torch_input = torch.Tensor(self.ExitPointsPlus[k])
                torch_input.requires_grad = True
                torch_eval = self.u_pinn(torch_input)
                pinn_u_grad_eval_plus[k] = torch.autograd.grad(torch_eval, torch_input, retain_graph=True)[0]
            exact_grad_u_eval_plus = grad_u_exact(self.ExitPointsPlus)

        if self.ConstantMinus > 0.0:
            pinn_u_grad_eval_minus = np.zeros(shape=(self.iter_range, self.process_dimension))
            for k in range(self.iter_range):
                torch_input = torch.Tensor(self.ExitPointsMinus[k])
                torch_input.requires_grad = True
                torch_eval = self.u_pinn(torch_input)
                pinn_u_grad_eval_minus[k] = torch.autograd.grad(torch_eval, torch_input, retain_graph=True)[0]
            exact_grad_u_eval_minus = grad_u_exact(self.ExitPointsMinus)

        for v in self.perturbations:
            v_str_test = str(v)
            print(v_str_test)

            pinn_mc_plus, pinn_mc_minus = 0.0, 0.0
            exact_mc_plus, exact_mc_minus = 0.0, 0.0

            if self.ConstantPlus > 0.0:
                v_dir_eval_exit_plus = v_dir_exact(self.ExitPointsPlus, v_str_test)

                for k in range(self.iter_range):
                    pinn_mc_plus += v_dir_eval_exit_plus[k, :] @ pinn_u_grad_eval_plus[k, :]
                    exact_mc_plus += v_dir_eval_exit_plus[k, :] @ exact_grad_u_eval_plus[k, :]
                pinn_mc_plus /= self.iter_range
                exact_mc_plus /= self.iter_range

            if self.ConstantMinus > 0.0:
                v_dir_eval_exit_minus = v_dir_exact(self.ExitPointsMinus, v_str_test)
                for k in range(self.iter_range):
                    pinn_mc_minus += v_dir_eval_exit_minus[k, :] @ pinn_u_grad_eval_minus[k, :]
                    exact_mc_minus += v_dir_eval_exit_minus[k, :] @ exact_grad_u_eval_minus[k, :]
                pinn_mc_minus /= self.iter_range
                exact_mc_minus /= self.iter_range

            print(f"Monte Carlo shape evaluation - PINN - plus = {pinn_mc_plus} and minus = {pinn_mc_minus}")
            print(f"Monte Carlo shape evaluation - exact u - plus = {exact_mc_plus} and minus = {exact_mc_minus}")

            # evaluate Stokes integral.
            n_boundary_samples = int(1e7)
            points_uniform = np.random.uniform(0.0, 2.0 * np.pi, n_boundary_samples)
            uniform_boundary_samples = np.column_stack((np.sin(points_uniform), np.cos(points_uniform)))

            v_dir_eval_boundary = v_dir_exact(uniform_boundary_samples, v_str_test)
            tracking_eval_boundary = tracking_data(uniform_boundary_samples)

            stokes_component = 0.0
            for k in range(n_boundary_samples):
                stokes_component += (1 / 2) * v_dir_eval_boundary[k, :] @ outer_normal(uniform_boundary_samples[k]) * \
                                    tracking_eval_boundary[k] ** 2

            stokes_component /= n_boundary_samples  # Monte Carlo rescaling
            stokes_component *= 2 * np.pi  # equalizer in terms of surface measure (from uniform distribution)
            print(f"stokes component value = {stokes_component}")

            pinn_monte_carlo_directional_derivative_value = - stokes_component + self.ConstantPlus * pinn_mc_plus - self.ConstantMinus * pinn_mc_minus

            exact_scaled_sum_exit_expectations = self.ConstantPlus * exact_mc_plus - self.ConstantMinus * exact_mc_minus
            exact_monte_carlo_directional_derivative_value = - stokes_component + exact_scaled_sum_exit_expectations

            print(f"mesh free monte carlo shape derivative - PINN = {pinn_monte_carlo_directional_derivative_value}")

            print(f"mesh free monte carlo - scaled sum of exit expectations = {exact_scaled_sum_exit_expectations}")
            print(f"mesh free monte carlo shape derivative - exact derivative = {exact_monte_carlo_directional_derivative_value}")

            self.pinn_results_dict.setdefault(v_str_test, []).append(pinn_monte_carlo_directional_derivative_value)
            self.exact_results_dict.setdefault(v_str_test, []).append(exact_monte_carlo_directional_derivative_value)
        pass

    def monte_carlo_shape_derivative_test(self):
        # mesh free method
        # this method evaluates the probabilistic representation of the shape derivative multiple times and afterwards estimates the variance
        for k in range(self.monte_carlo_variance_estimator_number):
            print(f"########################################## Monte Carlo Iteration {k}  ##########################################")
            self.evaluate_shape_derivative()
        pass
