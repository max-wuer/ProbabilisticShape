from typing import Union
import dolfin as df
df.set_log_level(35)


def solve_fem_pde(measure_dx: df.Measure,
                  function_space: df.FunctionSpace(),
                  source_h: Union[float, df.Function, df.Expression],
                  discount_c: Union[float, df.Expression],
                  boundary_g: Union[float, df.Expression]) -> df.Function:

    def boundary(x, on_boundary):
        return on_boundary
    bc = df.DirichletBC(function_space, boundary_g, boundary)

    u_tmp = df.TrialFunction(function_space)
    v_tmp = df.TestFunction(function_space)
    a = -(df.inner(df.grad(u_tmp), df.grad(v_tmp)) + discount_c * u_tmp * v_tmp) * measure_dx
    L = source_h * v_tmp * measure_dx

    u_tmp = df.Function(function_space)
    df.solve(a == L, u_tmp, bcs=bc)

    return u_tmp


def tracking_type_functional(dx: df.Measure,
                             u: df.Function,
                             u_d: Union[df.Function, df.Expression]) -> float:
    j = (u - u_d) ** 2 / 2
    return df.assemble(j * dx)


def perturbe_mesh(mesh: df.Mesh,
                  step: Union[float, int],
                  V_dir: Union[df.Function, df.Expression]) -> df.Mesh:
    V_dir_step = V_dir.copy(True)
    V_dir_step.vector()[:] = V_dir_step.vector()[:] * step
    df.ALE.move(mesh, V_dir_step)
    return mesh


class FEMShapeDerivative(object):
    def __init__(self, perturbation_list):

        self.perturbations = perturbation_list
        self.feynman_results_dict = {'name': 'feynman'}
        self.volume_results_dict = {'name': 'volume'}
        self.boundary_results_dict = {'name': 'boundary'}

        #
        self.shape_functional_differences_dict = {'name': 'shape functional differences'}

        #
        self.my_mesh = df.UnitDiscMesh.create(df.MPI.comm_world, 44, 1, 2)
        self.domain_dx = df.Measure('dx', self.my_mesh)
        self.boundary_ds = df.Measure('ds', self.my_mesh)

        self.function_space_degree = 2
        self.FunctionSpace = df.FunctionSpace(self.my_mesh, 'CG', self.function_space_degree)
        self.VectorFunctionSpace = df.VectorFunctionSpace(self.my_mesh, 'CG', 1)  # self.function_space_degree)

        # solve the PDE with fem
        self.g_expr = df.Constant(0.0)
        self.c_expr = df.Constant(0.0)
        self.h_expr = df.Constant(-1.0)
        self.tracking_data_expr = df.Expression('x[0]*(1-x[0])*x[1]*(1-x[1])', degree=self.function_space_degree)

        self.u_fem = solve_fem_pde(self.domain_dx, self.FunctionSpace, self.h_expr, self.c_expr, self.g_expr)
        self.grad_u_fem = df.grad(self.u_fem)
        self.p_fem = solve_fem_pde(self.domain_dx, self.FunctionSpace, -(self.u_fem - self.tracking_data_expr), self.c_expr, self.g_expr)

    def eval_shape_derivative(self):

        for v in self.perturbations:
            v_str = str(v)
            v_expr = df.Expression(v, degree=self.function_space_degree)

            if v_str == "('sin(6*atan(x[0]/x[1]))*x[0]', 'sin(6*atan(x[0]/x[1]))*x[1]')":
                arctan_expr = df.Expression('sin(6*atan(x[0]/x[1]))', degree=self.function_space_degree)
                # test_cond = df.conditional(abs(arctan_expr) >= 0, arctan_expr, df.Constant(0.0))
                test_cond = df.conditional(abs(arctan_expr) >= 0, v_expr, df.Constant((0.0, 0.0)))
                v_interpolation = df.project(test_cond, self.VectorFunctionSpace)
            else:
                v_interpolation = df.interpolate(v_expr, self.VectorFunctionSpace)

            normals = df.FacetNormal(self.my_mesh)

            # feynman kac evaluation of probabilistic shape derivative
            stokes_component = df.assemble((1 / 2) * (self.tracking_data_expr ** 2) * df.dot(v_expr, normals) * self.boundary_ds)

            probabilistic_adjoint = solve_fem_pde(measure_dx=self.domain_dx, function_space=self.FunctionSpace,
                                                  source_h=df.Constant(0.0), discount_c=0.0,
                                                  boundary_g=df.dot(self.grad_u_fem, v_interpolation))
            fem_deter_exit_int = df.assemble(((self.u_fem - self.tracking_data_expr) * probabilistic_adjoint) * self.domain_dx)

            feynman_fem = (-1) * stokes_component + fem_deter_exit_int

            # classic boundary shape derivative
            boundary_fem = df.assemble((df.dot(self.grad_u_fem, normals) * df.dot(df.grad(self.p_fem), normals) + self.tracking_data_expr ** 2 / 2) * df.dot(v_expr, normals) * self.boundary_ds)

            # classic volume shape derivative
            tracking_data_interpol = df.interpolate(self.tracking_data_expr, self.FunctionSpace)

            pre_shd_1 = df.div(v_interpolation) * (self.u_fem - self.tracking_data_expr) ** 2 / 2
            pre_shd_2 = - (self.u_fem - self.tracking_data_expr) * df.dot(df.grad(tracking_data_interpol), v_interpolation)
            pre_shd_3 = - df.dot(df.grad(self.u_fem), df.dot(
                df.div(v_interpolation) * df.Identity(2) - df.grad(v_interpolation) - df.grad(v_interpolation).T,
                df.grad(self.p_fem)))
            pre_shd_4 = df.div(v_interpolation) * self.c_expr * self.u_fem * self.p_fem
            pre_shd_5 = - df.dot(df.div(self.h_expr * v_interpolation), self.p_fem)

            volume_fem = df.assemble((pre_shd_1 + pre_shd_2 + pre_shd_3 + pre_shd_4 + pre_shd_5) * self.domain_dx)

            self.feynman_results_dict.setdefault(v_str, []).append(feynman_fem)
            self.volume_results_dict.setdefault(v_str, []).append(volume_fem)
            self.boundary_results_dict.setdefault(v_str, []).append(boundary_fem)
        pass

    def shape_functional_differences(self, epsilons):
        j_omega = tracking_type_functional(self.domain_dx, self.u_fem, self.tracking_data_expr)

        for v in self.perturbations:
            v_str = str(v)
            v_expr = df.Expression(v, degree=self.function_space_degree)

            if v_str == "('sin(6*atan(x[0]/x[1]))*x[0]', 'sin(6*atan(x[0]/x[1]))*x[1]')":
                arctan_expr = df.Expression('sin(6*atan(x[0]/x[1]))', degree=self.function_space_degree)
                test_cond = df.conditional(abs(arctan_expr) >= 0, v_expr, df.Constant((0.0, 0.0)))
                v_interpolation = df.project(test_cond, self.VectorFunctionSpace)
            else:
                v_interpolation = df.interpolate(v_expr, self.VectorFunctionSpace)

            differences = []
            for stepsize in epsilons:
                self.my_mesh = perturbe_mesh(self.my_mesh, stepsize, v_interpolation)
                tmp_u_fem = solve_fem_pde(self.domain_dx, df.FunctionSpace(self.my_mesh, 'CG', self.function_space_degree), self.h_expr, self.c_expr, self.g_expr)

                tmp_j = tracking_type_functional(self.domain_dx, tmp_u_fem, self.tracking_data_expr)

                diff = tmp_j - j_omega
                differences.append(diff)
                self.shape_functional_differences_dict.setdefault(v_str, []).append(diff)

                v_interpolation.vector()[:] *= -1
                self.my_mesh = perturbe_mesh(self.my_mesh, stepsize, v_interpolation)
                v_interpolation.vector()[:] *= -1
        pass
