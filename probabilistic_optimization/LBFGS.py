import numpy as np
from dolfin import Function

class MyLBFGS:
    def __init__(self):
        #"step" or "secant" list: Stores all H1-gradient boundary deformation fields
        self.s_list = []
        #"list of differences in gradients" for 2nd order directional derivative approximation
        #along s, i.e., y = s_k+1 - s_k: This is a sekant through the gradients
        self.y_list = []
        #scaling factor to satisfy the secant condition: 1 / (y_k ⋅ s_k)
        self.rho_list = []
        #memory length
        self.m = 5
        
        # Store previous gradient to compute y_k
        self.prev_gradient = None

    def Store(self, V_current):
        """
        Store current H1 deformation field in L-BFGS memory.
        
        Parameters
        ----------
        V_current : dolfin.Function
            Current H1-identified deformation field (vector Function)
        """
        # Convert current deformation to numpy array
        V_array = V_current.vector().get_local()

        if self.prev_gradient is None:
            # First iteration: store and return
            self.prev_gradient = V_array.copy()
            return

        # Step s_k is the deformation applied (current vector)
        s_k = V_array.copy()
        # y_k is the difference in H1 gradient
        y_k = V_array - self.prev_gradient

        # Curvature condition
        denom = np.dot(s_k, y_k)
        if denom > 1e-10:
            rho_k = 1.0 / denom

            # Maintain rolling memory
            if len(self.s_list) == self.m:
                self.s_list.pop(0)
                self.y_list.pop(0)
                self.rho_list.pop(0)

            self.s_list.append(s_k)
            self.y_list.append(y_k)
            self.rho_list.append(rho_k)

        # Update previous gradient
        self.prev_gradient = V_array.copy()

    def TwoLoopRecursion(self, g):
        """
        Compute the L-BFGS search direction for a given gradient g.
        
        Parameters
        ----------
        g : numpy array from V_current
            Current H1-identified deformation field as numpy array.
        
        Returns
        -------
        p : numpy array
            Search direction for L-BFGS step
        """
        q = g.copy()
        alpha_list = []

        # Backward loop
        for s, y, rho in reversed(list(zip(self.s_list, self.y_list, self.rho_list))):
            a = rho * np.dot(s, q)
            alpha_list.append(a)
            q -= a * y

        # Initial Hessian scaling (scalar)
        if self.s_list:
            H0 = np.dot(self.s_list[-1], self.y_list[-1]) / np.dot(self.y_list[-1], self.y_list[-1])
        else:
            H0 = 1.0
        r = H0 * q

        # Forward loop
        for i, (s, y, rho) in enumerate(zip(self.s_list, self.y_list, self.rho_list)):
            a = alpha_list[-(i+1)]
            beta = rho * np.dot(y, r)
            r += s * (a - beta)

        p = -r
        return p

    def ApplyStep(self, V_current, alpha=1.0):
        """
        Compute and return the L-BFGS step as a FEniCS Function.
        
        Parameters
        ----------
        V_current : dolfin.Function
            Current H1-identified deformation field
        alpha : float
            Step scaling factor
        
        Returns
        -------
        V_step : dolfin.Function
            L-BFGS deformation field to apply
        """
        V_array = V_current.vector().get_local()
        p_array = self.TwoLoopRecursion(V_array) * alpha

        # Convert back to FEniCS Function
        V_step = Function(V_current.function_space())
        V_step.vector()[:] = p_array
        return V_step
