import lbuc
# from lbuc.matrices import vec_to_numpy, mat_to_numpy
from lbuc.matrices import convert_vec, convert_mat


from solutions import NumericalSolution, VariationalSolution, VariationalSolutionSet, FlowstarSolution, NumericalSolutionSet

import sage.all as sg
from scipy.integrate import solve_ivp
import random
import numpy as np
import sympy

from time import perf_counter

def flatten_matrix(x):
    return sum(map(tuple, x), ())


def sympy_gens(gs):
    return [sympy.Symbol(repr(g)) for g in gs]


def vec_to_numpy(gs, v):
    t_ = sympy.var('t_')
    return sympy.lambdify((t_, sympy_gens(gs)),
                          convert_vec(v),
                          modules='numpy')


def mat_to_numpy(gs, m):
    t_ = sympy.var('t_')
    return sympy.lambdify((t_, sympy_gens(gs)),
                          convert_mat(m),
                          modules='numpy')    

class System(lbuc.System):
    @property
    def jacobian(self):
        return sg.jacobian(self.y, self.x)
    
    @property
    def variations(self):
        return [
            sg.var(f"δ_{x}_{y}")
                for x in self.x
                for y in self.x
        ]

    @property
    def R_extended(self):
        if self.R is sg.SR:
            return sg.SR
        else:
            return sg.PolynomialRing(
                self.R.base_ring(),
                ','.join(
                    [str(g) for g in list(self.x) + self.variations]
                )
            )

    @property
    def variational_matrix(self):
        n = len(self.x)

        return sg.matrix([
            self.variations[i*n:(i+1)*n]
                for i in range(n)
        ])

    @property
    def variational_extension(self):
        R = self.R_extended
        var_gens = (tuple(self.x) + tuple(self.variations)
            if self.R is sg.SR
            else R.gens())
        return VariationalSystem(
            R,
            var_gens,
            tuple(self.y0)
                + tuple(sg.identity_matrix(len(self.x)).list()),
            tuple(map(R, self.y))
                + tuple((self.jacobian*self.variational_matrix).list()),
        )

    def with_y0(self, y0, y0_ctx=None):
        assert len(y0) == len(self.x)
        assert y0_ctx is None or len(y0_ctx) == len(self.x)
        return self.__class__(self._R, self.x, y0, self.y,
            varmap=self.varmap,
            y0_ctx=self._y0_ctx if y0_ctx is None else y0_ctx)
    
    def random_refinement(self):
        y0_refinement = [
            random.uniform(*yy0.endpoints())
                for yy0 in self.y0
        ]
        return self.with_y0(y0_refinement)

    def centered(self):
        y0_centred = [
            yy0.center() for yy0 in self.y0
        ]
        return self.with_y0(y0_centred)

    def random_refinements(self, n):
        return [
            self.random_refinement()
                for _ in range(n)
        ]

    def solve_numerical(self, t, **kwargs):
        f = vec_to_numpy(self.x, self.y)
        jac = mat_to_numpy(self.x, self.jacobian)
        
        if "method" not in kwargs:
            kwargs["method"] = "LSODA"
        
        t0 = perf_counter()
        sol = solve_ivp(
            f,
            t,
            self.y0,
            jac=jac,
            vectorized=True,
            dense_output=True,
            **kwargs,
        )
        t1 = perf_counter()

        return NumericalSolution(self, t, t1 - t0, sol)

    def solve_flowstar(self, t, **kwargs):
        from flowstar_toolbox.reach import reach

        x = [str(x) for x in self.x]
        y = [
            str(yi).replace("?","")
                for yi in self.y
        ]
        y0 = list(self.y0)

        print(f"flowstar odes:")
        for xi,yi in zip(x,y):
            print(f"d{xi}/dt = {yi}")

        if isinstance(t, tuple):
            t = t[1]
        elif hasattr(t, 'upper'):
            t = t.upper()

        res = reach(x, y, y0, t, **kwargs)

        return FlowstarSolution(
            self,
            res.timedomain,
            res,
        )

    def solve_sensitive(self, n_samples, t, **kwargs):
        '''Solve using sensitivity-based reachability'''
        # Compute variational system extension
        V = self.variational_extension

        # Sample solutions to variational system
        sol_set = V.solve_sampled(n_samples, t, **kwargs)

        # Compute expansion factors
        expansion_factors = sol_set.expansion_factors

        # Solve base system (at midpoint)
        sol = self.solve_numerical(t, **kwargs)

        # Expand solution based on expansion factors
        sol_expanded = sol.expand(expansion_factors)

        return sol_expanded

    def solve_sampled(self, n_samples, t, **kwargs) -> NumericalSolutionSet:
        t0 = perf_counter()
        sols = [
            refinement.solve_numerical(t, **kwargs)
                for refinement
                in self.random_refinements(n_samples)
        ]
        t1 = perf_counter()

        return NumericalSolutionSet(
            self,
            t,
            t1 - t0,
            sols,
        )

    def solve_grid_sampled(self, n, t, **kwargs) -> NumericalSolutionSet:
        t0 = perf_counter()
        coord_axes = [
            np.linspace(*self.y0[i].endpoints(), n+1)[:-1] + self.y0[i].absolute_diameter()/(2*n)
                for i in range(len(self.x))
        ]
        grid = np.meshgrid(*coord_axes)
        centers = zip(*(M.flatten() for M in grid))
        sols = [
            self.with_y0(sg.vector(v)).solve_numerical(t, **kwargs)
                for v
                in centers
        ]
        t1 = perf_counter()

        return NumericalSolutionSet(
            self,
            t,
            t1 - t0,
            sols,
        )

    # Should be equivalent to Breach method
    def solve_sensitive_grid(self, n_samples, t, **kwargs):
        V = self.variational_extension
        sol_set = V.solve_grid_sampled(n_samples, t, **kwargs)
        delta = (1/n_samples)*V.state_radii
        return sol_set.expanded_solution(delta)
    
    def solve_lipschitz_sampled(self, n_samples, t, gamma=0.0, **kwargs):
        solC = self.centered().solve_numerical(t, **kwargs)
        sol_set = self.solve_sampled(n_samples, t, **kwargs)
        L = lambda t: sol_set.lipschitz_scp(t, gamma=gamma, solC=solC)
        diams = sg.vector([y00.absolute_diameter()*0.5 for y00 in self.y0])
        return solC.expand(lambda t: sg.vector([l*r for l, r in zip(L(t), diams)]))
        #

    def solve_lipschitz_sampled_fixed(self, n_samples, t, gamma=0.0, **kwargs):
        solC = self.centered().solve_numerical(t, **kwargs)
        sol_set = self.solve_sampled(n_samples, t, **kwargs)
        L = lambda t: sol_set.lipschitz_scp(t, gamma=gamma, p=sg.Infinity,
                                            solC=solC)
        diams = sg.vector([y00.absolute_diameter()*0.5 for y00 in self.y0])
        return solC.expand(lambda t: L(t)*diams.norm(sg.Infinity))


class VariationalSystem(System):
    @property
    def state_dim(self):
        return int((1/2) * ((4*len(self.x) + 1)**(1/2) - 1))

    @property
    def state_radii(self):
        # The expansion factor should be use the radius which is
        # half the range
        # (which was used directly in Paulius' code)
        # since we expand from the centre of the trajectory
        return sg.vector([y00.absolute_diameter() for y00 in self.y0[0:self.state_dim]]) * 0.5

    def solve_numerical(self, t, **kwargs):
        sol = super().solve_numerical(t, **kwargs)
        return VariationalSolution(self, sol.timedomain, sol.runtime, sol.sol)

    def solve_sampled(self, n_samples, t, **kwargs):
        t0 = perf_counter()
        sols = [
            refinement.solve_numerical(t, **kwargs)
                for refinement
                in self.random_refinements(n_samples)
        ]
        t1 = perf_counter()

        return VariationalSolutionSet(
            self,
            t,
            t1 - t0,
            sols,
        )

    def solve_grid_sampled(self, n, t, **kwargs) -> NumericalSolutionSet:
        t0 = perf_counter()
        dim = self.state_dim
        sens_initial = list(self.y0[dim:].apply_map(float))
        coord_axes = [
            np.linspace(*self.y0[i].endpoints(), n+1)[:-1] + self.y0[i].absolute_diameter()/(2*n)
                for i in range(dim)
        ]
        grid = np.meshgrid(*coord_axes)
        centers = zip(*(M.flatten() for M in grid))
        sols = [
            self.with_y0(sg.vector(list(v) + sens_initial)).solve_numerical(t, **kwargs)
                for v
                in centers
        ]
        t1 = perf_counter()

        return VariationalSolutionSet(
            self,
            t,
            t1 - t0,
            sols,
        )
