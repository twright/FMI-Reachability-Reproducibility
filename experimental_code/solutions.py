from typing import List, Optional
from functools import reduce, partial

import sage.all as sg
import numpy as np
from scipy.optimize import minimize
import cvxpy as cp

from lbuc import System


def split_variational_vector(n, v):
    '''Helper function to split a vector produced from a variational system
    into a state vector and a sensitivity matrix.'''
    return (
        sg.vector(v[0:n]),
        sg.matrix([v[n*(i + 1):n*(i + 2)] for i in range(n)]),
    )

def split_variational_sol(n, f):
    '''Split a scipy/numpy variational system solution.'''
    return (
        (lambda t: split_variational_vector(n, f.sol(t))[0]),
        (lambda t: split_variational_vector(n, f.sol(t))[1]),
    )


class Solution:
    def __init__(self, sys: System, timedomain, runtime: float) -> None:
        self._system = sys
        self._timedomain = sg.RIF(timedomain)
        self._runtime = runtime

    @property
    def system(self) -> System:
        return self._system

    @property
    def timedomain(self) -> sg.RIF:
        return self._timedomain

    @property
    def runtime(self) -> int:
        return self._runtime

    def plot(self, xs=None, t=None, **kwargs):
        pass


class NumericalSolution(Solution):
    def __init__(self, sys : System, timedomain: sg.RIF, runtime: float, sol) -> None:
        Solution.__init__(self, sys, timedomain, runtime)
        self._sol = sol

    @property
    def sol(self):
        return self._sol

    def state(self, t):
       return self.sol.sol(t) 

    def _soli(self, i):
        return lambda t: self.state(t)[i]

    def _solx(self, x):
        return self._soli(
            list(self.system.x).index(
                self.system.R(x)))

    def expand(self, expansion_factors):
        return ExpandedNumericalSolution(
            self.system,
            self.timedomain,
            self.runtime,
            self.sol,
            expansion_factors,
        )
    
    def plot(self, xs=None, t=None, **kwargs):
        if xs is None:
            xs = list(self.system.x)
        if not isinstance(xs, (list, tuple)):
            xs = [xs]
        if t is None:
            t = self.timedomain.endpoints()
        return sg.plot(
            [ self._solx(x) for x in xs ],
            t,
            **kwargs,
        )


class VariationalSolution(NumericalSolution):
    @property
    def state(self):
        return split_variational_sol(self.system.state_dim, self.sol)[0]

    @property
    def sensitivity(self):
        return split_variational_sol(self.system.state_dim, self.sol)[1]

    def expansion_factors(self, t : float, delta=None) -> sg.vector:
        n = self.system.state_dim

        delta_vect = (
            (sg.vector([delta]*n)
             if isinstance(delta, (float, int))
             else delta)
            if delta is not None
            else self.system.state_radii)
        
        # Compute the expansion factor based on the sensitivity matrix
        sens = self.sensitivity(t)
        return sg.matrix([
            [ sens[i][j].abs()
            for i in range(n) ]
            for j in range(n) 
        ]).T*delta_vect

    def expansion_factor(self, t : float, delta : Optional[float] = None) -> sg.vector:
        return self.expansion_factors(t, delta).norm()*(self.system.state_dim)**(-1/2)

    def _soli(self, i):
        # Use raw state for flexibility in plotting
        return lambda t: self.sol.sol(t)[i]

    def plot_state(self, t=None, **kwargs):
        xs = list(self.system.x)[0:self.system.state_dim]
        return super().plot(xs, t, **kwargs)


def scale_interval(I, f):
    """Scale an interval by factor f."""
    x = I.center()
    r = I.absolute_diameter()/2
    return sg.RIF(x - r*f, x + r*f)


class NumericalSolutionSet(Solution):
    # .sol is a set of VariationalSolutions
    def __init__(self, sys : 'System', timedomain: sg.RIF, runtime : float,
            sols: List[NumericalSolution]) -> None:
        super().__init__(sys, timedomain, runtime)
        self._solutions = sols

    @property
    def solutions(self) -> List[NumericalSolution]:
        return self._solutions

    def plot(self, xs=None, t=None, **kwargs):
        return sum(
            (sol.plot(xs, t, **kwargs)
                for sol in self.solutions),
            sg.Graphics(),
        )

    def plot_state(self, t=None, **kwargs):
        xs = list(self.system.x)[0:self.system.state_dim]
        return self.plot(xs, t, **kwargs)

    def directional_lipschitz_pairwise(self, t : float, delta=None) -> sg.vector:
        n = len(self.system.x)

        return sg.vector([
            max(
                (abs(f.state(t)[i] - g.state(t)[i])/sg.vector(f.state(0) - g.state(0)).norm()
                 if sg.vector(f.state(0) - g.state(0)).norm()
                 else 0.0)
                    for f in self.solutions
                    for g in self.solutions
                    if delta is None or abs(f.state(0)[i] - g.state(0)[i]) < delta
            )
            for i in range(n)
        ])

    def lipschitz_pairwise(self, t : float, delta=None) -> sg.vector:
        return max(
            (np.linalg.norm(f.state(t) - g.state(t))/np.linalg.norm(f.state(0) - g.state(0))
            if np.linalg.norm(f.state(0) - g.state(0)) > 0
            else 0.0)
                for f in self.solutions
                for g in self.solutions
                if delta is None or np.linalg.norm(f.state(0) - g.state(0)) < delta
        )

    def lipschitz_single_scp(self, t, gamma=0.0, solC=None):
        if solC is None:
            solC = self.system.centered().solve_numerical(self.timedomain.edges())
        SC = solC.system

        L = cp.Variable()

        constraints = [
            L >= 0.0
        ] + [
            (  L*float((sol.system.y0 - SC.y0).apply_map(lambda x: x.center()).norm())
            + gamma
            >= float(sg.vector(sol.state(t) - solC.state(t)).norm()) )
            for sol in self.solutions
        ]

        return cp.Problem(cp.Minimize(L), constraints).solve()

    def lipschitz_scp(self, t, gamma=0.0, l0=1.0, p=2, solC=None):
        if solC is None:
            solC = self.system.centered().solve_numerical(self.timedomain.edges())
        SC = solC.system
        n = len(self.system.x)

        L = cp.Variable(n, nonneg=True)
        # Ls = [cp.Variable(nonneg=True)
            #   for _ in range(n)]
        c = np.array([1]*n)

        constraints = [
            (r - L*d + gamma) <= 0
            for sol in self.solutions
            for r in [np.array(sg.vector(sol.state(t) - solC.state(t)).apply_map(
                    lambda x: abs(float(x))))]
            for d in [float(sg.vector(sol.system.y0 - SC.y0).norm(p))]
        ]
        # constraints = [
        #     (  L[i]*abs(float((sol.system.y0 - SC.y0)[i].center()))
        #     + gamma
        #     >= d)
        #     for sol in self.solutions
        #     for d in [float(sg.vector(sol.state(t) - solC.state(t)).norm())]  
        #     for i in range(n)
        # ]

        cp.Problem(cp.Minimize(c.T @ L), constraints).solve()

        return sg.vector(L.value)

        # res = minimize(
        #     (lambda l: l),
        #     l0,
        #     constraints = [{
        #         'type': 'ineq', 'fun': (lambda l: l),
        #     }] + [{
        #         'type': 'ineq',
        #         'fun': (lambda l:
        #                 - sg.vector(sol.state(t) - solC.state(t)).norm()
        #                 + l * (sol.system.y0 - SC.y0).apply_map(lambda x: x.center())
        #                 + gamma)
        #     } for sol in self.solutions
        #     ],
        # )

        # return res, res.x[0]


class VariationalSolutionSet(NumericalSolutionSet):
    # .sol is a set of VariationalSolutions
    def __init__(self, sys : 'VariationalSystem', timedomain: sg.RIF,
            runtime: float, sols: List[VariationalSolution]) -> None:
        super().__init__(sys, timedomain, runtime, sols)

    @property
    def states(self):
        return [x.state for x in self.solutions]

    @property
    def sensitivities(self):
        return [x.sensitivity for x in self.solutions]

    def maximal_absolute_sensitivity(self, t: float) -> sg.matrix:
        n = self.system.state_dim
        return sg.matrix([
            [ max(s(t)[i][j].abs() for s in self.sensitivities)
            for i in range(n) ]
            for j in range(n)
        ])

    def lipschitz_vector_sensitive(self, t : float) -> sg.vector:
        n = self.system.state_dim
        
        # Compute the expansion factor based on each sensitivity matrix
        lvs = [
            sg.vector([y.norm(1) for y in s.sensitivity(t).rows()])
            for s in self.solutions
        ]

        # Take the componentwise maximum
        return sg.vector([
            max(lv[i] for lv in lvs)
            for i in range(n)
        ])

    def expansion_factors(self, t : float, delta=None) -> sg.vector:
        diams = sg.vector([y00.absolute_diameter()*0.5
                           for y00 in self.system.y0])
        #return self.lipschitz_vector_sensitive(t)*diams.norm(sg.Infinity)
        n = self.system.state_dim

        delta_vect = (
           (sg.vector([delta]*n)
            if isinstance(delta, (float, int))
            else delta)
           if delta is not None
           else self.system.state_radii
        )
        
        # Compute the expansion factor based on each sensitivity matrix
        efs = [
           s.expansion_factors(t, delta_vect)
           for s in self.solutions
        ]

        # Take the componentwise maximum
        return sg.vector([
           max(ef[i] for ef in efs)
           for i in range(n)
        ])

        # return self.maximal_absolute_sensitivity(t)*self.system.state_radii

    def expansion_factor(self, t : float, delta : Optional[float] = None) -> sg.vector:
        return self.expansion_factors(t, delta).norm()*(self.system.state_dim)**(-1/2)

    def expanded_states(self, t, delta=None):
        n = self.system.state_dim

        delta_vect = (
            (sg.vector([delta]*n)
             if isinstance(delta, (float, int))
             else delta)
            if delta is not None
            else self.system.state_radii
        )
        
        for state, sens in zip(self.states, self.sensitivities):
            # expansion_intervals = sg.vector([
                # sg.RIF(-contraction_factor*r, contraction_factor*r)
                    # for r in self.system.state_radii
            # ])
            expansion_intervals = sg.vector([
                #x.intersection()
                # Restrict expansion region to space in initial set around
                # centre
                sg.RIF(-d, d).intersection(x - y)
                for d, y, x
                in zip(delta_vect, state(0), self.system.y0)
            ])
            yield state(t).apply_map(sg.RIF) + sg.matrix([
               [ sg.RIF(sens(t)[i][j])
                for i in range(n) ]
                for j in range(n) 
            ])*expansion_intervals

    def expanded_state(self, t, delta=None):
        def _vector_union(xs, ys):
            return sg.vector(x.union(y) for x,y in zip(xs, ys))
        return reduce(_vector_union, self.expanded_states(t, delta))

    def expanded_solution(self, delta=None):
        return IntervalFnSolution(
            self.system,
            self.timedomain,
            # This only records the cost of running the numerical solver
            # since the expanded set is computed on evaluation
            self.runtime,
            partial(self.expanded_state, delta=delta),
        )


    def plot(self, xs=None, t=None, **kwargs):
        return sum(
            (sol.plot(xs, t, **kwargs)
                for sol in self.solutions),
            sg.Graphics(),
        )

    def plot_state(self, t=None, **kwargs):
        xs = list(self.system.x)[0:self.system.state_dim]
        return self.plot(xs, t, **kwargs)
    

class IntervalFnSolution(Solution):
    def __init__(self, sys : System, timedomain: sg.RIF, runtime, fn) -> None:
        super().__init__(sys, timedomain, runtime)
        self._fn = fn

    @property
    def fn(self):
        return self._fn

    def state(self, t):
        return self.fn(t) 

    def _soli(self, i):
        def res(t):
            state = self.state(t)
            return state[i] if len(state) > 0 else sg.RIF('NaN')

        return res

    def _solx(self, x):
        return self._soli(
            list(self.system.x).index(
                self.system.R(x)))
    
    def _solx_lower(self, x):
       return lambda t: self._solx(x)(t).lower() 

    def _solx_upper(self, x):
       return lambda t: self._solx(x)(t).upper() 
    
    def plot(self, xs=None, t=None, **kwargs):
        if xs is None:
            xs = list(self.system.x)
        if not isinstance(xs, (list, tuple)):
            xs = [xs]
        if t is None:
            t = self.timedomain.endpoints()
        return sg.plot(
            sum(
                [
                    [
                        self._solx_lower(x),
                        self._solx_upper(x),
                    ]
                    for x in xs
                 ],
                 [],
            ),
            t,
            **kwargs,
        )


class ExpandedNumericalSolution(NumericalSolution, IntervalFnSolution):
    def __init__(self, sys : System, timedomain: sg.RIF, runtime: float, sol, expansion_factors) -> None:
        NumericalSolution.__init__(self, sys, timedomain, runtime, sol)
        self._expansion_factors = expansion_factors

    @property
    def fn(self):
        return self.solution.sol

    @property
    def expansion_factors(self):
        return self._expansion_factors

    def expansion_vector(self, t: float) -> sg.vector:
        ef = self.expansion_factors(t)
        return sg.vector([
            sg.RIF(-x, x)
            for x in ef
        ])

    def state(self, t):
        return sg.vector([
            sg.RIF(str(u)) for u in self.sol.sol(t)
        ]) + self.expansion_vector(t)

    def _solx_lower(self, x):
       return lambda t: self._solx(x)(t).lower() 

    def _solx_upper(self, x):
       return lambda t: self._solx(x)(t).upper() 

    def plot(self, xs=None, t=None, **kwargs):
        if xs is None:
            xs = list(self.system.x)
        if not isinstance(xs, (list, tuple)):
            xs = [xs]
        if t is None:
            t = self.timedomain.endpoints()
        return sg.plot(
            sum(
                [
                    [
                        self._solx_lower(x),
                        self._solx_upper(x),
                    ]
                    for x in xs
                 ],
                 [],
            ),
            t,
            **kwargs,
        )


class FlowstarSolution(IntervalFnSolution):
    def __init__(self, sys : System, timedomain: sg.RIF, flowstar_result) -> None:
        super().__init__(sys, timedomain, flowstar_result.runtime, flowstar_result)

    @property
    def flowpipe(self):
        return self.fn