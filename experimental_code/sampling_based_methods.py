import math
import lbuc.interval_utils
import matplotlib.pyplot as plt
from time import perf_counter

import sage.all as sg
import numpy as np

from scipy.optimize import curve_fit


def gen_n_tempo(acc, prob, dim):

    return int(np.ceil((1/acc) * (np.e / (np.e - 1)) * np.log((1/prob) + dim))) 


# Calculates the number of samples needed to find a Max Lip that is within eps
# of the true Lip with probability of (1-beta)
# Consersevative lip required.
def gen_n_eps_bar(eps, beta, dim, conservative_Lip):
    lg = conservative_Lip * 2 * np.sqrt(2)
    eps_bar = np.power(eps / lg, 2 * dim)
    n = np.ceil(np.log(beta) / np.log(1 - eps_bar))

    try:
        n = int(n)
    except:
        n = 10000000000

    return n    


def inv_weib(x, n, b, l):  # x - variable, n - scale parameter, b - shape parameter, l - location parameter

    return (b / n) * ((l - x)/ n)**(b - 1) * np.exp(-((l - x)/n)**b)


def fit_weibull(samples):
    q25, q75 = np.percentile(samples, [25, 75])

    bin_width = 2 * (q75 - q25) * len(samples) ** (-1 / 3)
    bins = round((max(samples) - min(samples)) / bin_width)
 
    ydata, xdata = np.histogram(samples, bins, density=True) 
    
    for i in range(len(xdata) - 1):
        xdata[i] = xdata[i] + ((xdata[i+1] - xdata[i]) / 2)
    xdata = xdata[:-1]

    bounds = {0: {'min': 0.001, 'max': 10},
              1: {'min': 0.001, 'max': 10},
              2: {'min': max(samples), 'max': 20}}

    popt, pcov = curve_fit(
        inv_weib, xdata, ydata,
        bounds=[(bounds[0]['min'], bounds[1]['min'], bounds[2]['min']),
                (bounds[0]['max'], bounds[1]['max'], bounds[2]['max'])],
        maxfev=1000000,
    )
    
    return popt  #popt[2]


def close_refinements(M, n, delta):
    for _ in range(n):
        M1 = M.random_refinement()
        while True:
            d = sg.random_vector(sg.RR, len(M1.y0), min=-delta, max=delta).apply_map(sg.RIF)
            if d.norm().upper() > delta or any(x not in y for (x,y) in zip(M1.y0 + d, M.y0)):
                continue
            M2 = M.with_y0(M1.y0 + d)
            break
        yield (M1, M2)


def expansion(s1, s2, t):
    return ( np.linalg.norm(s1.state(t) - s2.state(t))
           / (s1.system.y0 - s2.system.y0).norm().upper() )


def estimate_lipschitz_sampling(M, T, n, m, delta, **kwargs):
    trajectories = [
        [(M1.solve_numerical(T, **kwargs),
          M2.solve_numerical(T, **kwargs))
         for (M1, M2) in close_refinements(M, n, delta)]
        for _ in range(m)
    ]
    def res(t):
        Ls = [
            max(expansion(s1, s2, t) for s1, s2 in trajs)
            for trajs in trajectories
        ]
        try:
            return fit_weibull(Ls)[2]
        except:
            return None
    return res

def estimate_lipschitz_sampling_timed(M, T, n, m, delta, t):
    t0 = perf_counter()
    L = estimate_lipschitz_sampling(M, T, n, m, delta)
    res = L(t)
    t1 = perf_counter()
    return res, t1 - t0


def estimate_lipschitz_sensitive(M, T, n, m, delta, **kwargs):
    sol_sets = [
        M.variational_extension.solve_sampled(n, T, **kwargs) 
        for _ in range(m)
    ]
    def res(t):
        Ls = [float(sol_set.expansion_factor(t, delta)) for sol_set in sol_sets]
        return fit_weibull(Ls)[2]
        # return invweibull(*invweibull.fit(Ls)).mean()
    return res


def estimate_directional_lipschitz_sensitive(M, T, n, m, delta, **kwargs):
    sol_sets = [
        M.variational_extension.solve_sampled(n, T, **kwargs) 
        for _ in range(m)
    ]
    def res(t):
        k = len(M.y0)
        Ls = [sol_set.expansion_factors(t, delta) for sol_set in sol_sets]
        return [fit_weibull([float(L[i]) for L in Ls])[2]
                for i in range(k)]
    return res