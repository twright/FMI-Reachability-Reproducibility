import sage.all as sg
from scipy.linalg import expm
from time import perf_counter

from enum import Enum, auto

from varsys import System


def random_matrix_with_eigenvalues(xs):
    dim = len(xs)
    D = sg.diagonal_matrix(xs)
    S = sg.random_matrix(sg.RR, dim)
    return S**(-1)*D*S


def random_2x2_matrix_with_complex_eigenvalue(x):
    dim = 2
    C = sg.matrix([
        [x.real(), -x.imag()],
        [x.imag(), x.real()],
    ])
    S = sg.random_matrix(sg.RR, dim)
    return S**(-1)*C*S


def linear_system_lipschitz(M, t):
    if len(M.x) == 1: 
        return (sg.jacobian(M.y, M.x)*t).exp().norm()
    else:
        # Use numpy for fast approximate matrix exponentiation
        return sg.Matrix(expm((sg.jacobian(M.y, M.x)*t).numpy())).norm()

def linear_system_lipschitz_vector(M, t):
    if len(M.x) == 1: 
        E = (sg.jacobian(M.y, M.x)*t).exp()
    else:
        # Use numpy for fast approximate matrix exponentiation
        E = sg.Matrix(expm((sg.jacobian(M.y, M.x)*t).numpy()))

    return sg.vector([
        y.norm(1)
            for y in E.rows()
    ])


class ODEClass(str, Enum):
    SADDLE = 'Saddle'
    STABLE_NODE = 'Stable'
    UNSTABLE_NODE = 'Unstable'


def classify_2x2(M):
    x, y = M.eigenvalues()

    if x.real() > 0 and y.real() > 0:
        return ODEClass.UNSTABLE_NODE
    if x.real() < 0 and y.real() < 0:
        return ODEClass.STABLE_NODE
    if x.real()*y.real() < 0:
        return ODEClass.SADDLE


def generate_random_2D_matricies(n):
    res = {
        ODEClass.SADDLE: [],
        ODEClass.STABLE_NODE: [],
        ODEClass.UNSTABLE_NODE: [],
    }

    while min(map(len, res.values())) < n:
        M = sg.random_matrix(sg.RR, 2, min=-1, max=1)
        M_class = classify_2x2(M)

        if len(res[M_class]) < n:
            res[M_class].append(M)

    return res


def generate_2D_linear_systems(R, x, n):
    return {
        k: [System(
            R,
            x,
            [sg.RIF(-1, 1)]*2,
            M*sg.vector(x) + sg.random_vector(sg.RR, 2, -1.0, 1.0),
        ) for M in Ms]
        for k, Ms in generate_random_2D_matricies(n).items()
    }


def is_stable(M):
    return all([e.real() <= 0 for e in M.eigenvalues()])


def generate_random_matricies(m, n):
    res = {
        True: [],
        False: [],
    }

    while min(map(len, res.values())) < n:
        M = sg.random_matrix(sg.RDF, m, min=-1, max=1)
        M_class = is_stable(M)

        if len(res[M_class]) < n:
            res[M_class].append(M)

    return res


def random_RIF_vector(m):
    res = []

    for _ in range(m):
        xl = sg.RIF.random_element()
        xh = sg.RIF.random_element(min=xl)
        res.append(sg.RIF(xl, xh))

    return sg.vector(res)


def generate_linear_systems(m, n):
    R, x = sg.PolynomialRing(sg.RDF, m,
        [f"x{i}" for i in range(1, m+1)]).objgens()
    return {
        k: [System(
            R,
            x,
            # Random initial conditions
            random_RIF_vector(m),
            # [sg.RIF(-1, 1)]*m,
            M*sg.vector(x)
                + sg.random_vector(sg.RDF, m, -1.0, 1.0),
        ) for M in Ms]
        for k, Ms in generate_random_matricies(m, n).items()
    }


def timed(f):
    def g(*args, **kwargs):
        t0 = perf_counter()
        res = f(*args, **kwargs)
        t1 = perf_counter()

        return res, t1 - t0
    
    return g