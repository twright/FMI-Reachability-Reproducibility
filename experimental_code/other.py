import math
import numpy

import numpy as np

from scipy.optimize import curve_fit

def comp_volume(sol, T, t):

    volume = 0.0
    ts = 0.0
    
    while ts < T:

        interval= ([t.endpoints() for t in sol.state(ts)])
        volume += math.prod([abs(pair[0] - pair[1]) for pair in interval])*ts

        ts += t

    return volume    

def volume_error(solf_flow, sol, T, t):

    '''
        compute proportional volume difference of the solutions 
    '''
    
    volume_sen = 0.0
    volume_flow = 0.0

    ts = 0.0
    
    while ts < T:
        interval_sen = ([t.endpoints() for t in sol.state(ts)])
        interval_flow = ([t.endpoints() for t in solf_flow.state(ts)])

        volume_sen += math.prod([abs(pair[0] - pair[1]) for pair in interval_sen])
        volume_flow += math.prod([abs(pair[0] - pair[1]) for pair in interval_flow])

        ts += t

    volume_sen *= t
    volume_flow *= t


    return (((volume_flow-volume_sen)/volume_flow)*100)


def gen_n_tempo(acc, prob, dim):

    return int(np.ceil((1/acc) * (np.e / (np.e - 1)) * np.log((1/prob) + dim))) 
    
#Calculates the number of samples needed to find a Max Lip that is within eps of the true Lip with probability of (1-beta)
#Consersevative lip required.
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


def fit_weibul(samples):


    q25, q75 = np.percentile(samples, [25, 75])

    bin_width = 2 * (q75 - q25) * len(samples) ** (-1 / 3)
    bins = round((max(samples) - min(samples)) / bin_width)
 
    ydata, xdata = np.histogram(samples, bins, density=True) 
    
    for i in range(len(xdata) - 1):
        xdata[i] = xdata[i] + ((xdata[i+1] - xdata[i]) / 2)
    xdata = xdata[:-1]

    bounds = {0: {'min': 0.001, 'max': 10},
              1: {'min': 0.001, 'max': 10},
              2: {'min': max(samples), 'max': 10}}

    popt, pcov = curve_fit(inv_weib, xdata, ydata,
                               bounds=[(bounds[0]['min'], bounds[1]['min'], bounds[2]['min']),
                                       (bounds[0]['max'], bounds[1]['max'], bounds[2]['max'])],
                               maxfev=1000000)
    
    return popt[2]