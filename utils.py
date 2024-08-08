import numpy as np

def neutrino_cs(T):
    # neutrino cross-section
    T_min = 1.806
    fit = lambda x: 6.56706866e-44 - 1.92856195e-43*x + 8.66520979e-44*x*x
    result = np.zeros(len(T))
    s = T>=T_min
    result[s] = fit(T[s])
    return result
