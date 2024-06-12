import numpy as np
import matplotlib.pyplot as plt

def neutrino_cs(T):
    # neutrino cross-section
    T_min = 1.806
    fit = lambda x: 6.56706866e-44 - 1.92856195e-43*x + 8.66520979e-44*x*x
    result = np.zeros(len(T))
    s = T>=T_min
    result[s] = fit(T[s])
    return result

def make_ref_fig(T, P_e, P_nu, P_e_ref, P_nu_ref):
    fig, axs = plt.subplots(2, 3)
    row1, row2 = axs[0], axs[1]

    row1[0].plot(T, P_e, label='P_e-')
    row1[1].plot(T, P_nu,  label='P_nu')
    row1[2].plot(T, neutrino_cs(T)*P_nu, label='sigma*P_nu')
    row1[0].plot(T, P_e_ref, label='P_e- ref')
    row1[1].plot(T, P_nu_ref,  label='P_nu ref')
    row1[2].plot(T, neutrino_cs(T)*P_nu_ref, label='sigma*P_nu ref')

    row1[0].set_xlabel('E [MeV]'); row1[0].set_ylabel('Liczba zliczeń [1/MeV]')
    row1[1].set_xlabel('E [MeV]'); row1[1].set_ylabel('Liczba zliczeń [1/MeV]')
    row1[2].set_xlabel('E [MeV]'); row1[2].set_ylabel('Liczba oddziaływań [cm2/MeV]')
    row1[2].set_xlim(left=1.7)
    row1[0].legend(); row1[1].legend(); row1[2].legend()

    row2[0].plot(T, P_e-P_e_ref, label='różnica P_e-')
    row2[1].plot(T, P_nu-P_nu_ref,  label='różnica P_nu')
    row2[2].plot(T, neutrino_cs(T)*(P_nu-P_nu_ref), label='różnica sigma*P_nu')

    row2[0].set_xlabel('E [MeV]'); row2[0].set_ylabel('Liczba zliczeń [1/MeV]')
    row2[1].set_xlabel('E [MeV]'); row2[1].set_ylabel('Liczba zliczeń [1/MeV]')
    row2[2].set_xlabel('E [MeV]'); row2[2].set_ylabel('Liczba oddziaływań [cm2/MeV]')
    row2[2].set_xlim(left=1.7)
    row2[0].legend(); row2[1].legend(); row2[2].legend()

    fig.set_size_inches(15, 8)
    return fig

def make_fig(T, P_e, P_nu):
    fig, row1 = plt.subplots(1, 3)

    row1[0].plot(T, P_e, label='P_e-')
    row1[1].plot(T, P_nu,  label='P_nu')
    row1[2].plot(T, neutrino_cs(T)*P_nu, label='sigma*P_nu')

    row1[0].set_xlabel('E [MeV]'); row1[0].set_ylabel('Liczba zliczeń [1/MeV]')
    row1[1].set_xlabel('E [MeV]'); row1[1].set_ylabel('Liczba zliczeń [1/MeV]')
    row1[2].set_xlabel('E [MeV]'); row1[2].set_ylabel('Liczba oddziaływań [cm2/MeV]')
    row1[2].set_xlim(left=1.7)
    row1[0].legend(); row1[1].legend(); row1[2].legend()

    fig.set_size_inches(15, 4)
    return fig
