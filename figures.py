import numpy as np
import matplotlib.pyplot as plt

def make_ref_fig(T, *args, file_paths=[]):
    if len(args)==6:
        P_e, P_nu, cs_P_nu, P_e_ref, P_nu_ref, cs_P_nu_ref = args
        fig, axs = plt.subplots(2, 3)
        fig.set_size_inches(15, 8)
    elif len(args)==4:
        P_e, P_nu, P_e_ref, P_nu_ref = args
        fig, axs = plt.subplots(2, 2)
        fig.set_size_inches(10, 8)
    else:
        raise Exception('Figure: wrong number of columns, should be 4 or 6')

    row1, row2 = axs[0], axs[1]
    fig.subplots_adjust(top=0.85)
    
    row1[0].plot(T, P_e, label='P_e-')
    row1[1].plot(T, P_nu,  label='P_nu')
    row1[0].plot(T, P_e_ref, label='P_e- ref')
    row1[1].plot(T, P_nu_ref,  label='P_nu ref')
    row1[0].set_xlabel('E [MeV]'); row1[0].set_ylabel('Liczba zliczeń [1/MeV]')
    row1[1].set_xlabel('E [MeV]'); row1[1].set_ylabel('Liczba zliczeń [1/MeV]')
    row1[0].legend(); row1[1].legend()

    row2[0].axhline(0, color='lightgray', linestyle='-')
    row2[1].axhline(0, color='lightgray', linestyle='-')
    row2[0].plot(T, P_e-P_e_ref, label='różnica P_e-')
    row2[1].plot(T, P_nu-P_nu_ref,  label='różnica P_nu')
    row2[0].set_xlabel('E [MeV]'); row2[0].set_ylabel('Liczba zliczeń [1/MeV]')
    row2[1].set_xlabel('E [MeV]'); row2[1].set_ylabel('Liczba zliczeń [1/MeV]')
    row2[0].legend(); row2[1].legend()

    if len(args)==6:
        row1[2].plot(T, cs_P_nu, label='sigma*P_nu')
        row1[2].plot(T, cs_P_nu_ref, label='sigma*P_nu ref')
        row1[2].set_xlabel('E [MeV]'); row1[2].set_ylabel('Liczba oddziaływań [cm2/MeV]')
        row1[2].set_xlim(left=1.7); row1[2].legend()

        row2[2].axhline(0, color='lightgray', linestyle='-')
        row2[2].plot(T, cs_P_nu-cs_P_nu_ref, label='różnica sigma*P_nu')
        row2[2].set_xlabel('E [MeV]'); row2[2].set_ylabel('Liczba oddziaływań [cm2/MeV]')
        row2[2].set_xlim(left=1.7); row2[2].legend()

    fig.suptitle('\n'.join(file_paths), horizontalalignment='left', x=0.1)
    return fig

def make_fig(T, *args, file_paths=[]):
    if len(args)==3:
        P_e, P_nu, cs_P_nu = args
        fig, row1 = plt.subplots(1, 3)
        fig.set_size_inches(15, 4)
    elif len(args)==2:
        P_e, P_nu = args
        fig, row1 = plt.subplots(1, 2)
        fig.set_size_inches(10, 4)
    else:
        raise Exception('Figure: wrong number of columns, should be 2 or 3')

    fig.subplots_adjust(top=0.8)

    row1[0].plot(T, P_e, label='P_e-')
    row1[1].plot(T, P_nu,  label='P_nu')
    row1[0].set_xlabel('E [MeV]'); row1[0].set_ylabel('Liczba zliczeń [1/MeV]')
    row1[1].set_xlabel('E [MeV]'); row1[1].set_ylabel('Liczba zliczeń [1/MeV]')
    row1[0].legend(); row1[1].legend()

    if len(args)==3:
        row1[2].plot(T, cs_P_nu, label='sigma*P_nu')
        row1[2].set_xlabel('E [MeV]'); row1[2].set_ylabel('Liczba oddziaływań [cm2/MeV]')
        row1[2].set_xlim(left=1.7); row1[2].legend()

    fig.suptitle('\n'.join(file_paths), horizontalalignment='left', x=0.1)
    return fig
