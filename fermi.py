import numpy as np
from math import sqrt
from scipy.special import gamma as func_gamma

m = 0.511      # masa elektronu w MeV
alpha = 1/137

class FermiBeta():
    # FermiBeta methods work with energies given in MeV
    def __init__(self, Z, A, is_beta_plus=False):
        Z_p, self.A = Z, A
        self.Z = Z-1 if is_beta_plus else Z+1   # Z_p - mother/parent Z; Z - daughter Z
        self.sign = -1 if is_beta_plus else 1
        self.V_0 = 1.13* alpha**2 * Z_p**(4/3)
        self.T_min = 0 if is_beta_plus else m*self.V_0
        self.S = sqrt(1 - alpha**2 * Z**2) - 1

    def electron_shape(self, T, Q, I=1, transition_type=None):
        result = np.zeros(len(T))

        s = np.logical_and( self.T_min<T, T<=Q )
        T_sliced = T[s]

        result[s] = self._basic_fermi(T_sliced, Q, I, transition_type)
        return result

    def neutrino_shape(self, T, Q, I=1, transition_type=None):
        result = np.zeros(len(T))

        s = np.logical_and( self.T_min<=T, T<=Q )
        T_sliced = T[s]

        first = 0
        for i in range(len(s)):
            if s[i]:
                first = i
                break
        new_s = np.append( s[:first], s[first:] )

        P_normalized = self._basic_fermi(T_sliced, Q, I, transition_type)
        result[new_s] = P_normalized[::-1]
        return result

    def _basic_fermi(self, T, Q, I=1, transition_type=None):
        # normalized Fermi shape
        # makes sense only for T between T_min and Q
        gamma = 1 + T/m
        gamma_max = 1 + Q/m
        gamma_0 = gamma - self.sign*self.V_0
        eta_0 = np.sqrt(gamma_0**2 - 1)
        delta_0 = alpha*self.Z*gamma_0/eta_0

        C = 1 
        if transition_type != None:
            SPIN_DELTA, IS_PARITY_CHANGED = 0, 1
            if transition_type[IS_PARITY_CHANGED]:
                if transition_type[SPIN_DELTA] in (0,1,2):
                    value = (2, -4/3, 0)[transition_type[SPIN_DELTA]]
                    C = self._forbidden_shape_factor(T, Q, value)

        F = eta_0**(2*self.S) *np.e**( self.sign*np.pi*delta_0 ) *np.absolute( func_gamma(1+self.S+1j*delta_0) )**2
        P = gamma_0*eta_0 *(gamma_max-gamma)**2 *F *C

        return P if I==None else I/np.trapz(P, T) *P

    @staticmethod
    def _forbidden_shape_factor(T, Q, value):
        # shape factor for electron transition
        # makes sense only for T between T_min and Q
        E_e = T+m
        E_0 = Q+m
        E_nu = E_0-E_e
        p_e = np.sqrt( E_e**2 + 2*m*E_e ) # p in MeV
        beta = p_e/E_e

        return p_e**2 + E_nu**2 + value* beta**2 *E_nu*E_e
