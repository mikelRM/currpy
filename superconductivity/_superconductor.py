import numpy as np
from ._selfconsistency import sc_delta

class Superconductor:

    def __init__(self, mu=0, h=0, delta00=1, T=0.3, dynes=1e-3):
        self.__mu      = mu
        self.__h       = h
        self.__delta00 = delta00
        self.__ETA     = dynes
        self.__T       = T

        self.__delta   = delta00 * sc_delta(T, h)

    @property
    def mu(self):
        return self.__mu

    @property
    def h(self):
        return self.__h

    @property
    def T(self):
        return selg.__T

    @property
    def delta00(self):
        return self.__delta00

    @property
    def delta(self):
        return self.__delta

    @property
    def dynes(self):
        return self.__ETA

    @mu.setter
    def mu(self, x):
        self.__mu = x

    @h.setter
    def h(self, x):
        self.__h = x
        self.__delta = delta00 * sc_delta(self.__T, self.__h)

    @T.setter
    def T(self, x):
        self.__T = x
        self.__delta = delta00 * sc_delta(self.__T, self.__h)

    @delta00.setter
    def delta00(self, x):
        self.__delta00 = x
        self.__delta = delta00 * sc_delta(self.__T, self.__h)

    @dynes.setter
    def dynes(self, x):
        self.__ETA = x


    
    def dos_qp(self, E):
        '''
        Density of States of the quasiparticles (qp) a spin-split superconductor:
          - E: Energy 
          - h: Spin-splitting amplitude
          - Rdelta: Order parameter
        
        Returns a tuple containing (parallel, antiparallel) spin polarization
        '''
        aux_E = (E+self.__h+self.__mu, E-self.__h+self.__mu)
        return (np.sign(aux_E[0])*np.real((aux_E[0] + self.__ETA*1j) / np.sqrt((aux_E[0] + self.__ETA*1j)**2 - self.delta**2)),
                np.sign(aux_E[1])*np.real((aux_E[1] + self.__ETA*1j) / np.sqrt((aux_E[1] + self.__ETA*1j)**2 - self.delta**2)))
                



    def dos_cd(self, E):
        '''
        "Density of states" of the condensate (cd) with a given spin polarization
        in a spin-split superconductor.
          - E: Energy
          - h: spin-splitting amplitude
          - Rdelta: Superconducting order parameter
        
        Returns a tuple containing (parallel, antiparallel) spin polarization
        '''
        aux_E = (E+self.__h+self.__mu, E-self.__h+self.__mu)
        return (np.sign(aux_E[0])*np.real(self.__delta / np.sqrt((aux_E[0] + self.__ETA*1j)**2 - self.delta**2)),
                np.sign(aux_E[1])*np.real(self.__delta / np.sqrt((aux_E[1] + self.__ETA*1j)**2 - self.delta**2)))
            


    def gf_ret(self, E):
        '''
        Retarded Green's functions of a spin-split superconductor:
          - E: Energy
          - h: spin-splitting amplitude
          - Rdelta: Superconducting order parameter

        G_S = g tau_3 + if tau_2 is supposed. The return of the function
        is the tuple (g^R_up, g^R_down, f^R_up, f^R_down)
        '''

        aux = (E + self.__mu + 1j*self.__ETA + self.__h,
               E + self.__mu + 1j*self.__ETA - self.__h)

        D   = np.sqrt(np.power(aux,2) - self.__delta**2)
        D   = np.where(np.real(aux/D) < 0, -D, D)
        
        return (aux[0]/D[0], self.__delta/D[0],
                aux[1]/D[1], self.__delta/D[1])

    def gf_adv(self, E):
        '''
        RetardedAdvanced Green's functions of a spin-split superconductor:
          - E: Energy
          - h: spin-splitting amplitude
          - Rdelta: Superconducting order parameter

        G_S = g tau_3 + if tau_2 is supposed. The return of the function
        is the tuple (g^A_up, g^A_down, f^A_up, f^A_down)
        '''

        aux = (E + self.__mu + 1j*self.__ETA + self.__h,
               E + self.__mu + 1j*self.__ETA - self.__h)

        D   = np.sqrt(np.power(aux,2) - self.__delta**2)
        D   = np.where(np.real(aux/D) > 0, -D, D)
        
        return (aux[0]/D[0], self.__delta/D[0],
                aux[1]/D[1], self.__delta/D[1])
