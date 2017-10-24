import numpy as np
from ._selfconsistency import sc_delta

#Riemman zeta of five:
ZETA_5 = 0
for n in range(1,1001):
    ZETA_5 += 1/n**5

def sampler_1d(points, lim=(-5,5), thr=0.2, bigstep=2e-2, shortstep=5e-4):
    '''
    Intelligent sampler for integration purposes:
    
    INPUT:
      - points      -   List of points that have to be sampled shorter
      - lim=(-5,5)  -   Ordered tuple containing maximum and minimum of the sample
      - thr=0.2     -   Distance to sample shorter around the points
      - bigstep=.   -   Steps between the points in the rough sampling
      - shorstep=.  -   Steps between the points in the thin  sampling

    OUTPUT:
      - Numpy array containing the sampling
    '''
    
    ret = np.arange(lim[0], lim[1]+shortstep, bigstep)
    
    points = np.sort(points)
    for p in points:
        idm = np.searchsorted(ret, p-thr, 'left')-1
        idp = np.searchsorted(ret, p+thr, 'left')
        
        ret = np.concatenate([ret[:idm], np.arange(ret[idm], ret[idp], shortstep), ret[idp:]])
    
    return ret


    
class Superconductor:

    '''
    TODO: DOC and ep_qdot
    '''
    
    def __init__(self, delta00=1, mu=0, T=0.3, h=0, dynes=1e-3):
        self.__mu      = mu
        self.__h       = h
        self.__delta00 = delta00
        self.__ETA     = dynes
        self.__Te      = T
        self.__Tp      = T       #THIS IS HANDLED AS A CONSTANT

        self.delta    = delta00 * sc_delta(T, h)
        self.points   = np.unique([-self.delta-mu-h, -self.delta-mu+h, self.delta-mu-h, self.delta-mu+h])



    @property
    def mu(self):
        return self.__mu

    @property
    def h(self):
        return self.__h

    @property
    def Te(self):
        return self.__Te

    @property
    def delta00(self):
        return self.__delta00

    @property
    def dynes(self):
        return self.__ETA

    @mu.setter
    def mu(self, x):
        self.__mu = x
        self.points  = np.unique([-self.delta-self.__mu-self.__h, -self.delta-self.__mu+self.__h,
                                   self.delta-self.__mu-self.__h,  self.delta-self.__mu+self.__h])
    
    @h.setter
    def h(self, x):
        self.__h = x
        self.__delta = delta00 * sc_delta(self.__Te, self.__h)
        self.points  = np.unique([-self.delta-self.__mu-self.__h, -self.delta-self.__mu+self.__h,
                                   self.delta-self.__mu-self.__h,  self.delta-self.__mu+self.__h])

            
    @T.setter
    def Te(self, x):
        self.__Te = x
        self.delta = delta00 * sc_delta(self.__Te, self.__h)
        self.points  = np.unique([-self.delta-self.__mu-self.__h, -self.delta-self.__mu+self.__h,
                                   self.delta-self.__mu-self.__h,  self.delta-self.__mu+self.__h])
        
    @delta00.setter
    def delta00(self, x):
        self.__delta00 = x
        self.delta = delta00 * sc_delta(self.__Te, self.__h)
        self.points  = np.unique([-self.delta-self.__mu-self.__h, -self.delta-self.__mu+self.__h,
                                   self.delta-self.__mu-self.__h,  self.delta-self.__mu+self.__h])

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
        return (np.sign(aux_E[0])*np.real(self.delta / np.sqrt((aux_E[0] + self.__ETA*1j)**2 - self.delta**2)),
                np.sign(aux_E[1])*np.real(self.delta / np.sqrt((aux_E[1] + self.__ETA*1j)**2 - self.delta**2)))
            


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

        D   = np.sqrt(np.power(aux,2) - self.delta**2)
        D   = np.where(np.real(aux/D) < 0, -D, D)
        
        return (aux[0]/D[0], self.delta/D[0],
                aux[1]/D[1], self.delta/D[1])

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

        D   = np.sqrt(np.power(aux,2) - self.delta**2)
        D   = np.where(np.real(aux/D) > 0, -D, D)
        
        return (aux[0]/D[0], self.delta/D[0],
                aux[1]/D[1], self.delta/D[1])


    def fermi(E):
        if self.__Te == 0:
            return np.where(E>0, 1, 0)
        else:
            return np.where(E/self.__Te>-20, np.where(E/self.__Te<20, 1/(1+np.exp(E/self.__Te)), 0), 1)


    def ep_qdot():
        pass



class Normal:

    def __init__(mu=0, T=0.3):
        self.__mu   = mu
        self.__Te   = T
        self.__Tp   = T   #THIS IS HANDLED AS A CONSTANT
        self.points = []


    @property
    def mu(self):
        return self.__mu

    @property
    def Te(self):
        return self.__Te

    @mu.setter
    def mu(self, x):
        self.__mu = x
    
    @T.setter
    def Te(self, x):
        self.__Te = x


    def dos_qp(E):
        try:
            return np.ones(len(E))
        except TypeError:
            return 1


    def gf_ret(self, E):
        '''
        Retarded Green's functions of a normal metal:
          - E: Energy
          - h: spin-splitting amplitude
          - Rdelta: Superconducting order parameter

        The return of the function is the tuple (g^R_up, g^R_down, f^R_up, f^R_down)
        '''
        try:
            return (np.ones(len(E)), np.zeros(len(E)),
                    np.ones(len(E)), np.zeros(len(E)))
        except TypeError:
            return (1, 0, 1, 0)
    

    def gf_adv(self, E):
        '''
        RetardedAdvanced Green's functions of a normal metal:
          - E: Energy
          - h: spin-splitting amplitude
          - Rdelta: Superconducting order parameter

        The return of the function is the tuple (g^A_up, g^A_down, f^A_up, f^A_down)
        '''
        try:
            return (-np.ones(len(E)), np.zeros(len(E)),
                    -np.ones(len(E)), np.zeros(len(E)))
        except TypeError:
            return (-1, 0, -1, 0)


    def ep_qdot():
        return self.__Te**5 - self.__Tp**5

    
    def fermi(E):
        if self.__Te == 0:
            return np.where(E>0, 1, 0)
        else:
            return np.where(E/self.__Te>-20, np.where(E/self.__Te<20, 1/(1+np.exp(E/self.__Te)), 0), 1)

        
