import numpy as np
from ._selfconsistency import sc_delta

#RESCALED RIEMMAN'S ZETA(5):
ZETA = 0
for n in range(1,1001):
    ZETA += 1/n**5
ZETA *= 48


def sampler_1d(points, lim=(-5,5), thr=0.2, bigstep=2e-2, shortstep=5e-4):
    '''
    Intelligent sampler for integration purposes:
    
    INPUT:
      - points      -   List of points that have to be sampled shorter
      - lim=(-5,5)  -   Ordered tuple containing maximum and minimum of 
                        the sample
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
        
        ret = np.concatenate(
          [ret[:idm], np.arange(ret[idm], ret[idp], shortstep), ret[idp:]])
    
    return ret


    
class Superconductor:

    '''
    TODO: DOC and ep_qdot
    '''
    
    def __init__(self, delta00=1, mu=0, T=0.3, h=0, dynes=1e-3):
        self._mu = mu
        self._h = h
        self._delta00 = delta00
        self._Te = T

        self.Tp = T  
        self.dynes = dynes
        self.delta = delta00 * sc_delta(T, h)
        self.points = np.unique([-self.delta-mu-h, -self.delta-mu+h,
                                 self.delta-mu-h, self.delta-mu+h])


    @property
    def mu(self):
        return self._mu

    @property
    def h(self):
        return self._h

    @property
    def Te(self):
        return self._Te

    @property
    def delta00(self):
        return self._delta00

    @mu.setter
    def mu(self, x):
        self._mu = x
        self.points = np.unique(
          [-self.delta-self._mu-self._h, -self.delta-self._mu+self._h,
           self.delta-self._mu-self._h,  self.delta-self._mu+self._h])
    
    @h.setter
    def h(self, x):
        self._h = x
        self.delta = self._delta00 * sc_delta(self._Te, self._h)
        self.points  = np.unique(
          [-self.delta-self._mu-self._h, -self.delta-self._mu+self._h,
           self.delta-self._mu-self._h,  self.delta-self._mu+self._h])
            
    @Te.setter
    def Te(self, x):
        self._Te = x
        self.delta = self._delta00 * sc_delta(self._Te, self._h)
        self.points  = np.unique(
          [-self.delta-self._mu-self._h, -self.delta-self._mu+self._h,
           self.delta-self._mu-self._h,  self.delta-self._mu+self._h])
        
    @delta00.setter
    def delta00(self, x):
        self._delta00 = x
        self.delta = self._delta00 * sc_delta(self._Te, self._h)
        self.points  = np.unique(
          [-self.delta-self._mu-self._h, -self.delta-self._mu+self._h,
           self.delta-self._mu-self._h,  self.delta-self._mu+self._h])

    
    def dos_qp(self, E):
        '''
        Density of States of the quasiparticles (qp) a spin-split 
        superconductor:
          - E: Energy 
          - h: Spin-splitting amplitude
          - Rdelta: Order parameter
        
        Returns a tuple containing (parallel, antiparallel) spin 
        polarization.
        '''
        aux_E = (E+self._h+self._mu, E-self._h+self._mu)
        return (
            np.sign(aux_E[0]) * np.real((aux_E[0]+self.dynes*1j)
                / np.sqrt((aux_E[0]+self.dynes*1j)**2 - self.delta**2)),
            np.sign(aux_E[1]) * np.real((aux_E[1]+self.dynes*1j)
                / np.sqrt((aux_E[1]+self.dynes*1j)**2 - self.delta**2)),
        )
                
    def dos_cd(self, E):
        '''
        "Density of states" of the condensate (cd) with a given spin
        polarization in a spin-split superconductor.
          - E: Energy
          - h: spin-splitting amplitude
          - Rdelta: Superconducting order parameter
        
        Returns a tuple containing (parallel, antiparallel) spin 
        polarization.
        '''
        aux_E = (E+self._h+self._mu, E-self._h+self._mu)
        return (
            np.sign(aux_E[0])*np.real(self.delta
                / np.sqrt((aux_E[0] + self.dynes*1j)**2 - self.delta**2)),
            np.sign(aux_E[1])*np.real(self.delta
                / np.sqrt((aux_E[1] + self.dynes*1j)**2 - self.delta**2)),
        )
            
    def gf_ret(self, E):
        '''
        Retarded Green's functions of a spin-split superconductor:
          - E: Energy
          - h: spin-splitting amplitude
          - Rdelta: Superconducting order parameter

        G_S = g tau_3 + if tau_2 is supposed. The return of the function
        is the tuple (g^R_up, g^R_down, f^R_up, f^R_down)
        '''
        aux = (E + self._mu + 1j*self.dynes + self._h,
               E + self._mu + 1j*self.dynes - self._h)

        D   = np.sqrt(np.power(aux,2) - self.delta**2)
        D   = np.where(np.real(aux/D) < 0, -D, D)
        
        return (aux[0]/D[0], self.delta/D[0],
                aux[1]/D[1], self.delta/D[1])

    def gf_adv(self, E):
        '''
        RetardedAdvanced Green's functions of a spin-split
        superconductor:
          - E: Energy
          - h: spin-splitting amplitude
          - Rdelta: Superconducting order parameter

        G_S = g tau_3 + if tau_2 is supposed. The return of the 
        function is the tuple (g^A_up, g^A_down, f^A_up, f^A_down)
        '''
        aux = (E + self._mu + 1j*self.dynes + self._h,
               E + self._mu + 1j*self.dynes - self._h)

        D   = np.sqrt(np.power(aux,2) - self.delta**2)
        D   = np.where(np.real(aux/D) > 0, -D, D)
        
        return (aux[0]/D[0], self.delta/D[0],
                aux[1]/D[1], self.delta/D[1])

    def fermi(E):
        if self._Te == 0:
            return np.where(E+self._mu>0, 1, 0)
        else:
            return 1/(1+np.exp((E+self._mu)/self._Te))


    def ep_qdot():
        pass


    
class Normal:
    def __init__(mu=0, T=0.3):
        self.mu = mu
        self.Te = T
        self.Tp = T 
        self.points = []

        
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

        The return of the function is the tuple 
        (g^R_up, g^R_down, f^R_up, f^R_down)
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

        The return of the function is the tuple 
        (g^A_up, g^A_down, f^A_up, f^A_down)
        '''
        try:
            return (-np.ones(len(E)), np.zeros(len(E)),
                    -np.ones(len(E)), np.zeros(len(E)))
        except TypeError:
            return (-1, 0, -1, 0)

    def ep_qdot():
        return self.Te**5 - self.Tp**5
    
    def fermi(E):
        if self.Te == 0:
            return np.where(E+self.mu>0, 1, 0)
        else:
            return 1/(1+np.exp((E+self.mu)/self.Te))
