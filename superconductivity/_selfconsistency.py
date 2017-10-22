import scipy
from scipy.optimize import fsolve

import numpy as np

def _bcs_t(Delta, T, niter=100):
    '''
    This function computes the BCS order parameter given Delta and T.
    niter is the number of terms that are summed in the loop. Unless 
    specified, niter=100.
    
    For the calculations, used units are: 
        Delta --> Delta / (2pi T_c)
        T     --> T / T_c
    '''
    Delta = Delta * 1.76 / (2*np.pi)
    T = T*1.76
    
    i = np.arange(1, niter, 1)
    aux_sum = np.sum(-T / np.sqrt( ((i-0.5)*T)**2 + Delta**2 ))
    
    niter_aux = (niter*T)**2 + Delta**2
    return aux_sum + np.log(2*np.exp(0.577215664)) + np.log(niter*T + np.sqrt(niter_aux)) + \
             (1/24)*(niter*T**3 / (niter_aux**(1.5)))



def _bcs_th(Delta, T, h, niter=100):
    '''
    This function computes the BCS order parameter given Delta,
    T and h. niter is the number of terms that are summed in the loop. 
    Unless specified, niter=1000.
    
    For the calculations, energies are converted to: 
        Delta  --> Delta  / T_c
        Delta0 --> Delta0 / T_c
        T      --> T / T_c
        h      --> h / T_c
    '''
    info = fsolve(_bcs_t, 0.7, args=(T), full_output=1)
    if info[2] != 1:
        Delta0 = 0
        return Delta       #This way, finding the root sets Delta=0
        
    #else:
    
    Delta0 = info[0]*1.76
    Delta  = Delta*1.76
    T      = T*1.76
    h      = h*1.76

    n      = np.arange(niter)
    
    wn  = np.pi * T * (2*n+1)
    xip = np.sqrt(Delta**2 + (wn + 1j*h)**2)
    xim = np.sqrt(Delta**2 + (wn - 1j*h)**2)
    xi0 = np.sqrt(Delta0**2 + wn**2)
    
    return np.sum(0.5 * (1/xip + 1/xim) - 1/xi0).real



def template_bcs_th(h, T, Delta, niter=100):
    '''
        Template to find the solutions of h for each Delta(T).
        This way I am able to print those fancy curves.
    '''
    return _bcs_th(Delta, T, h, niter)




def sc_delta(T, h):
    '''
      Returns the value of Delta(T, h).

      All energies are in Delta_00 units
    '''
    info = fsolve(_bcs_th, 0.7, args=(T, h), full_output=True)

    if info[2] == 1:
        return info[0][0]
    else:
        return 0



