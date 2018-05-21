import scipy
from scipy.optimize import fsolve

import numpy as np

def _bcs_t(Delta, T, niter=50):
    '''
    This function computes the BCS order parameter given Delta and T.
    niter is the number of terms that are summed in the loop. Unless 
    specified, niter=100.
    
    For the calculations, used units are: 
        Delta --> Delta / (2pi T_c)
        T     --> T / T_c
    '''
    Delta = Delta*1.76 / (2*np.pi)
    T = T*1.76
    
    i = np.arange(1, niter, 1)
    aux_sum = np.sum(-T / np.sqrt( ((i-0.5)*T)**2 + Delta**2 ))
    
    niter_aux = (niter*T)**2 + Delta**2
    return aux_sum + np.log(2*np.exp(0.577215664)) \
        + np.log(niter*T + np.sqrt(niter_aux)) \
        + (1/24)*(niter*T**3 / (niter_aux**(1.5)))



def _bcs_th(Delta, T, h, Delta0, niter=100):
    '''
    This function computes the BCS order parameter given Delta,
    T and h. niter is the number of terms that are summed in the loop. 
    Unless specified, niter=1000.
    '''
    niter = max(niter, int(1/T))
    niter = min(niter, 1000)
    
    n      = np.arange(niter)
    
    wn  = np.pi * T * (2*n+1)
    xip = np.sqrt(Delta**2 + (wn + 1j*h)**2)
    xim = np.sqrt(Delta**2 + (wn - 1j*h)**2)
    xi0 = np.sqrt(Delta0**2 + wn**2)
    
    
    return np.sum(0.5 * (1/xip + 1/xim) - 1/xi0).real


def _template_bcs_th(h, T, Delta, niter=200):
    '''
        Template to find the solutions of h for each Delta(T).
        This way I am able to print those fancy curves.
    '''
    return _bcs_th(Delta, T, h, niter)


def _sc_delta_th(T, h, x0=1):
    '''
      Returns the value of Delta(T, h).

      All energies are in Delta_00 units
    '''
    info = fsolve(_bcs_t, 0.7, args=(T), full_output=True)
    if info[2] != 1:
        return 0.0
    
    
    if h == 0:
        return min(1, info[0][0])
        
    else:
        info = fsolve(_bcs_th, x0, args=(T, h, info[0][0]), full_output=True)
        
        if info[2] == 1:
            delta = info[0][0]

            if np.abs(delta) > np.abs(h):
                return min(1, delta)
                
    return 0.0

            

def sc_h(T, Delta, h0=0.0):
    '''
    Returns the value of h(T, Delta). Useful to find double possible 
    values of Delta(T,h).
    
    All energies are in Delta_00 units
    '''
    info = fsolve(_template_bcs_th, h0, args=(T,Delta), full_output=True)
    if info[2] == 1:
        return info[0][0]
    else:
        raise 0



def _usadel(s, eps, h=0, delta=1, tau=100, beta=0):
    '''
    Computes the system consisting on the Usadel equation of the system and
    the normalization condition. It's solution gives the GFs. 

    Parameters
    ----------
    s : 8-dim real array
        Real and imaginary parts of f0, f3, g0 and g3 respectively
    eps : real or cmplx
        Quasiparticle value of the energy
    h : real
        Value of the exchange field
    delta :
        Superconducting order parameter
    '''
    f0_re, f0_im, f3_re, f3_im, g0_re, g0_im, g3_re, g3_im = s
    
    f0 = f0_re + 1j*f0_im
    f3 = f3_re + 1j*f3_im
    g0 = g0_re + 1j*g0_im
    g3 = g3_re + 1j*g3_im
    
    if tau >= 100:
        w_4tau = 0
    else:
        w_4tau = 1/(4*tau)
        
    #Usadel equations
    us1 = -2*eps*f0 + 2*h*f3 + 2j*delta*g0 + 1j*w_4tau*(1+beta) * (f3*g3 - 3*f0*g0)
    us2 = -2*eps*f3 + 2*h*f0 + 2j*delta*g3 + 1j*w_4tau*((1-3*beta)*f0*g3 - (3-beta)*f3*g0)
    
    #Normalization condition
    nm1 = f0**2 + f3**2 + g0**2 + g3**2 - 1
    nm2 = f0*f3 + g0*g3
    
    return [np.real(us1), np.imag(us1), np.real(us2), np.imag(us2),
           np.real(nm1), np.imag(nm1), np.real(nm2), np.imag(nm2)]

def _jacobian_usadel (s, eps, h=0, delta=1, tau=100, beta=0):
    f0_re, f0_im, f3_re, f3_im, g0_re, g0_im, g3_re, g3_im = s
    
    if tau >= 100:
        w_4tau = 0
    else:
        w_4tau = 1/(4*tau)
    
    d_us1_re = [-2*np.real(eps) + 3*w_4tau*(1+beta)*g0_im, 2*np.imag(eps) + 3*w_4tau*(1+beta)*g0_re, 
                2*h - w_4tau*(1+beta)*g3_im, -w_4tau*(1+beta)*g3_re,
                3*w_4tau*(1+beta)*f0_im, -2*delta + 3*w_4tau*(1+beta)*f0_re,
               -w_4tau*(1+beta)*f3_im, -w_4tau*(1+beta)*f3_re]
    
    d_us1_im = [-2*np.imag(eps) - 3*w_4tau*(1+beta)*g0_re, 2*np.real(eps) + 3*w_4tau*(1+beta)*g0_im, 
                w_4tau*(1+beta)*g3_re, 2*h - w_4tau*(1+beta)*g3_im,
                2*delta - 3*w_4tau*(1+beta)*f0_re, + 3*w_4tau*(1+beta)*f0_im,
               -w_4tau*(1+beta)*f3_re, w_4tau*(1+beta)*f3_im]
    
    d_us2_re = [2*h - w_4tau*(1-3*beta)*g3_im, -w_4tau*(1-3*beta)*g3_re,
               -2*np.real(eps) + w_4tau*(3-beta)*g0_im, 2*np.imag(eps) + w_4tau*(3-beta)*g0_re,
               w_4tau*(3-beta)*f3_im, w_4tau*(3-beta)*f3_re,
               -w_4tau*(1-3*beta)*f0_im, -2*delta - w_4tau*(1-3*beta)*f0_re]
    
    d_us2_im = [w_4tau*(1-3*beta)*g3_re, 2*h - w_4tau*(1-3*beta)*g3_im,
               -2*np.imag(eps) - w_4tau*(3-beta)*g0_re, -2*np.real(eps) + w_4tau*(3-beta)*g0_im,
               -w_4tau*(3-beta)*f3_re, w_4tau*(3-beta)*f3_im,
               2*delta + w_4tau*(1-3*beta)*f0_re, - w_4tau*(1-3*beta)*f0_im]
    
    d_nm1_re = [2*f0_re, -2*f0_im, 2*f3_re, -2*f3_im, 2*g0_re, -2*g0_im, 2*g3_re, -2*g3_im]
    
    d_nm1_im = [2*f0_im, 2*f0_re, 2*f3_im, 2*f3_re, 2*g0_im, 2*g0_re, 2*g3_im, 2*g3_re]
    
    d_nm2_re = [f3_re, -f3_im, f0_re, -f0_im, g3_re, -g3_im, g0_re, -g0_im]
    
    d_nm2_im = [f3_im, f3_re, f0_im, f0_re, g3_im, g3_re, g0_im, g0_re]
    
    return [d_us1_re, d_us1_im, d_us2_re, d_us2_im,
            d_nm1_re, d_nm1_im, d_nm2_re, d_nm2_im]


def _sum_sc(delta, T, h, tau=100, beta=0, niter=100):
    delta0 = _sc_delta_th(T, 0)
    
    n = np.arange(0, niter, 1)
    wn_list = (1+2*n)*T*np.pi
    
    pre_res = [0, 0, 0, 0, 1, 0, 0, 0]
    s0 = pre_res
    
    suma = 0
    
    for wn in wn_list[::-1]:
        while True:
            res, info, ier, mssg = fsolve(_usadel, s0, args=(1j*wn, h, delta, tau, beta), fprime=_jacobian_usadel, full_output=True)
            if (res[4]>1e-10 and ier==1):
                pre_res = res
                s0 = pre_res
                break
            else:
                s0 = pre_res + 0.2*np.random.rand(8)
                pass
            
        suma += res[0]/delta - 1/np.sqrt(wn**2 + delta0**2)
        
    return suma


def sc_delta(T, h, tau=100, beta=0, niter=100, dD=0.1):
    delta_max = _sc_delta_th(T, h)
    
    if delta_max == 0:
        return 0.0
    elif delta_max < 0 or delta_max > 1:
        print("WARNING: something is working wrong on currpy._sc_delta_th")
        return 0.0
    
    if tau >= 100:
        return delta_max
    
    pre_delta = delta_max
    pre_suma = _sum_sc(pre_delta, T, h, tau, beta, niter)
    
    if pre_suma > 0:
        return delta_max
    elif _sum_sc(0.01, T, h, tau, beta, niter) < 0:
        return 0.0
    
    #else
    while True:
        delta = pre_delta - dD
        if delta<0: 
            delta = 0.01
        
        suma = _sum_sc(delta, T, h, tau, beta, niter)
        
        if suma > 0:
            return delta - (pre_delta - delta)/(pre_suma - suma) * suma
        
        else:
            pre_delta = delta
            pre_suma = suma
            

def _gfs_ret (eps, T, h, delta, tau=100, beta=0, pre_gf=[]):
    if pre_gf == [] or tau >= 100:
        delta_th = _sc_delta_th(T, h)

        gp = (eps-h) / np.sqrt((eps-h)**2 - delta_th**2 + 0j) / 2
        gm = (eps+h) / np.sqrt((eps+h)**2 - delta_th**2 + 0j) / 2
        fp = (1j*delta_th) / np.sqrt((eps-h)**2 - delta_th**2 + 0j) / 2
        fm = (1j*delta_th) / np.sqrt((eps+h)**2 - delta_th**2 + 0j) / 2
        
        fp = fp * np.where(gp < 0, -1, 1)
        fm = fm * np.where(gm < 0, -1, 1)
        gp = gp * np.where(gp < 0, -1, 1)
        gm = gm * np.where(gm < 0, -1, 1)

        s0 = np.array([np.real(fp+fm), np.imag(fp+fm),np.real(fp-fm), np.imag(fp-fm),
                       np.real(gp+gm), np.imag(gp+gm),np.real(gp-gm), np.imag(gp-gm)])
        
        if tau >= 100:
            return s0

    else:
        s0 = pre_gf
        
    count = 0
    while True:
        count += 1
        
        res, info, ier, mssg = fsolve(_usadel, s0, args=(eps, h, delta, tau, beta), fprime=_jacobian_usadel, full_output=True)
        s0 = pre_gf +  0.1*np.random.rand(8)
        
        if (res[4]>0 and np.sign(eps)*res[1]>=0 and ier==1):
            break
            
        elif count > 300 and res[4]>0 and ier==1:
            break
    
    return res



def gfs_full (h=0, T=0.005, tau=100, beta=0, E_max=3, dE=0.01):
    """
    gfs_full (h=0, T=0.005, tau=100, beta=0, E_max=3, dE=0.01)
    
    Calculates de Green's Functions of a superconductor with spin orbit,
    spin flipping and exchange fields.

    Parameters
    ----------
    h : real
        Exchange field (in units of $\\Delta_{00}$)
        
    T : real
        Temperature (in units of $\\Delta_{00}$)
        
    tau : real
        $\\tau_{sn}^{-1} = \\tau_{sf}^{-1} + \\tau_{so}^{-1}$ (in units of $\\Delta_{00}^{-1}$)
        
    beta : real
        Dial-parameter to switch between spin-relaxation (beta=1) and spin-orbit relaxation (beta<1)
        $$\\beta = \frac{\\tau_{so} - \\tau_{sf}}{\\tau_{so} + \\tau_{sf}}
        
    E_max : real
        Boundaries of the energy sample where the GFs are calculated
        
    dE : real
        Spacing of the energy sample
        
    Returns
    -------
    eps_list : ndarray
        Energy sampling array
        
    gfs_ret : ndarray
        Real and imaginary parts of all the retarded GFs. It has shape (len(eps_list), 8) 
        with the following components:
        ndarray([[f0_re, f0_im, f3_re, f3_im, g0_re, g0_im, g3_re, g3_im], [...], ...])
        
    gfs_adv : ndarray
        Same as gfs_ret, but for the advanced GFs
    """
    
    E_max = np.abs(E_max)
    dE = np.abs(dE)
    T = max(T, 0.005)
    
    eps_list = np.arange(E_max, 0-dE, -dE)
    delta = sc_delta(T, h, tau, beta)
    
    if tau >= 100:
        gfsp = _gfs_ret(eps_list, T, h, delta, tau, beta)
        gfsm = _gfs_ret(-eps_list, T, h, delta, tau, beta)
        
        gfsp = np.transpose(gfsp)
        gfsm = np.transpose(gfsm)
        
    else:
        gfsp = []
        gfsm = []
        
        auxp = [0, 0, 0, 0, 1, 0, 0, 0]
        auxm = [0, 0, 0, 0, 1, 0, 0, 0]
        for ie, eps in enumerate(eps_list):
            auxp = _gfs_ret(eps, T, h, delta, tau, beta, pre_gf=auxp)
            auxm = _gfs_ret(-eps, T, h, delta, tau, beta, pre_gf=auxm)
            
            gfsp.append(auxp)
            gfsm.append(auxm)
            
        gfsp = np.array(gfsp)
        gfsm = np.array(gfsm)
        

    eps_list = np.append(-eps_list, eps_list[::-1])
    gfs_ret = np.append(gfsm, gfsp[::-1,:], axis=0)
    gfs_adv = np.array([1, -1, 1, -1, -1, 1, -1, 1]) * gfs_ret
    
    return (eps_list, gfs_ret, gfs_adv)
