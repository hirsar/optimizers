"""
Author: Hiren Saravana
Implementation of gradient free optimizer using Nelder Mead.
"""

import numpy as np

def reflect(x, xc, a=1.0):
    '''Function that performs reflection.'''
    xr = xc + a*(xc-x)
    return xr

def expand(x, xc, a=1.9):
    '''Function that performs expansion.'''
    xe = xc + a*(xc-x)
    return xe

def oc(x, xc, a=0.5):
    '''Function that performs outside contraction.'''
    xoc = xc + a*(xc-x)
    return xoc

def ic(x, xc, a=-0.5):
    '''Function that performs inside contraction.'''
    xic = xc + a*(xc-x)
    return xic

def shrink(xn, g=0.5):
    '''Function that shrinks simplex.'''
    x0 = xn[:, 0]
    n = x0.size
    for i in range(n):
        xn[:, i] = x0 + g*(xn[:, i]-x0)
    return xn

def neldermead(f, x0, lb, ub, tol=1e-6):
    '''Implementation of Nelder Mead.'''
    #f.calls = 0
    n = x0.size
    x = np.zeros((n, n+1))
    x[:, 0] = x0
    s = np.zeros(n)
    l = 1
    for i in range(n):
        for j in range(n):
            if j == i:
                s[j] = l/(n*np.sqrt(2))*(np.sqrt(n+1)-1)
            else:
                s[j] = l/(n*np.sqrt(2))*(np.sqrt(n+1)-1)+l/(np.sqrt(2))
        x[:, i+1] = x0+s
    converged = False
    miter = 0
    J = np.zeros(n+1)
    for i in range(n+1):
        J[i] = f(x[:, i].flatten())
    oldhist = np.zeros((n, (n+1)*(miter+1)))
    oldhist[:, (n+1)*miter:(n+1)*(miter+1)] = x
    while not converged:
        tmparr = np.zeros((n+1, n+1))
        tmparr[0, :] = J
        tmparr[1:, :] = x
        tmparr = tmparr[ :, tmparr[0].argsort()]
        x = tmparr[1:, :]
        J = tmparr[0, :]
        sm = np.zeros(n)
        fbar = 0
        for i in range(n):
            sm += x[:, i].flatten()
        xc = sm/n
        for i in range(n+1):
            fbar += J[i]
        fbar = fbar/(n+1)
        df = 0
        for i in range(n+1):
            df += (J[i]-fbar)**2
        df = np.sqrt(df/(n+1))
        if df < tol:
            converged = True
            break
        xr = reflect(x[:, n].flatten(), xc)
        Jxr = f(xr)
        if Jxr < J[0]:
            xe = expand(x[:, n].flatten(), xc)
            Jxe = f(xe)
            if Jxe < J[0]:
                x[:, n] = xe
                J[n] = Jxe
            else:
                x[:, n] = xr
                J[n] = Jxr
        elif Jxr <= J[n-1]:
            x[:, n] = xr
            J[n] = Jxr
        else:
            if Jxr > J[n]:
                xic = ic(x[:, n].flatten(), xc)
                Jic = f(xic)
                if Jic < J[n]:
                    x[:, n] = xic
                    J[n] = Jic
                else:
                    x = shrink(x)
                    for i in range(n+1):
                        J[i] = f(x[:, i].flatten())
            else:
                xoc = oc(x[:, n].flatten(), xc)
                Joc = f(xoc)
                if Joc < Jxr:
                    x[:, n] = xoc
                    J[n] = Joc
                else:
                    x = shrink(x)
                    for i in range(n+1):
                        J[i] = f(x[:, i].flatten())
        miter += 1
        newhist = np.zeros((n, (n+1)*(miter+1)))
        newhist[:, :(n+1)*(miter)] = oldhist
        newhist[:, (n+1)*miter:(n+1)*(miter+1)] = x
        oldhist = newhist
    xst = x[:, 0].flatten()
    #fcalls = f.calls
    output = {'alias': 'MelderNead', \
              'major_iterations': miter, \
              'objective': J[0], 'tol': df,\
              'hist': newhist}
    return xst, J[0], output
            
        
def gradFreeOpt(f, lowerBound=np.zeros(2), upperBound=10*np.ones(2)):
    '''Function that performs gradient free optimization with Nelder Mead.'''
    #f.calls = 0
    n = lowerBound.size
    x0 = np.ones(n)
    if np.any(lowerBound > 1) or np.any(upperBound < 1):
        x0 = 0.5*(lowerBound+upperBound)*np.ones(1)
    xst, J, output = neldermead(f, x0, lowerBound, upperBound)
    return xst, J, output
    
    