"""
Author: Hiren Saravana
Implementation of SQP constrained optimizer for use with drag model.
"""

from drag import drag_obj
from conlongbar import g, g_der
import numpy as np
import algopy

def L(X, s):
    '''Lagrangian.'''
    A = X[0]
    S = X[1]
    xin = np.array([A, S])
    lg = drag_obj(xin) + s*g(xin)
    return lg

def L_der(x, s):
    '''Gradient of Lagrangian.'''
    x = algopy.UTPM.init_jacobian(x)
    return algopy.UTPM.extract_jacobian(L(x, s))

def L_hess(x, s):
    '''Hessian of Lagrangian.'''
    x = algopy.UTPM.init_hessian(x)
    return algopy.UTPM.extract_hessian(x.size, L(x, s))

def sqp(X0, tol):
    '''Constrained optimizer using SQP.'''
    x = np.array([X0[0], X0[1]])
    l = X0[2]
    converged = False
    miter = 1
    while not converged:
        A = np.zeros((3, 3))
        A[0:2, 0:2] = L_hess(x, l)
        A[2, 0:2] = g_der(x)
        A[0:2, 2] = g_der(x)
        tmpLder = L_der(x, l)
        TAU = np.amax(np.abs(tmpLder))
        if TAU < tol:
            converged = True
        ng = -g(x)
        B = np.zeros((3, 1))
        B[0:2, 0] = -tmpLder
        B[2, 0] = ng
        s = np.linalg.solve(A, B)
        x = x+np.array([s[0,0], s[1,0]])
        l = l+s[2,0]
        miter = miter + 1
    J = drag_obj(x)
    return x, J, miter