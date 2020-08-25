"""
Author: Hiren Saravana
Implementation of drag function, weight estimation, etc.
"""

import numpy as np
import algopy

def w(x):
    '''Weight estimation code, which uses simple Newton-Raphson method to 
    estimate the weight. Since the quadratic has two roots, the upper one
    is the one that is correct since the lower one is less than W0 and
    can also be negative.'''
    W0 = 4940
    A = x[0]
    S = x[1]
    b = np.sqrt(A*S)
    Nult = 2.5
    t_c = 0.12
    qB = (-8.71e-5)*Nult*b**3*np.sqrt(W0)/(S*t_c)
    qC = -W0 - 45.42*S
    qA = 1
    X = lambda x : qA*x**2+qB*x+qC
    xguess = np.sqrt(10*W0)
    for i in range(50):
        step = -1/(2*qA*xguess+qB)*X(xguess)
        xguess = xguess+step
    x = xguess
    W = x**2
    return W

def drag_obj(x):
    '''Function for drag calculation that includes laminar-to-turbulence
    transition.'''
    A = x[0]
    S = x[1]
    b = np.sqrt(A*S)
    c = S/b
    rho = 1.23
    mu = 17.8e-6
    V = 35
    Swet = 2.05*S
    k = 1.2
    e = 0.96
    Re = rho*V*c/mu
    if c <= 0.85:
        Cf = 1.328/np.sqrt(Re)
    else:
        Cf = 0.074/(Re**0.2)
    W = w(x)
    CL = W/(0.5*rho*(V**2)*S)
    CD = 0.03062702/S + k*Cf*Swet/S + (CL**2)/(np.pi*A*e)
    D = 0.5*rho*(V**2)*S*CD
    drag_obj.calls += 1
    return D

def drag_der(x):
    '''Gradient of drag function using algorithmic differentiation.'''
    x = algopy.UTPM.init_jacobian(x)
    return algopy.UTPM.extract_jacobian(drag_obj(x))

def drag(x):
    '''Function that returns drag and gradient for use within optimizer.'''
    J = drag_obj(x)
    g = drag_der(x)
    drag.calls += 1
    return J, g