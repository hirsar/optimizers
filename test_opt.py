import numpy as np
from gradFreeOpt import gradFreeOpt

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
    global FUNCCALLS
    FUNCCALLS += 1
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
    return D

def rosenbrock_val(x):
    """Implementation of N-D Rosenbrock function.
    
    Args:
        x: input vector (x1, x2, ..., xN)
    
    Returns:
        J: value of objective function
        g: gradient vector
    """
    global FUNCCALLS
    FUNCCALLS += 1
    n = len(x)
    J = 0
    for j in np.arange(0, n-1):
        J = J + 100*(x[j+1] - x[j]**2)**2 + (1 - x[j])**2
    return J


def paraboloid(x):
    """
        Example function to show how
        the functions used to test your
        optimizer with be implemented


    Parameters
    ----------
    x : ndarray
        point in 2D space (but

    Outputs
    -------
    f : float
        function value
    g : ndarray
        gradient of the function with respect to each of the design variables
    """
    global FUNCCALLS
    FUNCCALLS += 1

    funcValue = x[0]**2 + x[1]**2

    return funcValue


testFuncs = [paraboloid, rosenbrock_val, drag_obj]

fopt_global = [0.0, 0.0, 191.53595751019435]
funcEval_best = 100.0 #just a made up number in this for the example

score = 0

# Switch gradFreeOpt with uncon to compare

for ii, func in enumerate(testFuncs):
    FUNCCALLS = 0
    xopt, fopt, output = gradFreeOpt(func, lowerBound=np.ones(2)*-5,
                                     upperBound=np.ones(2)*5)

    err = abs(fopt_global[ii] - fopt) / (abs(fopt_global[ii]) + 1.0)
    print(FUNCCALLS)
    print(fopt)
    score += funcEval_best/FUNCCALLS + max(1 - err, 0)

print(output['alias'], score)
