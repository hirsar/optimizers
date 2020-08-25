import numpy as np

def rosenbrock(x):
    """Implementation of N-D Rosenbrock function.
    
    Args:
        x: input vector (x1, x2, ..., xN)
    
    Returns:
        J: value of objective function
        g: gradient vector
    """
    n = len(x)
    J = 0
    g = np.zeros(n)
    for j in np.arange(0, n-1):
        J = J + 100*(x[j+1] - x[j]**2)**2 + (1 - x[j])**2
        if j == 0:
            g[j] = -200*(x[j+1] - x[j]**2)*2*x[j] - 2*(1-x[j])
        else:
            g[j] = 200*(x[j] - x[j-1]**2) -\
                   200*(x[j+1] - x[j]**2)*2*x[j] - 2*(1 - x[j])
        if j == n - 2:
            g[n-1] = 200*(x[n-1] - x[n-2]**2)
    rosenbrock.calls += 1
    return J, g

def rosenbrock_val(x):
    '''Function that returns value of Rosenbrock function.'''
    rosenbrock.calls = 0
    J, g = rosenbrock(x)
    rosenbrock_val.calls += 1
    return J

def rosenbrock_der(x):
    '''Function that returns exact gradient of Rosenbrock function.'''
    rosenbrock.calls = 0
    J, g = rosenbrock(x)
    return g

def rosen_der(x):
    '''Function that returns finite difference gradient of
       Rosenbrock funciton.'''
    rosenbrock.calls = 0
    n = x.size
    xtmp = np.zeros(n)
    for i in range(n):
        xtmp[i] = x[i]
    J, g = rosenbrock(x)
    h = 1e-8
    rder = np.zeros(n)
    for i in range(n):
        xtmp[i] += h
        Jh, gh = rosenbrock(xtmp)
        rder[i] = (Jh-J)/h
        for i in range(n):
            xtmp[i] = x[i]
    return rder

def rosen_FD(x):
    '''Rosenbrock function with finite difference gradient.'''
    rosenbrock.calls = 0
    rosenbrock_val.calls = 0
    J, g = rosenbrock(x)
    rder = rosen_der(x)
    rosen_FD.calls += 1
    return J, rder