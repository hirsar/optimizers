"""
Author: Hiren Saravana
Alias: N-dimensional_Space_Alien
Implementation of unconstrained gradient-based optimizer.
"""

import numpy as np

def pinpoint(f, J0, g0, x0, mu1, mu2, \
             alow, Jalow, galow, ahigh, Jahigh, gahigh, p):
    """Implementation of pinpointing algorithm."""
    j = 0
    Jprime0 = g0.T@p
    Jprimealow = galow.T@p
    amin = (2*alow*(Jahigh-Jalow)+Jprimealow*(alow**2-ahigh**2))/\
           (2*(Jahigh-Jalow+Jprimealow*(alow-ahigh)))
    Jamin, gamin = f(x0+amin*p)
    while True:
        Jprimeamin = gamin.T@p
        if (Jamin > J0+mu1*amin*Jprime0) or (Jamin > Jalow):
            ahigh = amin
            Jahigh = Jamin
            amin = (2*alow*(Jahigh-Jalow)+Jprimealow*(alow**2-ahigh**2))/\
                    (2*(Jahigh-Jalow+Jprimealow*(alow-ahigh)))
            Jamin, gamin = f(x0+amin*p)
        else:
            if (np.abs(Jprimeamin) <= np.abs(mu2*Jprime0)): 
                astar = amin
                return Jamin, gamin, astar
            elif (Jprimeamin*(ahigh-alow) >= 0):
                ahigh = alow
            alow = amin
            Jalow = Jamin
            galow = gamin
            Jprimealow = galow.T@p
            amin = (2*alow*(Jahigh-Jalow)+Jprimealow*(alow**2-ahigh**2))/\
                    (2*(Jahigh-Jalow+Jprimealow*(alow-ahigh)))
            Jamin, gamin = f(x0+amin*p)
        j += 1

def bracketing(f, x0, J0, g0, a, amax, mu1, mu2, p):
    """Implementation of bracketing algorithm."""
    a0 = 0
    a1 = a
    Ja0 = J0
    ga0 = g0
    Jprime = g0.T@p
    i = 0
    Ja1, ga1 = f(x0+a1*p)
    while True:
        if (Ja1 > (J0 + mu1*a1*Jprime)) or (Ja1 > Ja0 and i > 1):
            Ja, ga, astar = pinpoint(f, J0, g0, x0, mu1, mu2, \
                                     a0, Ja0, ga0, a1, Ja1, ga1, p)
            return Ja, ga, astar
        Jprimea1 = ga1.T@p
        if (np.abs(Jprimea1) <= np.abs(mu2*Jprime)):
            astar = a1
            return Ja1, ga1, astar
        elif (Jprimea1 >= 0):
            Ja, ga, astar = pinpoint(f, J0, g0, x0, mu1, mu2, \
                                     a1, Ja1, ga1, a0, Ja0, ga0, p)
            return Ja, ga, astar
        else:
            a0 = a1
            Ja0 = Ja1
            a1 = 1.2*a1
            Ja1, ga1 = f(x0+a1*p)
        i += 1
        
def linesearch(f, x0, J0, g0, a, amax, mu1, mu2, p):
    """Implementation of line search using pinpointing and bracketing to
    determine step size.
    """
    J, g, a = bracketing(f, x0, J0, g0, a, amax, mu1, mu2, p)
    x = x0+a*p
    return x, J, g, a

def backtrack(f, x0, J, g, alpha, rho, mu, p):
    """Implementation of backtracking algorithm for determining step size."""
    phi_prime = np.matmul(np.transpose(g), p)
    phi0 = J
    (phi_alpha, g_alpha) = f(x0+alpha*p)
    while phi_alpha > phi0+mu*alpha*phi_prime:
        alpha = rho*alpha
        (phi_alpha, g_alpha) = f(x0+alpha*p)
    x = x0+alpha*p
    return x, phi_alpha, g_alpha, alpha

def quasinewton(x, V, alpha, p, gprev, gnew):
    """Implementation of quasinewton method for determining search 
    direction.
    """
    broken = False
    p = np.array(p)[np.newaxis]
    gnew = np.array(gnew)[np.newaxis]
    gprev = np.array(gprev)[np.newaxis]
    gk = gprev.T
    gk1 = gnew.T
    yk = gk1 - gk
    sk = alpha*p.T
    I = np.eye(len(x))
    if sk.T@yk == 0:
        broken = True
        p = p.flatten()
        return p, V, broken
    rhok = (1/(sk.T@yk)).flatten()
    v = I - (rhok*yk@sk.T)
    Vnew = (v.T@V@v) + (rhok*sk@sk.T)
    pnew = -Vnew@gk1
    pnew = pnew.flatten()
    return pnew, Vnew, broken


def conjgradient(x, p, gprev, gnew):
    """Implementation of conjugate gradient method for determining search
    direction."""
    gnew = np.array(gnew)[np.newaxis]
    gprev = np.array(gprev)[np.newaxis]
    gnew = gnew.T
    gprev = gprev.T
    beta = (gnew.T)@gnew/((gprev.T)@gprev)
    gnew = gnew.flatten()
    beta = beta.flatten()
    p = -gnew + beta*p
    return p
    
def uncon(f, x0, tol, mu1=1e-6, mu2=0.99996, adef=1, amax=200, method='QN', \
          search='BT'):
    """Implementation of unconstrained gradient-based optimizer.
    
    Args:
        f: objective function
        x0: starting point
        tol: convergence tolerance
        mu1: sufficient decrease parameter (optional)
        mu2: curvature parameter (optional)
        adef: default step size (optional)
        amax: maximum step size (optional)
        method: search direction method (optional)
        search: step size method (optional)
    
    Returns:
        x: optimal point as found by optimizer
        fopt: objective function at optimal point
        output: dictionary of relevant information
    """
    f.calls = 0
    converged = False
    J, g = f(x0)
    p = -g.T
    gprev = g
    gnew = gprev
    x = x0
    miter = 0
    V = np.identity(len(x))
    switch = False
    n = x0.size
    oldhist = np.zeros((n, miter+1))
    oldhist[:, miter] = x0
    while not converged:
        if search == 'SW':
            x, J, gnew, a = linesearch(f, x, J, gnew, \
                                       adef, amax, mu1, mu2, p)
        elif search == 'BT':
            x, J, gnew, a = backtrack(f, x, J, gnew, adef, 0.8, 1e-6, p)
        if method == 'QN':
            p, V, broken = quasinewton(x, V, a, p, gprev, gnew)
            if broken:
                method = 'CG'
                switch = True
            elif miter > 200:
                method = 'CG'
        elif method == 'CG':
            p = conjgradient(x, p, gprev, gnew)
            if switch is True:
                switch = False
                method = 'QN'
        elif method == 'SD':
            p = -g.T
        TAU = np.amax(np.abs(gnew))
        if TAU < tol:
            converged = True
        miter += 1
        newhist = np.zeros((n, miter+1))
        newhist[:, :miter] = oldhist
        newhist[:, miter] = x
        oldhist = newhist
        gprev = gnew
    output = {'alias': 'N-dimensional_Space_Alien', \
              'major_iterations': miter, \
              'objective': J, 'hess_inv': V, 'g-norm': TAU, \
              'func_calls': f.calls, 'hist': newhist}
    return x, J, output