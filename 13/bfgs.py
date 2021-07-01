# BFGS Algorithm
# By ZincCat
import numpy as np


def linesearch_Armijo(f, x, grad, d, alpha=0.4, beta=0.8):
    # backtrack linesearch using Armijo rules
    t = 1.0
    value = f(x)
    g = grad(x)
    while f(x + t*d) > value + alpha*t*np.dot(g, d):
        t *= beta
    return t


def linesearch_Wolfe(f, x, grad, d, start=0, end=1e10, rho=0.3, sigma=0.4):
    # linesearch using strong Wolfe rules
    value = f(x)
    g = grad(x)
    reg1 = np.dot(g, d)
    reg2 = sigma*g
    t = 0
    while t < 50:
        alpha = (start + end)/2
        x_new = x + alpha*d
        cond1 = (f(x_new) < value + rho*alpha*reg1)
        cond2 = (np.abs(np.dot(grad(x_new), d))
                 < np.abs(np.dot(reg2, d)))
        if (cond1 and cond2):
            break
        if not cond1:
            end = alpha
        else:
            start = alpha
        t += 1
    return alpha


def updateH(H, dx, dg, eps=1e-30):
    finished = False
    t1 = H@dg
    t2 = np.dot(dg, dx)
    if np.abs(t2) < eps:
        finished = True
    return H + (1+np.dot(dg, t1)/t2)*np.outer(dx, dx)/t2 - (np.outer(t1, dx)+np.outer(dx, t1))/t2, finished


def BFGS(x0, f, grad, n, maxIter=100, eta=1e-10, eps=1e-20, display=True, a=0.4, b=0.8):
    timestep = 0
    x = x0.copy()
    H = np.eye(n)  # H_0
    g = grad(x)
    d = -g
    while True:
        if display:
            print(timestep, "th iteration, f(x)=", f(x))
        if np.linalg.norm(g) < eta:
            break
        alpha = linesearch_Armijo(f, x, g, d, a, b)
        # alpha = linesearch_Wolfe(f, x, grad, d)
        dx = alpha*d
        x += dx
        dg = grad(x) - g
        g += dg
        H, finished = updateH(H, dx, dg, eps)
        if finished:
            break
        d = -H@g
        timestep += 1
        if timestep >= maxIter:
            break
    return x
