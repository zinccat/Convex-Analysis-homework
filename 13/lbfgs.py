# Low Memory BFGS Algorithm
# By ZincCat
import numpy as np


def linesearch(f, x, g, d, a=0.4, b=0.8):
    # backtrack linesearch
    t = 0.8
    value = f(x)
    while f(x + t*d) > value + a*t*np.dot(g, d):
        t *= b
    return t


def getd(grad, m):
    q = grad.copy()
    alphalist = []
    for i, xg in enumerate(m):
        alphalist.append(np.dot(xg[0], q)/np.dot(xg[0], xg[1]))
        q -= alphalist[i]*xg[1]
    l = len(m)
    if l > 0:  # H_0
        q *= np.dot(m[0][0], m[0][1])/np.dot(m[0][1], m[0][1])  # p
    for i, xg in enumerate(reversed(m)):
        beta = np.dot(xg[1], q)/np.dot(xg[0], xg[1])
        q += (alphalist[l-i-1] - beta)*xg[0]
    return -q


def LBFGS(x0, f, grad, memlimit=3, maxIter=100, eta=1e-10, display=True, a=0.4, b=0.8):
    timestep = 0
    x = x0
    g = grad(x)
    d = -g
    m = []
    while True:
        if display:
            print(timestep, "th iteration, f(x)=", f(x))
        if np.linalg.norm(g) < eta:
            break
        d = getd(g, m)
        alpha = linesearch(f, x, g, d, a, b)
        dx = alpha*d
        x += dx
        dg = grad(x) - g
        g += dg
        if len(m) >= memlimit:
            m.pop()
        m.insert(0, (dx, dg))
        timestep += 1
        if timestep >= maxIter:
            break
    return x
