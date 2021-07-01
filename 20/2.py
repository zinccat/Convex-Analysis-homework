# pLADMPSAP
# By ZincCat

from scipy.optimize import minimize
import numpy as np
from matplotlib import pyplot as plt
# from joblib import Parallel, delayed

np.random.seed(19890817)

n = 70
s = 30
x = np.random.normal(0, 1, (n, s))
y = np.random.choice([0, 1], s)
w0 = np.random.normal(0, 1, n)


def f(w):
    return np.sum(np.log(1+np.exp(-y*np.dot(w, x))))/s


def gradient_f(w):
    temp = np.exp(-y*np.dot(w, x))
    return np.sum(-temp*y/(1+temp)*x, axis=1)/s


def descent(w, grad, value, mode='2', alpha=0.4, beta=0.8, eta=1e-7):
    # 梯度下降函数
    # 输入目前x取值, 梯度, 梯度的范数, 下降模式
    # 输出下降后x取值, 步长t
    # 下降模式为'2'时采用2范数, 为'inf'时采用无穷范数
    g = grad(w)
    grad_norm = np.linalg.norm(g)
    if grad_norm <= eta:
        return w, True
    normalized_grad = g/grad_norm
    t = 1.0
    if mode == '2':
        # l_2 norm
        while f(w - t*normalized_grad) > value - alpha*t*np.dot(g, normalized_grad):
            t *= beta
        w -= t*normalized_grad
    elif mode == 'inf':
        # l_infty norm
        while f(w - t*np.sign(normalized_grad)) > value - alpha*t*np.dot(g, np.sign(normalized_grad)):
            t *= beta
        w -= t*np.sign(normalized_grad)
    return w, False


def gd(w0, eta=1e-5, maxIter=1000):
    w = w0.copy()
    timestep = 0
    while timestep <= maxIter:
        value = f(w)
        print("Iteration:", timestep, "Error", value)
        w, finished = descent(w, gradient_f, value,
                              mode='2', eta=eta)  # 此时使用2范数
        if finished:
            break
        timestep += 1
    return w


def grad(i, w):
    temp = np.exp(-y[i]*np.dot(w, x[:, i]))
    return -temp*y[i]/(1+temp)*x[:, i]/s


def pLADMAPSAP(w0, beta, eps1=1e-7, eps2=1e-5, maxBeta=1e1, maxIter=1e7, rho0=1.9):
    w = w0.copy()
    W = np.random.rand(s, n)
    newW = np.zeros_like(W)
    L = np.zeros_like(W)
    dL = np.zeros_like(W)
    Li = np.linalg.norm(x, axis=1)/4/s
    eta = s*np.ones_like(Li)
    tau = Li + beta*eta
    timestep = 0
    values = []
    while timestep <= maxIter:
        if timestep % 1000 == 0:
            print(timestep, f(w))
        values.append(f(w))
        # naive multithreading, however, too slow when matrices are small
        # Parallel(n_jobs=-1, backend='threading')(delayed(update)(i, W, w, Lam, beta)
        #                                         for i in range(s))
        # sequential update
        for i in range(s):
            newW[i] = w - L[i]/tau - grad(i, W[i])/tau
            dL[i] = W[i]-w
        w = (np.sum(W, axis=0)+np.sum(L, axis=0)/tau)/s
        L += tau*dL
        crit = np.linalg.norm(dL) < eps1
        W = newW
        # if beta*np.max(np.sqrt(n)*dW/np.linalg.norm(w)) < eps2:
        #     rho = rho0
        #     crit2 = True
        # else:
        #     rho = 1
        #     crit2 = False
        # beta = min(maxBeta, beta*rho)
        # tau = Li + beta*eta
        if crit:  # and crit2:
            print("Finished!!!")
            print(timestep, f(w))
            break
        timestep += 1
    return w, values


w3, values = pLADMAPSAP(w0, 0.001)
print(f(w3))
plt.plot(values)
plt.xlabel("Value")
plt.ylabel("Steps")
plt.savefig("pLADMPSAP")

w = gd(w0)
print(f(w))

"""
w2 = minimize(f, w0, jac=gradient_f)
print(w2.fun)
"""
