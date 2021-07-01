# Block Coordinate Descent
# By ZincCat

import numpy as np
from matplotlib import pyplot as plt

r = 5
m = 200
n = 300

np.random.seed(1)

U0 = np.random.normal(0, 1, (m, r))
VT0 = np.random.normal(0, 1, (r, n))
D = U0@VT0
c = 0.1  # mask constant
mask = np.random.uniform(0, 1, (m, n)) < c


def bcd(D, maxIter, eps):
    U = np.random.normal(0, 1, (m, r))
    VT = np.random.normal(0, 1, (r, n))
    A = U@VT
    A[mask] = D[mask]
    value = np.linalg.norm(U@VT - A)**2/2
    oldvalue = 0
    count = 0
    for i in range(maxIter):
        value = np.linalg.norm(U@VT - A)**2/2
        if abs(value - oldvalue) < eps:
            count += 1
            if count >= 10:
                print("Converged at step", i, "Value:", value)
                break
        else:
            count = 0
        if i % 100 == 0:
            print("Step:", i, "Value:", value)
        U = A@np.linalg.pinv(VT)
        VT = np.linalg.pinv(U)@A
        A = U@VT
        A[mask] = D[mask]
        oldvalue = value
    return U, VT, A


U, VT, A = bcd(D, 100000, 1e-20)
print(np.linalg.norm(A-D))
