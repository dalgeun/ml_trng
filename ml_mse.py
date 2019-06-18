import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#평균 오차 함수
def mse_line(x, t, w)
    y = w[0] * x + w[1]
    mse = np.mean((y - t)**2)
    return mse

#계산
xn = 100
w0_range = [-25, 25]
w1_range = [120, 170]
x0 = np.linspace(w0_range[0], w0_range[1], xn)
x1 = np.linspace(w1_range[0], w1_range[1], xn)
xx0, xx1 = np.meshgrid(x0, x1)
J = np.zeros((len(x0), len(x1)))

for i0 in range(xn)
    for i1 in range(xn)
        J[i0, i1]) = mse_line(X, T, (x0[i0], x1[i1]))
