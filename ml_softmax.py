import numpy as np
import matplotlib.pyplot as plt


def softmax(x0, x1, x2):
    u = np.exp(x0) + np.exp(x1) +np.exp(x2)
    return np.exp(x0) / u, np.exp(x1) / u, np.exp(x2) /u
## softmax test
y=softmax(2, 1, -1)
print(np.round(y,2))
print(np.sum(y))



from mpl_toolkits.mplot3d import Axes3D

xn = 20
x0 = np.linspace(-4, 4, xn)
x1 = np.linspace(-4, 4, xn)
y = np.zeros((xn, xn, 3))
for i0 in range(xn):
    for i1 in range(xn):
        y[i1, i0, :] = softmax(x0[i0], x1[i1], 1)

xx0, xx1 = np.meshgrid(x0, x1)
plt.figure(figsize=(8,3))
for i in range(2):
    ax = plt.subplot(1, 2, i+1, projection='3d')
    ax.plot_surface(xx0, xx1, y[:, :, i],
                   rstride=1, cstride=1, alpha=0.3, color='blue', edgecolor='black')
    ax.set_xlabel('$x_0$', fontsize=14)
    ax.set_ylabel('$x_1$', fontsize=14)
    ax.view_init(40, -125)

plt.show()