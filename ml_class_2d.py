# 리스트 6-2-(1)
import numpy as np
import matplotlib.pyplot as plt

#데이터 생성
np.random.seed(seed=1)
N = 100 # 데이터 수
K = 3 # 분포 수
T3 = np.zeros((N, 3), dtype=np.uint8)
T2 = np.zeros((N,2), dtype=np.uint8)
X = np.zeros((N,2))
X_range0 = [-3, 3]
X_range1 = [-3, 3]
Mu = np.array([[-.5, -.5], [.5, 1.0], [1, -.5]]) # 분포의 중심
Sig = np.array([[.7, .7], [.8, .3], [.3, .8]]) # 분포의 분산
Pi = np.array([0.4, 0.8, 1]) # (A) 각 분포에 대한 비율 0.4 0.8 1
for n in range(N):
    wk = np.random.rand()
    for k in range(K):
        if wk < Pi[k]:
            T3[n, k] = 1
            break
    for k in range(2):
        X[n, k] = (np.random.randn() * Sig[T3[n, :] == 1, k] + Mu[T3[n, :] == 1, k])
T2[:, 0] = T3[:, 0]
T2[:, 1] = T3[:, 1] | T3[:, 2]

print(X[:5, :])

#리스트 6-2-(3)
print(T2[:5, :])


# 리스트 6-2-(4)
print(T3[:5, :])

#리스트 6-2-(5)
#데이터 표시
def show_data2(x, t):
    wk , K = t.shape
    c = [[.5, .5, .5], [1, 1, 1], [0, 0, 0]]
    for k in range(K):
        plt.plot(x[t[:, k] == 1, 0], x[t[:, k] == 1, 1],
                 linestyle='none', markeredgecolor='black',
                 marker='o', color=c[k], alpha=0.8)
    plt.grid(True)

#메인
plt.figure(figsize=(7.5, 3))
plt.subplots_adjust(wspace=0.5)
plt.subplot(1, 2, 1)
show_data2(X, T2)
plt.xlim(X_range0)
plt.ylim(X_range1)

plt.subplot(1, 2, 2)
show_data2(X, T3)
plt.xlim(X_range0)
plt.ylim(X_range1)
plt.show()

# 리스트 6-2-(6)
#로지스틱 회귀 모델
def logistic2(x0, x1, w):
    y = 1 / (1 + np.exp(-(w[0] * x0 + w[1] * x1 + w[2])))
    return y

# 리스트 6-2-(7)
# 모델 3d보기
from mpl_toolkits.mplot3d import axes3d

def show3d_logistic2(ax, w):
    xn = 50
    x0 = np.linspace(X_range0[0], X_range0[1], xn)
    x1 = np.linspace(X_range1[0], X_range1[1], xn)
    xx0, xx1 = np.meshgrid(x0, x1)
    y = logistic2(xx0, xx1, w)
    ax.plot_surface(xx0, xx1, y, color='blue', edgecolor='gray',
                    rstride=5, cstride=5, alpha=0.3)


def show_data2_3d(ax, x, t):
    c = [[.5, .5, .5], [1, 1, 1]]
    for i in range(2):
        ax.plot(x[t[:, i] == 1, 0], x[t[:, i] == 1, 1], 1 - i,
                marker='o', color=c[i], markeredgecolor='black',
                linestyle='none', markersize=5, alpha=0.8)
    Ax.view_init(elev=25, azim=-30)

# test
Ax = plt.subplot(1, 1, 1, projection='3d')
W=[-1, -1, -1]
show3d_logistic2(Ax, W)
show_data2_3d(Ax, X, T2)

# 리스트 6-2-(8)
#모델 등고선 2D 표시
def show_contour_logistic2(w):
    xn = 30 # 매개변수의 분할 수
    x0 = np.linspace(X_range0[0], X_range0[1], xn)
    x1 = np.linspace(X_range1[0], X_range1[1], xn)
    xx0, xx1 = np.meshgrid(x0, x1)
    y = logistic2(xx0, xx1, w)
    cont = plt.contour(xx0, xx1, y, levels=(0.2, 0.5, 0.8), colors=['k', 'cornflowerblue', 'k'])
    cont.clabel(fmt='%1.1f', fontsize=10)
    plt.grid(True)

#test
plt.figure(figsize=(3,3))
W=[-1, -1, -1]
show_contour_logistic2(W)

#리스트 6-2-(9)
#크로스 엔트로피 오차
def cee_logistic2(w, x, t):
    X_n = x.shape[0]
    y = logistic2(x[:, 0], x[:, 1], w)
    cee = 0
    for n in range(len(y)):
        cee = cee - (t[n, 0] * np.log(y[n]) + (1 - t[n, 0] * np.log(1 - y[n])))
    cee = cee / X_n
    return cee

#피스트 6-2-(10)
# 크로스 엔트로피 오차의 미분
def dcee_logistic2(w, x, t):
    X_n = x.shape[0]
    y = logistic2(x[:, 0], x[:, 1], w)
    dcee = np.zeros(3)
    for n in range(len(y)):
        dcee[0] = dcee[0] + (y[n] - t[n, 0]) * x[n,0]
        dcee[1] = dcee[1] + (y[n] - t[n, 0]) * x[n,1]
        dcee[2] = dcee[2] + (y[n] - t[n,0])
    dcee = dcee / X_n
    return dcee

# test
W=[-1, -1, -1]
dcee_logistic2(W, X, T2)

# 리스트 6-2-(11)
from scipy.optimize import minimize

# 로지스틱 회귀 모델의 매개 변수 검색
def fit_logistic2(w_init, x, t):
    res = minimize(cee_logistic2, w_init, args=(x,t),
                   jac=dcee_logistic2, method="CG")
    return res.x

#메인
plt.figure(1, figsize=(7,3))
plt.subplots_adjust(wspace=0.5)

Ax = plt.subplot(1, 2, 1, projection='3d')
W_init = [-1, 0, 0]
W = fit_logistic2(W_init, X, T2)
print("w0 = {0:.2f}, w1 = {1:.2f}, w2= {2:.2f}".format(W[0], W[1], W[2]))
show3d_logistic2(Ax, W)

show_data2_3d(Ax, X, T2)
cee = cee_logistic2(W, X, T2)
print("CEE = {0:.2f}".format(cee))

Ax = plt.subplot(1, 2, 2)
show_data2(X, T2)
show_contour_logistic2(W)

plt.show()

# 리스트 6-2-(12)
# 3클래스용 로지스틱 회귀 모델
def logistic3(x0, x1, w):
    K = 3
    w = w.reshape((3,3))
    n = len(x1)
    y = np.zeros((n, K))
    for k in range(K):
        y[:, k] = np.exp(w[k, 0] * x0 + w[k, 1] * x1 + w[k, 2])
    wk = np.sum(y, axis=1)
    wk = y.T / wk
    y = wk.T
    return y

# test
W = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
y = logistic3(X[:3, 0], X[:3, 1], W)
print(np.round(y, 3))

# 리스트 6-2-(13)
# 교차 엔트로피 오차
def cee_logistic3(w, x, t):
    X_n = x.shape[0]
    y = logistic3(x[:, 0], x[:,1], w)
    cee = 0
    N, K = y.shape
    for n in range(N):
        for k in range(K):
            cee = cee - (t[n,k] * np.log(y[n, k]))
    cee = cee / X_n
    return cee
y
W = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
print(cee_logistic3(W, X, T3))

# 리스트 6-2-(14)
# 교차 엔트로피 오차의 미분
def dcee_logistic3(w, x, t):
    X_n = x.shape[0]
    y = logistic3(x[:,0], x[:,1], w)
    dcee = np.zeros((3, 3))
    N, K = y.shape
    for n in range(N):
        for k in range(K):
            dcee[k, :] = dcee[k, :] - (t[n, k] - y[n, k])*np.r_[x[n, :], 1]
    dcee = dcee / X_n
    return dcee.reshape(-1)

#test
W = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
print(dcee_logistic3(W, X, T3))

# 리스트 6-2-(15)
# 매개변수 검색
def fit_logistic3(w_init, x, t):
    res = minimize(cee_logistic3, w_init, args=(x,t), jac=dcee_logistic3, method="CG")
    return res.x

def show_contour_logistic3(w):
    xn = 30 # 매개변수의 분할 수
    x0 = np.linspace(X_range0[0], X_range0[1], xn)
    x1 = np.linspace(X_range1[0], X_range1[1], xn)

    xx0, xx1 = np.meshgrid(x0, x1)
    y = np.zeros((xn, xn, 3))
    for i in range(xn):
        wk = logistic3(xx0[:, i], xx1[:, i], w)
        for j in range(3):
            y[:, i, j] = wk[:, j]
    for j in range(3):
        cont = plt.contour(xx0, xx1, y[:, :, j], levels = (0.5, 0.9), colors=['cornflowerblue', 'k'])
        cont.clabel(fmt='%1.1f', fontsize=9)
    plt.grid(True)

# 리스트 6-2-(17)
# 메인
W_init = np.zeros((3,3))
W = fit_logistic3(W_init, X, T3)
print(np.round(W.reshape((3,3)), 2))
cee = cee_logistic3(W, X, T3)
print("CEE = {0:.2f}".format(cee))

plt.figure(figsize=(3, 3))
show_data2(X, T3)
show_contour_logistic3(W)
plt.show()