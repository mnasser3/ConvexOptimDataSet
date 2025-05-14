import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import pandas as pd
import time
from scipy.optimize import root


X_train_full = np.load('ConvexOptimDataSet/optim/X_train.npy')
X_val_full = np.load('ConvexOptimDataSet/optim/X_val.npy')
Y_train_full = np.load('ConvexOptimDataSet/optim/Y_train.npy')
Y_val_full = np.load('ConvexOptimDataSet/optim/Y_val.npy')


size = -1
X_train = X_train_full[:size]
Y_train = Y_train_full[:size]
X_val = X_val_full[:size]
Y_val = Y_val_full[:size]
print(X_train.shape, X_val.shape)
print(Y_train.shape, Y_val.shape)

n = X_train.shape[0]
XV_mean = np.mean(X_val, axis=0)
XT_mean = np.mean(X_train, axis=0)
Xc_train = X_train - XV_mean
Xc_Val = X_val - XV_mean

unique_labels = np.sort(np.unique(Y_val).astype(int))
labels_frac = np.array([(Y_val == unique_labels[i]).sum() for i in range(len(unique_labels))])/len(Y_val)
print(labels_frac)

target_mu = XV_mean
target_S = Xc_Val.T @ Xc_Val / len(X_val)

def f(w, X, Xc, mu, beta_cov, Sigma, alpha, tau, s):
    r = X.T.dot(w)
    diff = r - mu
    g1 = diff.dot(diff)

    Cw = Xc.T.dot(Xc * w[:, None])
    E = Cw - Sigma
    g2 = beta_cov * np.sum(E * E)

    g3 = -alpha * (s.dot(w))
    g4 = tau * np.sum(np.abs(w))

    return g1 + g2 + g3 + g4

def project_C(v, labels, p_c, delta):
    v = np.asarray(v, dtype=float)
    labels = np.asarray(labels)
    classes, inv = np.unique(labels, return_inverse=True)
    C = len(classes)

    p_minus = np.array([p_c[c] - delta for c in classes])
    p_plus  = np.array([p_c[c] + delta for c in classes])

    def residuals(duals):
        lam    = duals[0]
        a_minus = duals[1:1+C]
        a_plus  = duals[1+C:1+2*C]

        shift = v - lam + a_minus[inv] - a_plus[inv]
        w = np.clip(shift, 0.0, 1.0)

        res = [w.sum() - 1.0]

        for idx, c in enumerate(classes):
            mask = (labels == c)
            s_c = w[mask].sum()
            res.append(s_c - p_minus[idx])
            res.append(s_c - p_plus[idx])

        return np.array(res)

    x0 = np.zeros(1 + 2*C)
    sol = root(residuals, x0, method='hybr')
    lam    = sol.x[0]
    a_minus = sol.x[1:1+C]
    a_plus  = sol.x[1+C:1+2*C]

    shift = v - lam + a_minus[inv] - a_plus[inv]
    w = np.clip(shift, 0.0, 1.0)
    return w

def grad_g(y, X, mu, Sigma, alpha, beta_cov, s,M):
    r=X.T.dot(y)-mu
    g1=2*X.dot(r)
    C=M.T.dot(M*y[:,None])-Sigma
    tmp=M.dot(C)
    g2=2*beta_cov*np.sum(M*tmp,axis=1)
    return g1+g2-alpha*s

def prox_l1_projC(v, labels, p_c, delta, tau, step):
    z = np.sign(v) * np.maximum(np.abs(v) - step*tau, 0)
    return project_C(z, labels, p_c, delta)

def fista(X, mu, Sigma, s, alpha, beta_cov, tau, labels, p_c, delta, step=10, max_iter=1000):
    n = X.shape[0]
    w = np.ones(n) / n
    z = w.copy()
    q = 1.0
    step=step
    M  = X - mu  
    for _ in range(max_iter):
        g = grad_g(z, X, mu, Sigma, alpha, beta_cov, s,M)
        fz = f(z, X, M, mu, beta_cov, Sigma, alpha, tau, s)
        while True:
            w_temp = prox_l1_projC(z - step*g, labels, p_c, delta, tau, step)
            fnext = f(w_temp, X, M, mu, beta_cov, Sigma, alpha, tau, s)
            if fnext <= fz + g.dot(w_temp - z) + 1/(2*step)* np.linalg.norm(w_temp - z)**2:
                break
            step *= 0.5
        
        q_next = (1 + np.sqrt(1 + 4 * q**2))/ 2
        z = w_temp + (q - 1)/q_next*(w_temp - w)
        w, q = w_temp, q_next
        step = step / 0.5
        diff = fnext - fz
        if np.abs(diff) < 1e-4:
            break
    return w

s = np.zeros((X_train.shape[0],))
beta = 1
lambd = 0.1

start_time = time.time()
w_estimated = fista(X_train, target_mu, target_S, s, 0, beta, lambd, Y_train, labels_frac, 0.1)
end_time = time.time()

wki = sorted(w_estimated, reverse=True)
print("PGD\n")
print("top 15 estimated w:", wki[:5])
print(sum(wki))
print("min w: ", min(wki))
print("max w: ", max(wki))
start_obj_time = time.time()
print("obj value: ", f(w_estimated, X_train, X_train - target_mu, target_mu, beta, target_S, 0, lambd, s))
end_obj_time = time.time()
print("Objective computation time: {:.4f} seconds".format(end_obj_time - start_obj_time))
print("Execution time: {:.4f} seconds".format(end_time - start_time))


# print()
# print("CVXPY\n")
# w = cp.Variable(n,nonneg=True)
# constraints = [cp.sum(w) == 1]
# mean_match = cp.sum_squares(X_train.T @ w - target_mu)
# cov_match = cp.norm(Xc_train.T @ cp.diag(w) @ Xc_train - target_S, 'fro')**2
# for i in range(len((unique_labels))):
#     mask = (Y_train == unique_labels[i]).astype(int)
#     mask = np.reshape(mask, (n, 1))
#     constraints += [mask.T @w >= (labels_frac[i]-lambd)*cp.sum(w)]
#     constraints += [mask.T @w <= (labels_frac[i]+lambd)*cp.sum(w)]
# objective = mean_match + beta*cov_match
# prob = cp.Problem(cp.Minimize(objective), constraints)
# prob.solve()
# wi = sorted(w.value, reverse=True)
# print("Status:", prob.status)
# print("Optimal value:", prob.value)
# print("top 15 Optimal w:", wi[:5])