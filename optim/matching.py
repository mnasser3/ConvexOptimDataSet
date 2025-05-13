import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import pandas as pd

X_train_full = np.load('ConvexOptimDataSet/optim/X_train.npy')
X_val_full = np.load('ConvexOptimDataSet/optim/X_val.npy')
Y_train_full = np.load('ConvexOptimDataSet/optim/Y_train.npy')
Y_val_full = np.load('ConvexOptimDataSet/optim/Y_val.npy')


size = 300
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

def f(w,X, mu,beta_cov, Sigma, alpha, tau, s):
    n = X.shape[0]
    g = np.linalg.norm(X.T@w - mu)**2 + beta_cov * np.linalg.norm((X - np.ones((n,1))*mu.T).T@np.diag(w)@(X - np.ones((n,1))*mu.T) - Sigma, 'fro')**2 - alpha*s.dot(w)
    return g + tau*np.linalg.norm(w,1)

def proj_simplex(v):
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(u)+1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1)
    return np.maximum(v - theta, 0)

def proj_class_balance(w0, labels, p_c, delta):
    w0 = np.asarray(w0)
    labels = np.asarray(labels)
    s = np.sum(w0)
    classes, inverse = np.unique(labels, return_inverse=True)
    C = len(classes)
    m = np.bincount(inverse, weights=w0, minlength=C)
    lower = (p_c - delta) * s
    upper = (p_c + delta) * s
    m_proj = np.clip(m, lower, upper)
    scale = np.divide(m_proj, m, out=np.zeros_like(m_proj), where=m>0)
    return w0 * scale[inverse]


def project_C(v, labels, p_c, delta, max_iter=50, tol=1e-4):
    w = proj_simplex(v)
    for _ in range(max_iter):
        w_old = w.copy()
        w = proj_class_balance(w, labels, p_c, delta)
        w = proj_simplex(w)
        if np.linalg.norm(w - w_old) < tol:
            break
    return w

def grad_g(w, X, mu, Sigma, alpha, beta_cov, s):
    M = X - np.ones((X.shape[0],1)) * mu[np.newaxis,:]
    g1 = 2 * X @ (X.T @ w - mu)
    C = M.T @ (w[:,None]*M) - Sigma
    g2 = 2*beta_cov * np.einsum('ij,jk,ik->i', M, C, M)
    return g1 + g2 - alpha * s

def prox_l1_projC(v, labels, p_c, delta, tau, step):
    z = np.sign(v) * np.maximum(np.abs(v) - step*tau, 0)
    return project_C(z, labels, p_c, delta)

def fista(X, mu, Sigma, s, alpha, beta_cov, tau, labels, p_c, delta, step=10, max_iter=1000):
    n = X.shape[0]
    w = np.ones(n) / n
    z = w.copy()
    q = 1.0
    step=step
    
    for _ in range(max_iter):
        g = grad_g(z, X, mu, Sigma, alpha, beta_cov, s)
        fz = f(z, X, mu, beta_cov, Sigma, alpha, tau, s)
        while True:
            w_temp = prox_l1_projC(z - step*g, labels, p_c, delta, tau, step)
            fnext = f(w_temp, X, mu, beta_cov, Sigma, alpha, tau, s)
            if fnext <= fz + g.dot(w_temp - z) + 1/(2*step)* np.linalg.norm(w_temp - z)**2:
                break
            step *= 0.5
        
        q_next = (1 + np.sqrt(1 + 4 * q**2))/ 2
        z = w_temp + (q - 1)/q_next*(w_temp - w)
        w, q = w_temp, q_next
        step = step / 0.5
        if np.abs(fnext - fz) < 1e-4:
            break
    return w

s = np.zeros((X_train.shape[0],))
beta = 1
lambd = 0.1
w_estimated = fista(X_train, target_mu, target_S, s, 0, beta, lambd, Y_train, labels_frac, 0.1)
wki = sorted(w_estimated, reverse=True)
print("PGD\n")
print("top 15 estimated w:", wki[:5])
print(sum(wki))
print("min w: ", min(wki))
print("max w: ", max(wki))
print("obj value: ",f(w_estimated, X_train, target_mu, beta, target_S, 0, lambd, s))


print()
print("CVXPY\n")
w = cp.Variable(n,nonneg=True)
constraints = [cp.sum(w) == 1]
mean_match = cp.sum_squares(X_train.T @ w - target_mu)
cov_match = cp.norm(Xc_train.T @ cp.diag(w) @ Xc_train - target_S, 'fro')**2
for i in range(len((unique_labels))):
    mask = (Y_train == unique_labels[i]).astype(int)
    mask = np.reshape(mask, (n, 1))
    constraints += [mask.T @w >= (labels_frac[i]-lambd)*cp.sum(w)]
    constraints += [mask.T @w <= (labels_frac[i]+lambd)*cp.sum(w)]
objective = mean_match + beta*cov_match
prob = cp.Problem(cp.Minimize(objective), constraints)
prob.solve()
wi = sorted(w.value, reverse=True)
print("Status:", prob.status)
print("Optimal value:", prob.value)
print("top 15 Optimal w:", wi[:5])