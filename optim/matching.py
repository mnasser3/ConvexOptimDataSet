import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import pandas as pd

X_train = np.load('ConvexOptimDataSet/optim/X_train.npy')
X_val = np.load('ConvexOptimDataSet/optim/X_val.npy')
Y_train = np.load('ConvexOptimDataSet/optim/Y_train.npy')
Y_val = np.load('ConvexOptimDataSet/optim/Y_val.npy')


print(X_train.shape, X_val.shape)
print(Y_train.shape, Y_val.shape)

n = X_train.shape[0]
XV_mean = np.mean(X_val, axis=0)
XT_mean = np.mean(X_train, axis=0)
Xc_train = X_train - XV_mean
Xc_Val = X_val - XV_mean

unique_labels = np.sort(np.unique(Y_val).astype(int))
labels_frac = np.array([(Y_val == unique_labels[i]).sum() for i in range(len(unique_labels))])/len(Y_val)

target_mu = XV_mean
target_S = Xc_Val.T @ Xc_Val / len(X_val)

alpha = 1
beta = 1
lambd = 0.1
w = cp.Variable(n, nonneg=True)
W = cp.diag(w)


goal_mu = (X_train.T @ w)
goal_cov = Xc_train.T @ W @ Xc_train
mean_match = cp.sum_squares(goal_mu - target_mu)
cov_match  = cp.norm(goal_cov - target_S, 'fro')**2

constraints = [cp.sum(w) == 1]
constraints += [w >= 0]
constraints += [w <= 1]
for i in range(len((unique_labels))):
    mask = (Y_train == unique_labels[i]).astype(int)
    mask = np.reshape(mask, (n, 1))
    constraints += [mask.T @w >= (labels_frac[i]-lambd)*cp.sum(w)]
    constraints += [mask.T @w <= (labels_frac[i]+lambd)*cp.sum(w)]
    
objective = mean_match + beta*cov_match
prob = cp.Problem(cp.Minimize(objective), constraints)
prob.solve()
print("Status:", prob.status)
print("Optimal value:", prob.value)
print("top 15 Optimal w:", w[:15].value)