#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import basic_func as func
import numpy as np
import matplotlib.pyplot as plt

(xdata, ydata) = func.load_data()
targets = ydata
x = xdata
N_TRAIN = 2000;
x_train = x[0:N_TRAIN]
t_train = targets[0:N_TRAIN]
x_test = x[N_TRAIN:]
t_test = targets[N_TRAIN:]

# Plot a curve showing learned function.
# Use linspace to get a set of samples on which to evaluate    
x_ev = np.linspace(np.asscalar(min(x)), np.asscalar(max(x)), num=3601)
t_ev = np.linspace(np.asscalar(min(targets)), np.asscalar(max(targets)), num=3601)

(w, tr_err) = func.linear_regression(x_train, t_train, 'cosine', 0, 4)
(t_est, te_err) = func.evaluate_regression(x_test, t_test, w, 'cosine', 4)

# Evaluate regression on the linspace samples. (fitted curve)
(y_ev, t_err) = func.evaluate_regression(x_ev, t_ev, w, 'cosine', 4)

error = abs(y_ev-targets)
max_target = np.amax(targets)
error_perc = error*100/max_target

#plt.subplot(1, 2, 1)
plt.plot(x_ev, error_perc,'ko-')
plt.tight_layout()
plt.xlabel('x / dihedral angles')
plt.ylabel('% error / %(y_ev-targets)')
#plt.title('A visualization of the error using random outputs, cosine basis function')
plt.show()

#plt.subplot(1, 2, 2)
plt.plot(x_ev,y_ev,'r.-')
plt.plot(x_train,t_train,'bo')
plt.plot(x_test,t_test,'go')
plt.tight_layout()
plt.legend(['Learned cosine basis','Train data points','Test data points'])
plt.xlabel('x / dihedral angles')
plt.ylabel('t / rel. PES in meV')
#plt.title('A visualization of a regression estimate using random outputs, cosine basis function')
plt.show()

print('Training error: ', tr_err)
print('Test error: ', te_err)
#print('w Coeff.: ', w)

### 3
N = len(w)
gamma = 180*np.ones(N) 
V = np.zeros(N) 
V[0] = 0.5*(w[0]-(abs(w[1])+abs(w[2])+abs(w[3])+abs(w[4])))
gamma[0] = 0    

for i in range(1, N):
    V[i,] = abs(w[i,])
    if w[i,] > 0:
        gamma[i,] = 0
        
print('V: ', V)
print('gamma: ', gamma)
     
