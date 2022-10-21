"""
The main file to run BSDE solver to solve parabolic partial differential equations (PDEs).

"""

import json
#import munch
import os
import logging

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle

import equation
from solver import BSDESolver



T= 0.5
nbStep= 20
L=5
alpha= 0.2
delta=0.001
gamma=0.2
batch_size=512
valid_size=2048
num_hiddens= [L+10, L+10]
num_iterations= 1500
lp= 0
rp= 1
dtype="float64"
nTest=1

Y0_all=np.zeros((nTest,L))
print("Deep BSDE-3 Ex-2  *** Setup R1_11 ***"," nTest ",nTest," L ",L," nbStep ",nbStep," alpha ",alpha," delta ",delta, " Gamma ", gamma," batchSize ",batch_size)
for i in range (nTest):
    print(" ########  %%%%%%%%%%%   Run   ",i+1, "  %%%%%%%%%%%     ####     ")
    tf.keras.backend.set_floatx(dtype)
    bsde = equation.FBSDE(T,L,nbStep,num_hiddens, dtype, batch_size, valid_size, num_iterations,lp,rp,alpha, delta,gamma)
    bsde_solver = BSDESolver(T,L,nbStep,num_hiddens, dtype, batch_size, valid_size, num_iterations, bsde)
    training_history, y_init_history, loss_history = bsde_solver.train()
    Y0_all[i,:]=y_init_history[-1]
print("Mean Y0 from all simulations",np.mean(Y0_all,axis=0) )
print("Variance of Y0 from all simulations",np.var(Y0_all,axis=0))
print("Deep BSDE-3 Ex-2  *** Setup R1_11 ***"," nTest ",nTest," L ",L," nbStep ",nbStep," alpha ",alpha," delta ",delta, " Gamma ", gamma,"  batchSize ",batch_size)

xi, u0xS, u0xT=bsde.u0_simlV2(np.mean(Y0_all,axis=0))

# Draw the graph
plt.plot(xi,u0xT,label="Analytic Solution")
plt.plot(xi, u0xS,label="Deep BSDE-3")
plt.xlabel("Space")
plt.ylabel("u(0,x)")
plt.legend()
plt.show()




