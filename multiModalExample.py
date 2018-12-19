'''
This code is part of the software CLoVER.py
------------------------------------------------------------
Implementation of the example multimodal function from
paper

A.N. Marques, R.R. Lam, and K.E. Willcox,
Contour location via entropy reduction leveraging multiple
information sources,
Advances in Neural Information Processing Systems 31, 2018,
pp. 5222-5232.
------------------------------------------------------------
Distributed under the MIT License (see LICENSE.md)
Copyright 2018 Alexandre Marques, Remi Lam, and Karen Willcox
'''

import numpy as np
import functionLibrary as fl
import meanFunction as mf
import covarianceKernel as ck
import MISGP
import CLoVER

m0 = mf.meanZero(2)
k0 = ck.kernelSquaredExponential(2)

m1 = mf.meanZero(2)
k1 = ck.kernelSquaredExponential(2)

def hfm(x):
    nx = x.size/2
    return fl.multiModal(x), float(nx)
    
def lfm1(x):
    nx = x.size/2
    if nx == 1:
        return fl.multiModal(x) + np.sin(5.*(x[0] + 0.5*x[1])/22. + 1.25), 0.01*float(nx)
    else:
        return fl.multiModal(x) + np.sin(5.*(x[0, :] + 0.5*x[1, :])/22. + 1.25), 0.01*float(nx)
            
x0 = np.random.rand(2, 10)*11.
x0[0, :] += -4.
x0[1, :] += -3.
y0, _ = hfm(x0)
y1, _ = lfm1(x0)
x = np.tile(x0, 2)
y = np.concatenate((y0, y1))
source = 10*[0] + 10*[1]
f = MISGP.MISGP([m0, m1], [k0, k1])

print 'Training MISGP surrogate with initial set of observations... '
f.train(source, x, y)
print 'Done!'

x1s, x2s = np.meshgrid(np.linspace(-4., 7., 20), np.linspace(-3., 8., 20))
samplePoints = np.concatenate((np.reshape(x1s, (1, 400)), np.reshape(x2s, (1, 400))))
noiseVariance = 2*[0.]
cost = [1., 0.01]

nIter = 200
g = [hfm, lfm1]

print 'Starting CLoVER... '
f, history = CLoVER.CLoVER(g, f, cost, noiseVariance, samplePoints, nIter, integrationPoints=900)

np.savetxt('CLoVER_history.txt', history, fmt='%1.6e, %1.6e, %d, %1.6e, %1.6e, %1.6e, %1.6e, %1.6e, %1.6e, %1.6e, %1.6e')