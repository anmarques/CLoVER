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

def hfm(x):
    nx = x.size/2
    return fl.multiModal(x), float(nx)
    
def lfm1(x):
    nx = x.size/2
    if nx == 1:
        return fl.multiModal(x) + np.sin(5.*(x[0] + 0.5*x[1])/22. + 1.25), 0.01*float(nx)
    else:
        return fl.multiModal(x) + np.sin(5.*(x[0, :] + 0.5*x[1, :])/22. + 1.25), 0.01*float(nx)
            
def lfm2(x):
    nx = x.size/2
    if nx == 1:
        return fl.multiModal(x) + 3.*np.sin(5.*(x[0] + x[1] + 7.)/11.), 0.001*float(nx)
    else:
        return fl.multiModal(x) + 3.*np.sin(5.*(x[0, :] + x[1, :] + 7.)/11.), 0.001*float(nx)

x0 = np.random.rand(2, 10)*11.
x0[0, :] += -4.
x0[1, :] += -3.
y0, _ = hfm(x0)
y1, _ = lfm1(x0)
y2, _ = lfm2(x0)
x = np.tile(x0, 3)
y = np.concatenate((y0, y1, y2))
source = 10*[0] + 10*[1] + 10*[2]

m0 = mf.meanZero(2)
k0 = ck.kernelSquaredExponential(2, [np.std(y0), 11.0, 11.0])

m1 = mf.meanZero(2)
k1 = ck.kernelSquaredExponential(2, [np.std(y1 - y0), 11.0, 11.0])

m2 = mf.meanZero(2)
k2 = ck.kernelSquaredExponential(2, [np.std(y2 - y0), 11.0, 11.0])

f = MISGP.MISGP([m0, m1, m2], [k0, k1, k2], tolr=1.e-14)

print 'Training MISGP surrogate with initial set of observations... '
f.train(source, x, y)
print 'Done!'

x1s, x2s = np.meshgrid(np.linspace(-4., 7., 30), np.linspace(-3., 8., 30))
samplePoints = np.concatenate((np.reshape(x1s, (1, 900)), np.reshape(x2s, (1, 900))))
noiseVariance = 3*[0.]
cost = [1., 0.01, 0.001]

nIter = 400
g = [hfm, lfm1, lfm2]

f, history = CLoVER.CLoVER(g, f, cost, noiseVariance, samplePoints, nIter, integrationPoints=2500)

np.savetxt('CLoVER_history.txt', history, fmt='%d, %1.6e, %1.6e, %d, %1.6e, %1.6e, %1.6e, %1.6e, %1.6e, %1.6e, %1.6e, %1.6e, %1.6e, %1.6e, %1.6e, %1.6e')
