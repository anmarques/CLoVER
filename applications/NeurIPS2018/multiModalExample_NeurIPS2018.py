'''
This code is part of the software CLoVER.py
----------------------------------------------------------------------
Implementation of the example multimodal function from paper

A.N. Marques, R.R. Lam, and K.E. Willcox,
Contour location via entropy reduction leveraging multiple information
sources,
Advances in Neural Information Processing Systems 31, 2018,
pp. 5222-5232.
----------------------------------------------------------------------
Distributed under the MIT License (see LICENSE.md)
Copyright 2018 Alexandre Marques, Remi Lam, and Karen Willcox
'''

import numpy as np
import os
import sys
fileDir = os.path.dirname(os.path.abspath(__file__))
sourcePath = os.path.join(fileDir, '../../source')
sys.path.append(sourcePath)
import functionLibrary as fl
import meanFunction as mf
import covarianceKernel as ck
import MISGP
import CLoVER

# --------------------------------------------------------------------
# Information sources
# --------------------------------------------------------------------

def lfm0(x):
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

g = [lfm0, lfm1, lfm2]

# --------------------------------------------------------------------
# Query cost
# --------------------------------------------------------------------

cost = [1., 0.01, 0.001]

# --------------------------------------------------------------------
# Noise
# --------------------------------------------------------------------

noiseVariance = 3*[0.]

# --------------------------------------------------------------------
# GP priors
# --------------------------------------------------------------------

# Mean functions
m0 = mf.meanZero(2)
m1 = mf.meanZero(2)
m2 = mf.meanZero(2)

# Covariance Kernels
k0 = ck.kernelSquaredExponential(2, [1., 2./11., 2./11.], pmin=[0.1, 1./11., 1./11.], pmax=[10., 10./11., 10./11.])
k1 = ck.kernelSquaredExponential(2, [1., 2./11., 2./11.], pmin=[0.1, 1./11., 1./11.], pmax=[10., 10./11., 10./11.])
k2 = ck.kernelSquaredExponential(2, [1., 2./11., 2./11.], pmin=[0.1, 1./11., 1./11.], pmax=[10., 10./11., 10./11.])

# --------------------------------------------------------------------
# Sample points (for optimization step)
# --------------------------------------------------------------------

x1s, x2s = np.meshgrid(np.linspace(-4., 7., 30), np.linspace(-3., 8., 30))
samplePoints = np.concatenate((np.reshape(x1s, (1, 900)), np.reshape(x2s, (1, 900))))

# --------------------------------------------------------------------
# Integration points (for computation of contour Entropy)
# --------------------------------------------------------------------

x1i, x2i = np.meshgrid(np.linspace(-4., 7., 50), np.linspace(-3., 8., 50))
xInt = np.concatenate((np.reshape(x1i, (1, 2500)), np.reshape(x2i, (1, 2500))))
wInt = np.ones((50, 50))
wInt[1:-1, 0] = 0.5
wInt[1:-1, -1] = 0.5
wInt[0, 1:-1] = 0.5
wInt[-1, 1:-1] = 0.5
wInt[0, 0] = 0.25
wInt[0, -1] = 0.25
wInt[-1, 0] = 0.25
wInt[-1, -1] = 0.25
wInt = wInt.flatten()
wInt = wInt/(49.*49.)

# --------------------------------------------------------------------
# Maximum number of iterations
# --------------------------------------------------------------------

nIter = 300

# --------------------------------------------------------------------
# Minimum distance between samples
# --------------------------------------------------------------------

tolX = 2*[1.e-3]

# --------------------------------------------------------------------
# Contour Entropy stopping criterion
# --------------------------------------------------------------------

tolH = 1.e-8;

# --------------------------------------------------------------------
# Output directory
# --------------------------------------------------------------------

if not os.path.exists('runs'):
    os.makedirs('runs')

# --------------------------------------------------------------------
# Execution of CLoVER
# Repeat 100 times
# --------------------------------------------------------------------


for run in range(100):
    print 'Experiment ' + str(run)

    # Random initialization
    source = 10*[0] + 10*[1] + 10*[2]
    x0 = np.random.rand(2, 10)*11.
    x0[0, :] += -4.
    x0[1, :] += -3.
    y0, _ = lfm0(x0)
    y1, _ = lfm1(x0)
    y2, _ = lfm2(x0)
    x = np.tile(x0, 3)
    y = np.concatenate((y0, y1, y2))

    # Update prior to reflect variance of initial samples
    p0 = k0.getHyperparameter()
    p0min = k0.getHyperparameterLowerLimit()
    p0max = k0.getHyperparameterUpperLimit()
    p0 = k0.getHyperparameter()
    p0[0] = np.var(y0)
    p0min[0] = 0.5*p0[0]
    p0max[0] = 1.5*p0[0]
    k0.setHyperparameter(p0)
    k0.setHyperparameterLowerLimit(p0min)
    k0.setHyperparameterUpperLimit(p0max)

    p1 = k1.getHyperparameter()
    p1min = k1.getHyperparameterLowerLimit()
    p1max = k1.getHyperparameterUpperLimit()
    p1 = k1.getHyperparameter()
    p1[0] = np.var(y0 - y1)
    p1min[0] = 0.5*p1[0]
    p1max[0] = 1.5*p1[0]
    k1.setHyperparameter(p1)
    k1.setHyperparameterLowerLimit(p1min)
    k1.setHyperparameterUpperLimit(p1max)

    p2 = k2.getHyperparameter()
    p2min = k2.getHyperparameterLowerLimit()
    p2max = k2.getHyperparameterUpperLimit()
    p2 = k2.getHyperparameter()
    p2[0] = np.var(y0 - y2)
    p2min[0] = 0.5*p2[0]
    p2max[0] = 1.5*p2[0]
    k2.setHyperparameter(p2)
    k2.setHyperparameterLowerLimit(p2min)
    k2.setHyperparameterUpperLimit(p2max)

    f = MISGP.MISGP([m0, m1, m2], [k0, k1, k2], tolr=1.e-14)
    
    #Initial training
    print 'Training MISGP surrogate with initial set of observations... '
    f.train(source, x, y)
    print 'Done!'

    #Execute CLoVER
    f, history = CLoVER.CLoVER(g, f, cost, noiseVariance, samplePoints, nIter, tolH=tolH, tolX=tolX, integrationPoints=xInt, integrationWeights=wInt, log=True)
    
    #Record results
    fileName = 'runs/multiModal_' + str(run).zfill(2) + '.txt'
    os.rename('CLoVER_history.txt', fileName)
