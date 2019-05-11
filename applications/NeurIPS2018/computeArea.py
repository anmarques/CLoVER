'''
This code is part of the software CLoVER.py
----------------------------------------------------------------------
Postprocessing of results computed by multiModalExample_NeurIPS2019.py

Compute area of region where multi-information source surrogate is
positive.
Compute error with respect to reference result
----------------------------------------------------------------------
Distributed under the MIT License (see LICENSE.md)
Copyright 2018 Alexandre Marques, Remi Lam, and Karen Willcox
'''

import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import sys
fileDir = os.path.dirname(os.path.abspath(__file__))
sourcePath = os.path.join(fileDir, '../../source')
sys.path.append(sourcePath)
import functionLibrary as fl
import meanFunction as mf
import covarianceKernel as ck
import MISGP

# --------------------------------------------------------------------
# Information sources
# --------------------------------------------------------------------

def lfm0(x):
    nx = x.size/2
    return fl.multiModal(x), float(nx)

# --------------------------------------------------------------------
# GP priors
# --------------------------------------------------------------------

# Mean functions
m0 = mf.meanZero(2)
m1 = mf.meanZero(2)
m2 = mf.meanZero(2)

# Covariance Kernels
k0 = ck.kernelSquaredExponential(2, [1., 1./11., 1./11.])
k1 = ck.kernelSquaredExponential(2, [1., 1./11., 1./11.])
k2 = ck.kernelSquaredExponential(2, [1., 1./11., 1./11.])

f = MISGP.MISGP([m0, m1, m2], [k0, k1, k2], tolr=1.e-14)

# --------------------------------------------------------------------
# Output directory
# --------------------------------------------------------------------

if not os.path.exists('area'):
    os.makedirs('area')
    
# --------------------------------------------------------------------
# Reference area estimate
# --------------------------------------------------------------------

areaRef = 0.
for n in range(20):
    x = np.random.rand(2, int(1e6))*11.
    x[0, :] -= 4.
    x[1, :] -= 3.
    fRef, _ = lfm0(x)
    areaRef += np.sum(fRef > 0.)*121./(20.*1.e6)
    
print 'Reference area = ' + str(areaRef)
    
# --------------------------------------------------------------------
# Loop through runs and records the results for each
# --------------------------------------------------------------------

fileList = sorted(glob.glob('runs/multiModal_*.txt'))
area = np.zeros((100, 16))
x = np.random.rand(2, int(1e6))*11.
x[0, :] -= 4.
x[1, :] -= 3.
for run in range(len(fileList)):
    print 'Processing ' + fileList[run] + ' ...'
    history = np.loadtxt(fileList[run])
    nIter = int(history[-1, 0])
    
    costRun = np.unique(history[:, 2])
    costRun = costRun.flatten()
    costRun = costRun[1:]
    
    areaRun = []
    for it in range(nIter+1):
        row = np.argwhere(history[:, 0] < it+1)
        row = row[-1][0]
        source = [int(s) for s in history[:(row+1), 3]]
        f.setObservations(source, history[:(row+1), 4:6].T, history[:(row+1), 6])
        f.setHyperparameter(3*[None], [history[row, 7:10], history[row, 10:13], history[row, 13:16]])
        f.decomposeCovariance()
        areaRun += [np.sum(f.evaluateMean([0]*int(1e6), x) > 0.)*121./1.e6]

    out = np.vstack((range(nIter+1), costRun, areaRun))
    fileName = 'area/multiModal_' + str(run).zfill(2) + '.dat'
    np.savetxt(fileName, out.T, fmt='%1.6e')
    
    area[run, :] = np.interp(range(16), costRun, areaRun)
           
# --------------------------------------------------------------------
# Computes statistics
# --------------------------------------------------------------------

error = np.abs(area - areaRef)
errorMedian = np.median(error, axis=0)
error25th = np.percentile(error, 25, axis=0)
error75th = np.percentile(error, 75, axis=0)

# --------------------------------------------------------------------
# Plot statistics
# --------------------------------------------------------------------

plt.rcParams.update({'font.size': 14, 'axes.titlesize':16, 'axes.labelsize': 14, 'legend.fontsize': 12, 'xtick.labelsize':12, 'ytick.labelsize':12, 'text.usetex':True})

fig = plt.figure(figsize=(5., 4.))
ax = fig.add_axes([0.15, 0.12, 0.8, 0.8])
ax.set_xlabel(r'Cost')
ax.set_ylabel(r'Error')
ax.set_yscale('log')
plt.errorbar(11.01 + np.arange(16), errorMedian, yerr=np.array([error25th, error75th]))
plt.savefig('areaError.png', dpi=200, format='png')
