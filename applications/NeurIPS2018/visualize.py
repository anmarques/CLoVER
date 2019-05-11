'''
This code is part of the software CLoVER.py
----------------------------------------------------------------------
Postprocessing of results computed by multiModalExample_NeurIPS2019.py

Compute area of region where multi-information source surrogate is
positive
----------------------------------------------------------------------
Distributed under the MIT License (see LICENSE.md)
Copyright 2018 Alexandre Marques, Remi Lam, and Karen Willcox
'''

import numpy as np
import matplotlib.pyplot as plt
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

f = MISGP.MISGP([m0, m1, m2], [k0, k1, k2], tolr=1.e-8)

# --------------------------------------------------------------------
# Output directory
# --------------------------------------------------------------------

if os.path.exists('visualization'):
    os.system('rm visualization/*')
else:
    os.makedirs('visualization')
    
# --------------------------------------------------------------------
# Load data from file
# --------------------------------------------------------------------

indx = sys.argv[1].zfill(2)
fileName = 'runs/multiModal_' + indx + '.txt'
print fileName
history = np.loadtxt(fileName)
nIter = int(history[-1, 0])
    
costRun = np.unique(history[:, 2])
costRun = costRun.flatten()
costRun = costRun[1:]

# --------------------------------------------------------------------
# Plotting setup
# --------------------------------------------------------------------

plt.rcParams.update({'font.size': 14, 'axes.titlesize':16, 'axes.labelsize': 14, 'legend.fontsize': 12, 'xtick.labelsize':12, 'ytick.labelsize':12, 'text.usetex':True})

markerStyles = ['s', 'o', 'd']
markerSizes = 3*[6]
markerColors = ['b', 'darkorange', 'gold']
markerFillStyles = ['full', 'none', 'none']
x1, x2 = np.meshgrid(np.linspace(-4., 7., 200), np.linspace(-3., 8., 200))
xPlot = np.array([x1.flatten(), x2.flatten()])
levels = np.linspace(-10., 10., 100)
fig = plt.figure(figsize=(5., 4.))

# --------------------------------------------------------------------
# Loop through iterations
# --------------------------------------------------------------------

for it in range(nIter+1):
    ax = fig.add_axes([0.12, 0.25, 0.65, 0.6])
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    cbaxes = fig.add_axes([0.79, 0.25, 0.04, 0.6]) 
    
    row = np.argwhere(history[:, 0] < it+1)
    row = row[-1][0]
    source = [int(s) for s in history[:(row+1), 3]]
    f.setObservations(source, history[:(row+1), 4:6].T, history[:(row+1), 6])
    f.setHyperparameter(3*[None], [history[row, 7:10], history[row, 10:13], history[row, 13:16]])
    f.decomposeCovariance()

# --------------------------------------------------------------------
# Set plot title
# --------------------------------------------------------------------

    ax.set_title(r'iteration {:d}, cost = {:.2f}\\contour entropy = {:.2e}'.format(it, history[row, 2]+11.01, history[row, 1]))

# --------------------------------------------------------------------
# Plot samples
# --------------------------------------------------------------------

    plots = []
    legends = []
    for source in range(1, 3):
        j = [k for k,x in enumerate(f.source) if x == source]
        p, = ax.plot(f.x[0, j], f.x[1, j], markerStyles[source], markersize=markerSizes[source], color=markerColors[source], fillstyle=markerFillStyles[source], markeredgewidth=1.5)
        plots += [p]
        legends += ['IS' + str(source)]
    
    j = [k for k,x in enumerate(f.source) if x == 0]
    p, = ax.plot(f.x[0, j], f.x[1, j], markerStyles[0], markersize=markerSizes[0], color=markerColors[0], fillstyle=markerFillStyles[0], markeredgewidth=1.5, markeredgecolor=markerColors[0])
    plots = [p] + plots
    legends = [r'IS0'] + legends
    
# --------------------------------------------------------------------
# Plot contours
# --------------------------------------------------------------------    
        
    y = f.evaluateMean(40000*[0], xPlot)
    y = np.reshape(y, (200, 200))

    yref, _ = lfm0(xPlot)
    yref = np.reshape(yref, (200, 200))

    cnt = ax.contour(x1, x2, y, levels)
  
# --------------------------------------------------------------------
# Plot zero contour
# --------------------------------------------------------------------    

    cnt0 = ax.contour(x1, x2, y, np.array([0.]), linewidths=2., colors='k', linestyles='dashed')
    h,_ = cnt0.legend_elements()
    plots = [h[0]] + plots
    legends = [r'surrogate'] + legends

# --------------------------------------------------------------------
# Plot reference zero contour
# --------------------------------------------------------------------    

    cnt0ref = ax.contour(x1, x2, yref, np.array([0.]), linewidths=2., colors='k', linestyles='solid')
    href,_ = cnt0ref.legend_elements()
    plots = [href[0]] + plots
    legends = [r'truth'] + legends


# --------------------------------------------------------------------
# Add legends
# --------------------------------------------------------------------    

    fig.legend(plots, legends, loc='lower center', ncol=5, numpoints=1, bbox_to_anchor=[0.12, 0.02, 0.7, 0.08], handletextpad=-0.2, labelspacing=0.2, columnspacing=0.3)

# --------------------------------------------------------------------
# Add colorbar
# --------------------------------------------------------------------    
    
    cbar = plt.colorbar(cnt, orientation='vertical', cax=cbaxes, format='% 2d', ticks=range(-10, 11, 4))
    cbar.set_label(r'$E[f(0, x)]$')
    
# --------------------------------------------------------------------
# Save figure
# --------------------------------------------------------------------    
    
    plt.savefig('visualization/multiModal_' + indx + '_' + str(it).zfill(3) + '.png', dpi=200, format='png')
    fig.clf()

plt.close('all')
