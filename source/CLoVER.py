'''
CLoVER (Contour Location Via Entropy Reduction)
----------------------------------------------------------------------
Implementation of the algorithm CLoVER, described in the paper

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
import scipy.stats as st
from MISGP import ismemberTol, argWhereSameXtol
from copy import deepcopy

#---------------------------------------------------------------------
# Function CLoVER
#---------------------------------------------------------------------

def CLoVER(g, f, cost, noiseVariance, samplePoints, nIter, tolH=1.e-4, tolX=None, epsilon=2., integrationPoints=2500, integrationWeights=None, log=False):
    print 'Start of CLoVER'

    tcost = 0.
    
    nIS = f.nIS
    d = f.dimension()
    tolX = tolX
    
    if type(integrationPoints) is int:
        nInt = integrationPoints
        if d == 1:
            xMin = np.amin(samplePoints)
            xMax = np.amax(samplePoints)
        else:
            xMin = np.amin(samplePoints, axis=1)
            xMax = np.amax(samplePoints, axis=1)
            
        xInt, wInt = entropyImportance(f, epsilon, nInt, xMin, xMax)
    else:
        xInt = integrationPoints
        wInt = integrationWeights/np.sum(integrationWeights)
        nInt = xInt.size/d
    
    KitPrior = f.evaluateCovariancePrior([0]*nInt, xInt, f.source, f.x)
    vInt = evaluateVariance(f, [0]*nInt, xInt, KitPrior)
    mInt = evaluateMean(f, [0]*nInt, xInt, KitPrior)
        
    H = contourEntropy(f, epsilon, xInt, wInt, mInt, vInt, KitPrior)        
    print 'Iteration {:d}, contour entropy = {:.3e}, cost = {:.3e}'.format(0, H, tcost)
    
    nx = len(f.source)
    history = []
    for i in range(nx-1):
        if d == 1:
            history = updateHistory(history, -nx+i+1, [f.source[i]], f.x[i], f.y[i], -1., -1., f)
        else:
            history = updateHistory(history, -nx+i+1, [f.source[i]], f.x[:, i], f.y[i], -1., -1., f)
    if d == 1:
        history = updateHistory(history, 0, [f.source[nx-1]], f.x[nx-1], f.y[nx-1], H, 0., f)
    else:
        history = updateHistory(history, 0, [f.source[nx-1]], f.x[:, nx-1], f.y[nx-1], H, 0., f)
    if log:
        np.savetxt('CLoVER_history.txt', history, fmt='%1.6e')

        
    for iter in range(nIter):
        source, x = chooseSample(f, cost, noiseVariance, epsilon, samplePoints, xInt, wInt, mInt, vInt, KitPrior, tolX)
        
        iCost = 0.
        if source == 0:
            i = argWhereSameXtol(samplePoints, x)[0]
            samplePoints = np.delete(samplePoints, i, axis=1)

            y = []
            source = []
            ct = 0
            for s in range(nIS):
                if not ismemberTol(f.source, f.x, s, x, tolX):
                    ct += 1
                    ys, iCosts = g[s](x)
                    source += [s]
                    y += [ys]
                    iCost += iCosts
            
            y = np.array(y)
            x = np.tile(np.reshape(x, (d, 1)), ct)
        else:
            y, iCost = g[source](x)
            source = [source]
        
        f.update(source, x, y)
        
        if type(integrationPoints) is int:
            xInt, wInt = entropyImportance(f, epsilon, nInt, xMin, xMax)

        KitPrior = f.evaluateCovariancePrior([0]*nInt, xInt, f.source, f.x)
        vInt = evaluateVariance(f, [0]*nInt, xInt, KitPrior)
        mInt = evaluateMean(f, [0]*nInt, xInt, KitPrior)
    
        H = contourEntropy(f, epsilon, xInt, wInt, mInt, vInt, KitPrior)
        tcost += iCost
        
        print 'Iteration {:d}, contour entropy = {:.3e}, cost = {:.3e}'.format(iter+1, H, tcost)
        
        history = updateHistory(history, iter+1, source, x, y, H, tcost, f)
        
        if log:
            np.savetxt('CLoVER_history.txt', history, fmt='%1.6e')
        
        if H < tolH:
            print 'CLoVER stopped because contour entropy is smaller than specified tolerance'            
            break

    if iter == nIter-1:
        print 'CLoVER stopped because it reached the specified maximum number of iterations'
    
    return f, history

#---------------------------------------------------------------------
# Function used to solve optimization problem that defines new
# samples
#---------------------------------------------------------------------

def chooseSample(f, cost, noiseVariance, epsilon, samplePoints, xInt, wInt, mInt, vInt, KitPrior, tolX):
    d = f.dimension()
    nSample = samplePoints.size/d
    HWx = entropy(f, epsilon, xInt, mInt, vInt, KitPrior)
            
    umax = -1.e10
    for s in range(f.nIS):
        for i in range(nSample):
            if d == 1:
                x = samplePoints[i]
            else:
                x = samplePoints[:, i]
            
            if not ismemberTol(f.source, f.x, s, x, tolX):
                if type(cost[s]) is float:
                    iCost = cost[s]
                else:
                    iCost = cost[s](x)
            
                if type(noiseVariance[s]) is float:
                    iNoise = noiseVariance[s]
                else:
                    iNoise = noiseVariance[s](x)
            
                EHWx = lookAheadEntropy(f, s, x, iNoise, epsilon, xInt, mInt, vInt, KitPrior)
                
                u = np.sum(wInt*np.maximum(HWx.copy() - EHWx.copy(), 0.))/iCost
            
                if u > umax:
                    umax = u
                    xSample = x
                    sourceSample = s
                                
    return sourceSample, xSample
    
#---------------------------------------------------------------------
# Functions used to compute contour entropy
#---------------------------------------------------------------------

def contourEntropy(f, epsilon, xInt, wInt, mInt=None, vInt=None, KitPrior=None):
    HWx = entropy(f, epsilon, xInt, mInt, vInt, KitPrior)
    
    return np.sum(HWx*wInt)
    

def entropy(f, epsilon, x, m=None, v=None, KxtPrior=None):
    d = f.dimension()
    nx = x.size/d
    
    if m is None:
        mean = evaluateMean(f, [0]*nx, x, KxtPrior)
    else:
        mean = m.copy()
        
    if v is None:
        sigma2 = evaluateVariance(f, [0]*nx, x, KxtPrior)
    else:
        sigma2 = v.copy()

    tolr = np.min(f.getRankTolerance())
    sigma = np.maximum(sigma2, tolr)
    sigma = np.sqrt(sigma2)
    
    if type(epsilon) is int or type(epsilon) is float:
        eps = epsilon*sigma
    else:
        eps = epsilon(f, x)

    pl = np.maximum(st.norm.cdf(-eps, mean, sigma), 1.e-12)
    pu = np.maximum(st.norm.sf(eps, mean, sigma), 1.e-12)
    pc = np.maximum(1. - pl - pu, 1.e-12)

    return -(pl*np.log(pl) + pc*np.log(pc) + pu*np.log(pu))

    
def entropyImportance(f, epsilon, nInt, xMin, xMax):
    d = f.dimension()
    
    nExplore = 10*nInt
    if d == 1:
        xExplore = np.random.rand(nExplore)*(xMax - xMin) + xMin
    else:
        xMin = np.reshape(xMin, (d, 1))
        xMax = np.reshape(xMax, (d, 1))
        xExplore = np.random.rand(d, nExplore)*np.tile((xMax - xMin), nExplore) + np.tile(xMin, nExplore)
    
    HWx = entropy(f, epsilon, xExplore)
    q = HWx/np.sum(HWx)
    iInt = np.digitize(np.random.rand(nInt), np.concatenate(([0.], np.cumsum(q)))) - 1

    if d == 1:
        xInt = xExplore[iInt]
    else:
        xInt = xExplore[:, iInt]
    
    wInt = np.divide(np.ones(nInt), q[iInt])
    wInt = wInt/float(nInt*nExplore)
    
    return xInt, wInt
    
def lookAheadEntropy(f, source, x, noise, epsilon, xInt, mInt, vInt, KitPrior):
    d = f.dimension()
    nInt = xInt.size/d

    shift = st.norm.ppf(np.exp(-1))
    c = np.exp(-1.)

    tolr = np.min(f.getRankTolerance())
    
    KtsPrior = f.evaluateCovariancePrior(f.source, f.x, source, x)
    aux = f.evaluateCovariancePrior([0]*nInt, xInt, source, x)
    Kis = f.evaluateCovariancePrior([0]*nInt, xInt, source, x) - np.dot(KitPrior, f.applyKinverse(KtsPrior))
    Kss = noise + evaluateVariance(f, source, x, KxtPrior=KtsPrior.T)
    Kss = np.maximum(Kss, tolr)

    mean = mInt.copy()
    meanVariance = np.power(Kis.flatten(), 2)/Kss
    lookAheadVariance = vInt - meanVariance
    lookAheadVariance = np.maximum(lookAheadVariance, tolr)
    sigmaHat = np.sqrt(vInt)
    lookAheadStd = np.sqrt(lookAheadVariance)

    if type(epsilon) is int or type(epsilon) is float:
        eps = epsilon*lookAheadStd
    else:
        eps = epsilon(f, xInt)
        
    EHWx = np.zeros(nInt)
    for i in range(2):
        for j in range(2):
            num = -0.5*np.power(mean + ((-1.)**i)*eps + ((-1.)**j)*shift*lookAheadStd, 2)
            EHWx += np.exp(np.divide(num, vInt))
    EHWx *= c*np.divide(lookAheadStd, sigmaHat)

    return EHWx

# --------------------------------------------------------------------    
# Functions used to speedup evaluation of mean and covariance with
# pre-computed covariance matrix
# --------------------------------------------------------------------    
    
def evaluateMean(f, source, x, KxtPrior=None):
    m = f.evaluateMeanPrior(source, x)
    if f.y is None:
        return m
    else:
        if KxtPrior is None:
            Kxt = f.evaluateCovariancePrior(source, x, f.source, f.x)
        else:
            Kxt = KxtPrior.copy()    
        return m + np.dot(Kxt, f.KiY)
        

def evaluateVariance(f, source, x, KxtPrior=None):
    V = f.evaluateVariancePrior(source, x)
    if f.y is None:
        return V
    else:
        if KxtPrior is None:
            Kxt = f.evaluateCovariancePrior(source, x, f.source, f.x)
        else:
            Kxt = KxtPrior.copy()
            
        aux = f.applyKinverse(Kxt.T)
        for i in range(V.size):
            V[i] -= np.dot(Kxt[i, :], aux[:, i])
    
    return V
    
# --------------------------------------------------------------------    
# Functions used for logging the solution
# --------------------------------------------------------------------    

def readHistory(f, file='CLoVER_history.txt'):
    d = f.dimension()
    npm, npk = f.hyperparameterDimension()
    nIS = f.nIS
    history = np.loadtxt(file)
    nRow, nCol = history.shape

    source = history[:, 3].flatten().tolist()
    source = [int(s) for s in source]
    tcost, indx = np.unique(history[:, 2], return_index=True)
    tcost = tcost.flatten()
    H = history[indx, 1]
    H = H.flatten()
    x = history[:, 4:(4+d)]
    x = x.T
    y = history[:, (4+d):(5+d)]
    y = y.flatten()
    pm = []
    pk = []
    ct = 0
    for s in range(nIS):
        pm += [history[-1, (5+d+ct):(5+npm[s]+d+ct)]]
        ct += npm[s]
    for s in range(nIS):
        pk += [history[-1, (5+d+ct):(5+npk[s]+d+ct)]]
        ct += npk[s]
    
    f.setObservations(source, x, y)
    f.setHyperparameter(pm, pk)
    f.decomposeCovariance()
    
    return f, tcost, H
    
    
def updateHistory(history, iter, source, x, y, H, tcost, f):
    d = f.dimension()
    nIS = f.nIS
    nEntry = len(source)
    for i in range(nEntry):
        historyIter = [iter, H, tcost] + [source[i]]
        ct = 4;
        if d == 1:
            ct += 1
            if nEntry > 1:
                historyIter += [x[i]]
            else:
                historyIter += [x]
        else:
            for j in range(d):
                ct += 1
                if nEntry > 1:
                    historyIter += [x[j, i]]
                else:
                    historyIter += [x[j]]
        
        ct += 1
        if nEntry > 1:
            historyIter += [y[i]]
        else:
            historyIter += [y]

        pm, pk = f.getHyperparameter()
        for s in range(nIS):
            if pm[s] is not None:
                for p in pm[s]:
                    if p is not None:
                        ct += 1
                        historyIter += [p]

        for s in range(nIS):
            if pk[s] is not None:
                for p in pk[s]:
                    if p is not None:
                        ct += 1
                        historyIter += [p]
            
        historyIter = np.reshape(np.array(historyIter), (1, ct))
    
        if len(history) == 0:
            history = historyIter
        else:
            history = np.concatenate((history, historyIter), axis=0)

    return history
    
