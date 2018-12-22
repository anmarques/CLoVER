'''
CLoVER (Contour Location Via Entropy Reduction)
------------------------------------------------------------
Implementation of the algorithm CLoVER, described in the
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
import scipy.stats as st
from MISGP import argwhereSameX
from copy import deepcopy


def CLoVER(g, f, cost, noiseVariance, samplePoints, nIter, tol=1.e-8, epsilon=2., integrationPoints=2500, integrationWeights=[]):
    print 'Start of CLoVER'

    tcost = 0.
    
    nIS = f.nIS
    d = f.dimension()
    
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
        wInt = integrationWeights
        nInt = xInt.size/d
    
    KitPrior = f.evaluateCovariancePrior([0]*nInt, xInt, f.source, f.x)
    vInt = evaluateVariance(f, [0]*nInt, xInt, KitPrior)
    mInt = evaluateMean(f, [0]*nInt, xInt, KitPrior)
        
    H = contourEntropy(f, epsilon, xInt, wInt, mInt, vInt, KitPrior)        
    print 'Iteration {:d}, contour entropy = {:.3e}, cost = {:.3e}'.format(0, H, tcost)
    
    if d == 1:
        history = updateHistory([], 0, [-1], 0., 0., H, tcost, f)
    else:
        history = updateHistory([], 0, [-1], np.zeros((d, 1)), 0., H, tcost, f)
        
    for iter in range(nIter):
        source, x = chooseSample(f, cost, noiseVariance, epsilon, samplePoints, xInt, wInt, mInt, vInt, KitPrior)
        
        iCost = 0.
        if source == 0:
            i = argwhereSameX(samplePoints, x)[0]
            samplePoints = np.delete(samplePoints, i, axis=1)

            y = np.zeros(nIS)
            for s in range(nIS):
                y[s], iCosts = g[s](x)
                iCost += iCosts

            x = np.tile(np.reshape(x, (d, 1)), nIS)
            source = range(nIS)
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
        
        if H < tol:
            print 'CLoVER stopped because contour entropy is smaller than specified tolerance'            
            break

    print 'CLoVER stopped it reached the specified maximum number of iterations'
    return f, history


def chooseSample(f, cost, noiseVariance, epsilon, samplePoints, xInt, wInt, mInt, vInt, KitPrior):
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
            
            if not ismember(f.source, f.x, s, x):
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
    

def contourEntropy(f, epsilon, xInt, wInt, mInt=[], vInt=[], KitPrior=[]):
    HWx = entropy(f, epsilon, xInt, mInt, vInt, KitPrior)
    
    return np.sum(HWx*wInt)
    

def entropy(f, epsilon, x, m=[], v=[], KxtPrior=[]):
    d = f.dimension()
    nx = x.size/d
    
    if len(m) == 0:
        mean = evaluateMean(f, [0]*nx, x, KxtPrior)
    else:
        mean = m.copy()
        
    if len(v) == 0:
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
    
    
def evaluateMean(f, source, x, KxtPrior=[]):
    m = f.evaluateMeanPrior(source, x)
    if len(f.y) == 0:
        return m
    else:
        if len(KxtPrior) == 0:
            Kxt = f.evaluateCovariancePrior(source, x, f.source, f.x)
        else:
            Kxt = KxtPrior.copy()    
        return m + np.dot(Kxt, f.KiY)
        

def evaluateVariance(f, source, x, KxtPrior=[]):
    V = f.evaluateVariancePrior(source, x)
    if len(f.y) == 0:
        return V
    else:
        if len(KxtPrior) == 0:
            Kxt = f.evaluateCovariancePrior(source, x, f.source, f.x)
        else:
            Kxt = KxtPrior.copy()
            
        aux = f.applyKinverse(Kxt.T)
        for i in range(V.size):
            V[i] -= np.dot(Kxt[i, :], aux[:, i])
    
    return V
    

def ismember(sourceArray, xArray, source, x):
    
    indx = argwhereSameX(xArray, x)
    if len(indx) > 0:
        if sourceArray[indx[0]] == source:
            return True
        else:
            return False
    else:
        return False
        
    
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
    

def readHistory(f, file='CLoVER_history.txt'):
    d = f.dimension()
    npm, npk = f.hyperparameterDimension()
    nIS = f.nIS
    history = np.loadtxt(file, delimiter=',')
    nRow, nCol = history.shape
    source = history[1:, 3].flatten().tolist()
    source = [int(s) for s in source]
    tcost, indx = np.unique(history[:, 2], return_index=True)
    tcost = tcost.flatten()
    H = history[indx, 1]
    H = H.flatten()
    x = history[1:, 4:(4+d)]
    x = x.T
    y = history[1:, (4+d):(5+d)]
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
        
    f.update(source, x, y)
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

        npm, npk = f.hyperparameterDimension()
        pm, pk = f.getHyperparameter()
        for s in range(nIS):
            if npm[s] > 0:
                for p in pm[s]:
                    ct += 1
                    historyIter += [p]

        for s in range(nIS):
            if npk[s] > 0:
                for p in pk[s]:
                    ct += 1
                    historyIter += [p]
            
        historyIter = np.reshape(np.array(historyIter), (1, ct))
    
        if len(history) == 0:
            history = historyIter
        else:
            history = np.concatenate((history, historyIter), axis=0)

    return history
    