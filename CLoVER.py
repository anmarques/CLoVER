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

def CLoVER(g, f, cost, noiseVariance, samplePoints, nIter, tol=1.e-8, epsilon=2., integrationPoints=2500, integrationWeights=[]):
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
        if d == 1:
            nInt = xInt.size
        else:
            nInt = xInt.shape[1]
        
    H = contourEntropy(f, epsilon, xInt, wInt)        
    print 'Iteration {:d}, contour entropy = {:.3e}, cost = {:.3e}'.format(0, H, tcost)
    
    if d == 1:
        history = updateHistory([], -1, 0., H, tcost, f)
    else:
        history = updateHistory([], -1, np.zeros((d, 1)), H, tcost, f)
        
    for iter in range(nIter):
        sourceSample, xSample = chooseSample(f, cost, noiseVariance, epsilon, samplePoints, xInt, wInt)
        
        iCost = 0.
        if sourceSample == 0:
            y = np.zeros(nIS)
            for s in range(nIS):
                y[s], iCosts = g[s](xSample)
                iCost += iCosts

            x = np.tile(np.reshape(xSample, (d, 1)), nIS)
            source = np.arange(nIS)
                
        else:
            y, iCost = g[sourceSample](xSample)
            x = xSample.copy()
            source = sourceSample
            
        f.update(source, x, y)
        if type(integrationPoints) is int:
            xInt, wInt = entropyImportance(f, epsilon, nInt, xMin, xMax)

        H = contourEntropy(f, epsilon, xInt, wInt)
        tcost += iCost
        
        print 'Iteration {:d}, contour entropy = {:.3e}, cost = {:.3e}'.format(iter+1, H, tcost)
        
        history = updateHistory(history, sourceSample, xSample, H, tcost, f)
        
        if H < tol:
            break

    return f, history


def updateHistory(history, source, x, H, tcost, f):
    d = f.dimension()
    nIS = f.nIS
    historyIter = [H, tcost, source]
    ct = 3;
    for i in range(d):
        ct += 1
        historyIter += [x[i]]
        
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
        return historyIter
    else:
        return np.concatenate((history, historyIter), axis=0)
    

def chooseSample(f, cost, noiseVariance, epsilon, samplePoints, xInt, wInt):
    d = f.dimension()
    
    nSample = samplePoints.size/d
    HWx = entropy(f, epsilon, xInt)
        
    umax = -1.e10
    for s in range(f.nIS):
        for i in range(nSample):
            if d == 1:
                x = samplePoints[i]
            else:
                x = samplePoints[:, i]
            
            if type(cost[s]) is float:
                iCost = cost[s]
            else:
                iCost = cost[s](x)
            
            if type(noiseVariance[s]) is float:
                iNoise = noiseVariance[s]
            else:
                iNoise = noiseVariance[s](x)
            
            EHWx = lookAheadEntropy(f, s, x, iNoise, epsilon, xInt)
            u = np.sum(wInt*np.maximum(HWx - EHWx, 0.)/iCost)
            
            if u > umax:
                umax = u
                xSample = x
                sourceSample = s
                indx = i
                                
    return sourceSample, xSample


def entropy(f, epsilon, x):
    d = f.dimension()
    nx = x.size/d
        
    mean = f.evaluateMean(np.zeros(nx), x)
    tolr = np.min(f.getRankTolerance())
    sigma = np.sqrt(np.maximum(f.evaluateVariance(np.zeros(nx), x), tolr))

    if type(epsilon) is int or type(epsilon) is float:
        eps = epsilon*sigma
    else:
        eps = epsilon(f, x)

    pl = np.maximum(st.norm.cdf(-(eps + mean)/sigma), 1.e-12)
    pu = np.maximum(1. - st.norm.cdf((eps - mean)/sigma), 1.e-12)
    pc = np.maximum(1. - pl - pu, 1.e-12)

    
    return -(pl*np.log(pl) + pc*np.log(pc) + pu*np.log(pu))


def lookAheadEntropy(f, source, x, noise, epsilon, xInt):
    d = f.dimension()
    if d == 1:
        nInt = xInt.size
    else:   
        nInt = xInt.shape[1]

    shift = st.norm.ppf(np.exp(-1));
    c = 1./np.exp(1.) 

    mean = f.evaluateMean(np.zeros(nInt), xInt)
    meanVariance = f.lookAheadMeanVariance(np.zeros(nInt), xInt, source, x, noise)
    lookAheadVariance = f.lookAheadVariance(np.zeros(nInt), xInt, source, x, noise)
    sigmaHat = np.sqrt(meanVariance + lookAheadVariance)
    lookAheadStd = np.sqrt(lookAheadVariance)

    if type(epsilon) is int or type(epsilon) is float:
        eps = epsilon*lookAheadStd
    else:
        eps = epsilon(f, xInt)
        
    EHWx = np.zeros(nInt)
    for i in range(2):
        for j in range(2):
            num = -0.5*(mean + ((-1.)**i)*eps + ((-1.)**j)*shift*lookAheadStd)
            EHWx += np.exp(np.divide(num, sigmaHat))
    EHWx *= np.divide(lookAheadStd, sigmaHat)
    
    return EHWx

    

def contourEntropy(f, epsilon, xInt, wInt):
    HWx = entropy(f, epsilon, xInt)
    
    return np.sum(HWx*wInt)
    

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
    wInt = wInt/np.sum(wInt)
    
    return xInt, wInt
