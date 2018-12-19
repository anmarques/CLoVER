'''
This code is part of the software CLoVER.py
------------------------------------------------------------
Implementation of multi-information source Gaussian process
class

This implementation is based on the paper

M. Poloczek, J. Wang, and P. Frazier,
Multi-information source optimization,
Advances in Neural Information Processing Systems 30, 2017,
pp. 4291-4301
------------------------------------------------------------
Distributed under the MIT License (see LICENSE.md)
Copyright 2018 Alexandre Marques, Remi Lam, and Karen Willcox
'''

import nlopt
import numpy as np
import scipy.spatial.distance as dist
import GP

class MISGP(object):

    def __init__(self, mean, kernel, tolr=1e-8, hyperparameterMethod='default', optimizationOptions='default'):
        nIS = len(mean)        
        if len(kernel) == nIS:
            d = mean[0].dimension()
            if type(tolr) == list:
                self.tolr = np.min(np.array(tolr))
                if not len(tolr) == nIS:
                    raise ValueError('The number of tolerance values must match the number of covariance kernels')
            else:
                self.tolr = tolr
                tolr = [tolr]*nIS
                
            if type(hyperparameterMethod) == list:
                if not len(hyperparameterMethod) == nIS:
                    raise ValueError('The number of hyperparameter estimate methods must match the number of covariance kernels')
            else:
               hyper = [hyperparameterMethod]*nIS

            if type(optimizationOptions) == list:
                if not len(optimizationOptions) == nIS:
                    raise ValueError('The number of optimization option sets must match the number of covariance kernels')
            else:
               opt = [optimizationOptions]*nIS

            self.f = []
            for s in range(nIS):
                if mean[s].dimension() == d and kernel[s].dimension() == d:
                    self.f += [GP.GP(mean[s], kernel[s], tolr[s], hyper[s], opt[s])]
                else:
                    raise ValueError('The dimension of all mean functions and covariance kernels must be the same')
        else:
            raise ValueError('The number of mean functions must match the number of covariance kernels')
                       
        self.d = d
        self.nIS = nIS
        self.source = []
        self.x = []
        self.y = []
      
       
    def __add__(self, other):
        if isinstance(other, MISGP):
            if self.nIS == other.nIS:
                plus = self.copy()
                for s in range(self.nIS):
                    plus.f[s] += other.f[s]
            else:
                raise ValueError('Added MISGP objects must be contain the same number of information sources')
        else:
            raise ValueError('MISGP object can only be added to other MISGP object')
            
        return plus
        

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self + other
            
            
    def __mul__(self, other):
        if type(other) is int or type(other) is float or callable(other):
            prod = self.copy()
            for s in range(self.nIS):
                plus.f[s] = other*plus.f[s].copy()
        else:
            raise ValueError('MISGP object can only be multiplied by integer, float, or method')
        
        return prod
        
        
    def __rmul__(self, other):
        if other == 1:
            return self
        else:
            return self * other


    def copy(self):
        mean = self.getMean()
        kernel = self.getKernel()
        hyper = []
        opt = []
        tolr = self.getRankTolerance()
        for s in range(self.nIS):
            tolr += [self.f[s].getRankTolerance()]
            hyper += [self.f[s].getHyperparameterMethod()]
            opt += [self.f[s].getOptimizationOptions()]
            
        cp = MISGP(mean, kernel, tolr, hyper, nlopt)
        return cp
        
        
    def getMean(self):
        mean = []
        for s in range(self.nIS):
            mean[s] = self.f[s].getMean()

        return mean
        
    
    def getKernel(self):
        kernel = []
        for s in range(self.nIS):
            kernel[s] = self.f[s].getKernel()

        return kernel


    def getHyperparameter(self):
        pm = []
        pk = []
        for s in range(self.nIS):
            pms, pks = self.f[s].getHyperparameter()
            pm += [pms]
            pk += [pks]
        
        return pm, pk


    def getRankTolerance(self):
        tolr = []
        for s in range(self.nIS):
            tolr += [self.f[s].getRankTolerance()]
        
        return tolr


    def setObservations(self, source, x, y):
        source = np.array(source)
        x = np.array(x)
        y = np.array(y)
        self.source = source.copy()
        self.x = x.copy()
        self.y = y.copy()
        d = self.dimension()

        nx = source.size
        if nx == 1:
            source = np.array([source])
            y = np.array([y])
            if d > 1:
                x = np.reshape(x, (d, 1))
        
        indx0 = np.argwhere(source == 0).flatten()
        n0 = indx0.size
        y0 = y[indx0]
        if d == 1:
            x0 = x[indx0]
        else:
            x0 = x[:, indx0]
        
        self.f[0].setObservations(x0, y0)
        for s in range(1, self.nIS):
            indxs = np.argwhere(source == s).flatten()
            ns = indxs.size
            ys = np.zeros(n0)
            
            for i in range(n0):
                for j in indxs:
                    if d == 1:
                        if np.abs(x0[i] - x[j]) < 1.e-10:
                            ys[i] = y[j] - y0[i]
                            break
                    else:
                        if np.all(np.abs(x0[:, i] - x[:, j]) < 1.e-10):
                            ys[i] = y[j] - y0[i]
                            break
            
            self.f[s].setObservations(x0, ys)
     
    
    def setHyperparameter(self, pm, pk):
        for s in self.nIS:
            self.f[s].setHyperparameter(pm[s], pk[s])


    def evaluateMeanPrior(self, source, x):
        source = np.array(source)
        x = np.array(x)

        d = self.dimension()
        nx = source.size
        if nx == 1:
            source = np.array([source])
            if d > 1:
                x = np.reshape(x, (d, nx))
        
        m = self.f[0].mean.evaluate(x)
        if np.any(source > 0):
            for s in range(1, self.nIS):
                if np.any(source == s):
                    indx = np.argwhere(source == s).flatten()
                    if d == 1:
                        m[indx] += self.f[s].mean.evaluate(x[indx])
                    else:
                        m[indx] += self.f[s].mean.evaluate(x[:, indx])
        return m
 
 
    def evaluateMean(self, source, x):            
        m = self.evaluateMeanPrior(source, x)
        if self.y.size == 0:
            return m
        else:
            K = self.evaluateCovariancePrior(source, x, self.source, self.x)
        
            return m + np.dot(K, self.KiY)
    
    
    def evaluateCovariancePrior(self, source1, x1, source2, x2):
        source1 = np.array(source1)
        x1 = np.array(x1)
        d = self.dimension()
        nx1 = source1.size
        if nx1 == 1:
            source1 = np.array([source1])
            if d > 1:
                x1 = np.reshape(x1, (d, nx1))

        source2 = np.array(source2)
        if source2.size > 0:
            x2 = np.array(x2)

            nx2 = source2.size
            if nx2 == 1:
                source2 = np.array([source2])
                if d > 1:
                    x2 = np.reshape(x2, (d, nx2))

            K = self.f[0].evaluateCovariancePrior(x1, x2)
            if np.any(source1 > 0) and np.any(source2 > 0):
                for s in range(1, self.nIS):
                    if np.any(source1 == s) and np.any(source2 == s):
                        indx1 = np.argwhere(source1 == s).flatten()
                        indx2 = np.argwhere(source2 == s).flatten()
                        indx1g, indx2g = np.meshgrid(indx1, indx2)
                        indx1g = indx1g.flatten()
                        indx2g = indx2g.flatten()
                        if d == 1:
                            K[indx1g, indx2g] += self.f[s].evaluateCovariancePrior(x1[indx1], x2[indx2]).flatten()
                        else:
                            K[indx1g, indx2g] += self.f[s].evaluateCovariancePrior(x1[:, indx1], x2[:, indx2]).flatten()

        else:
            K = self.f[0].evaluateCovariancePrior(x1, [])
            if np.any(source1 > 0):
                for s in range(1, self.nIS):
                    if np.any(source1 == s):
                        indx1 = np.argwhere(source1 == s).flatten()
                        indx1g, indx2g = np.meshgrid(indx1, indx1)
                        indx1g = indx1g.flatten()
                        indx2g = indx2g.flatten()
                        if d == 1:
                            K[indx1g, indx2g] += self.f[s].evaluateCovariancePrior(x1[indx1], []).flatten()
                        else:
                            K[indx1g, indx2g] += self.f[s].evaluateCovariancePrior(x1[:, indx1], []).flatten()
        
        return K
    

    def evaluateVariancePrior(self, source, x):
        source = np.array(source)
        x = np.array(x)
        d = self.dimension()
        nx = source.size
        if nx == 1:
            source = np.array([source])
            if d > 1:
                x = np.reshape(x, (d, nx))

        V = self.f[0].evaluateVariancePrior(x)
        if np.any(source > 0):
            for s in range(1, self.nIS):
                if np.any(source == s):
                    indx = np.argwhere(source == s).flatten()
                    if d == 1:
                        V[indx] += self.f[s].evaluateVariancePrior(x[indx])
                    else:
                        V[indx] += self.f[s].evaluateVariancePrior(x[:, indx])
        
        return V

    
    def evaluateCovariance(self, source1, x1, source2, x2):        
        K = self.evaluateCovariancePrior(source1, x1, source2, x2)       
        if self.y.size == 0:
            return K
        else:
            K1t = self.evaluateCovariancePrior(source1, x1, self.source, self.x)
            source2 = np.array(source2)
            if source2.size > 0:
                Kt2 = self.evaluateCovariancePrior(self.source, self.x, source2, x2)
                return K - np.dot(K1t, self.applyKinverse(Kt2))
            else:
                return K - np.dot(K1t, self.applyKinverse(K1t.T))


    def evaluateVariance(self, source, x):
        V = self.evaluateVariancePrior(source, x)       
        if len(self.y) == 0:
            return V
        else:
            Kxt = self.evaluateCovariancePrior(source, x, self.source, self.x)
            return V - np.diagonal(np.dot(Kxt, self.applyKinverse(Kxt.T)))

        
    def lookAheadMeanVariance(self, sourceEval, xEval, sourceSample, xSample, noise):        
        Kes = self.evaluateCovariance(sourceEval, xEval, sourceSample, xSample)
        Kss = noise + self.evaluateVariance(sourceSample, xSample)
        Kss = np.maximum(Kss, self.tolr)
        
        nSample = np.array(sourceSample).size
        if nSample == 1:
            return (np.power(Kes, 2)/Kss).flatten()
        else:
            return np.power(Kes, 2)/Kss

            
    def lookAheadVariance(self, sourceEval, xEval, sourceSample, xSample, noise):
        xEval = np.array(xEval)
        sourceEval = np.array(sourceEval)
        nEval = sourceEval.size
        d = self.dimension()
        
        if nEval == 1:
            sourceEval = np.array([sourceEval])
            if d > 1:
                xEval = np.reshape(xEval, (d, 1))
        
        V = self.evaluateVariance(sourceEval, xEval)
        Kss = noise + self.evaluateVariance(sourceSample, xSample)
        Kss = np.maximum(Kss, self.tolr)
        Kes = self.evaluateCovariance(sourceEval, xEval, sourceSample, xSample)

        return V - np.power(Kes.flatten(), 2)/Kss


    def applyKinverse(self, y):
        y = np.array(y)
        KiY = np.dot(self.U.T, y)
        KiY = np.linalg.solve(np.diag(self.S), KiY)
        KiY = np.dot(self.V.T, KiY)

        return KiY
        

    def decomposeCovariance(self):
        K = self.evaluateCovariancePrior(self.source, self.x, [], [])
        U, S, V = np.linalg.svd(K, full_matrices=0, compute_uv=1)
        r = np.sum(S/S[0] > self.tolr)
        self.U = U[:, :r]
        self.S = S[:r]
        self.V = V[:r, :]
        self.KiY = self.applyKinverse(self.y - self.evaluateMeanPrior(self.source, self.x))
        
    
    def dimension(self):
        return self.d


    def estimateHyperparameter(self):
        for s in range(self.nIS):
            pm, pk = self.f[s].estimateHyperparameter()
            self.f[s].setHyperparameter(pm, pk)
                        

    def hyperparameterDimension(self):
        nmp = []
        nkp = []
        for s in range(self.nIS):
            nmps, nkps = self.f[s].hyperparameterDimension()
            nmp += [nmps]
            nkp += [nkps]
        
        return nmp, nkp

        
    def train(self, source, x, y):
        self.setObservations(source, x, y)
        for s in range(self.nIS):
            self.f[s].train(self.f[s].x, self.f[s].y)
            
        self.decomposeCovariance()
       
    
    def update(self, source, x, y):
        source, x, y = uniqueObservations(self.source, self.x, self.y, source, x, y, self.dimension())
        self.setObservations(source, x, y)
        if np.any(source == 0):
            for s in range(self.nIS):
                self.f[s].train(self.f[s].x, self.f[s].y)
            
        self.decomposeCovariance()


# --- end of GP class ---

def uniqueObservations(source1, x1, y1, source2, x2, y2, d):
    source1 = np.array(source1)
    source2 = np.array(source2)
    nx1 = source1.size
    nx2 = source2.size
    
    if nx1 == 0 and nx2 == 0:
        x = []
        y = []
    elif nx1 > 0 and nx2 == 0:
        x = np.array(x1)
        y = np.array(y1)
        source = source1.copy()
    elif nx1 == 0 and nx2 > 0:
        x = np.array(x2)
        y = np.array(y2)
        source = source2.copy()
    else:
        x1 = np.array(x1)
        y1 = np.array(y1)
        source = source1.copy()
        
        if nx1 == 1:
            source1 = np.array([source1])
            y1 = np.array([y1])
            if d > 1:
                x1 = np.reshape(x1, (d, 1))
                  
        x2 = np.array(x2)
        y2 = np.array(y2)
        if nx2 == 1:
            source2 = np.array([source2])
            y2 = np.array([y2])
            if d > 1:
                x2 = np.reshape(x2, (d, 1))
                
        z1 = np.concatenate((np.reshape(source1, (1, nx1)), x1))
        z2 = np.concatenate((np.reshape(source2, (1, nx2)), x2))
        dz = dist.cdist(z1.T, z2.T, 'euclidean')
        iNew = np.argwhere(np.all(dz > 1e-10, axis=0)).flatten()
        nNew = iNew.size
        if nNew > 0:
            source2 = source2[iNew].flatten()
            y2 = y2[iNew].flatten()
            source = np.concatenate((source, source2))
            y = np.concatenate((y1, y2))
            if d == 1:
                x2 = x2[iNew].flatten()
                x = np.concatenate((x1, x2))
            else:
                x = np.concatenate((x1, x2[:, iNew]), axis=1)

            return source, x, y
        else:
            return source1, x1, y1
        