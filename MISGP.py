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
import GP
from copy import deepcopy

class MISGP(object):

    def __init__(self, mean, kernel, tolr=1e-8, hyperparameterMethod='default', optimizationOptions='default'):
        nIS = len(mean)
        d = mean[0].dimension()
        for s in range(nIS):
            if not mean[s].dimension() == d:
                raise ValueError('All mean functions must have the same dimension')

        if len(kernel) == nIS:
            for s in range(nIS):
                if not kernel[s].dimension() == d:
                    raise ValueError('All mean functions must have the same dimension')
        else:
            raise ValueError('The number of mean functions must match the number of covariance kernels')
        
        if type(tolr) == list:
            if not len(tolr) == nIS:
                raise ValueError('The number of tolerance values must match the number of covariance kernels')
        else:
            tolr = [tolr]*nIS
                
        if type(hyperparameterMethod) == list:
            if not len(hyperparameterMethod) == nIS:
                raise ValueError('The number of hyperparameter estimate methods must match the number of covariance kernels')
            else:
                hyper = deepcopy(hyperparameterMethod)
        else:
            hyper = [hyperparameterMethod]*nIS

        if type(optimizationOptions) == list:
            if not len(optimizationOptions) == nIS:
                raise ValueError('The number of optimization option sets must match the number of covariance kernels')
            else:
                opt = deepcopy(optimizationOptions)
        else:
            opt = [optimizationOptions]*nIS

        self.f = []
        for s in range(nIS):
            self.f += [GP.GP(mean[s], kernel[s], tolr[s], hyper[s], opt[s])]
                       
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


    def __mul__(self, other):
        if type(other) is int or type(other) is float or callable(other):
            prod = self.copy()
            for s in range(self.nIS):
                plus.f[s] = other*plus.f[s].copy()
        else:
            raise ValueError('MISGP object can only be multiplied by integer, float, or method')
        
        return prod
        
        
    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self + other


    def __rmul__(self, other):
        if other == 1:
            return self
        else:
            return self * other

            
    def applyKinverse(self, y):
        y = np.array(y)
        KiY = np.dot(self.U.T, y)
        KiY = np.linalg.solve(np.diag(self.S), KiY)
        KiY = np.dot(self.V.T, KiY)

        return KiY
        

    def copy(self):
        mean = self.getMean()
        kernel = self.getKernel()
        hyper = []
        opt = []
        for s in range(self.nIS):
            hyper += [self.f[s].hyper]
            opt += [self.f[s].nlopt]
            
        tolr = self.getRankTolerance()
            
        cp = MISGP(mean, kernel, tolr, hyper, opt)
        cp.setObservations(deepcopy(self.source), self.x.copy(), self.y.copy())
        pm, pk = self.getHyperparameter()
        cp.setHyperparameter(deepcopy(pm), deepcopy(pk))
        
        return cp
        
        
    def decomposeCovariance(self):
        K = self.evaluateCovariancePrior(self.source, self.x, [], [])
        U, S, V = np.linalg.svd(K, full_matrices=0, compute_uv=1)
        tolr = np.min(self.getRankTolerance())
        r = np.sum(S > tolr)
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

            
    def evaluateMean(self, source, x):
        m = self.evaluateMeanPrior(source, x)
        if self.y.size == 0:
            return m
        else:
            K = self.evaluateCovariancePrior(source, x, self.source, self.x)
        
            return m + np.dot(K, self.KiY)
            
            
    def evaluateMeanPrior(self, source, x):
        if not type(source) == list:
            source = [source]

        x = np.array(x)
        d = self.dimension()
        nx = len(source)
        if nx == 1:
            if d > 1:
                x = np.reshape(x, (d, nx))
        
        m = self.f[0].mean.evaluate(x)
        if any(z > 0 for z in source):
            for s in range(1, self.nIS):
                if any(z == s for z in source):
                    indx = argwhereList(source, s)
                    if d == 1:
                        m[indx] += self.f[s].evaluateMeanPrior(x[indx])
                    else:
                        m[indx] += self.f[s].evaluateMeanPrior(x[:, indx])
        return m
 
    
    def evaluateCovariance(self, source1, x1, source2, x2):        
        K = self.evaluateCovariancePrior(source1, x1, source2, x2)       
        if self.y.size == 0:
            return K
        else:
            K1t = self.evaluateCovariancePrior(source1, x1, self.source, self.x)
            if len(source2) > 0:
                Kt2 = self.evaluateCovariancePrior(self.source, self.x, source2, x2)
                return K - np.dot(K1t, self.applyKinverse(Kt2))
            else:
                return K - np.dot(K1t, self.applyKinverse(K1t.T))
    
    
    def evaluateCovariancePrior(self, source1, x1, source2, x2):
        d = self.dimension()
        if not type(source1) == list:
            source = [source1]
        
        nx1 = len(source1)
        x1 = np.array(x1)
        if nx1 == 1 and d > 1:
            x1 = np.reshape(x1, (d, nx1))

        if not type(source2) == list:
            source2 = [source2]
        
        if len(source2) > 0:
            nx2 = len(source2)
            x2 = np.array(x2)
            if nx2 == 1 and d > 1:
                x2 = np.reshape(x2, (d, nx2))

            K = self.f[0].evaluateCovariancePrior(x1, x2)
            if any(z > 0 for z in source1) and any(z > 0 for z in source2):
                for s in range(1, self.nIS):
                    if any(z == s for z in source1) and any(z == s for z in source2):
                        indx1 = argwhereList(source1, s)
                        indx2 = argwhereList(source2, s)
                        indx1g, indx2g = np.meshgrid(indx1, indx2, indexing='ij')
                        indx1g = indx1g.flatten().tolist()
                        indx2g = indx2g.flatten().tolist()
                        if d == 1:
                            K[indx1g, indx2g] += self.f[s].evaluateCovariancePrior(x1[indx1], x2[indx2]).flatten()
                        else:
                            K[indx1g, indx2g] += self.f[s].evaluateCovariancePrior(x1[:, indx1], x2[:, indx2]).flatten()

        else:
            K = self.f[0].evaluateCovariancePrior(x1, [])
            if any(z > 0 for z in source1):
                for s in range(1, self.nIS):
                    if any(z == s for z in source1):
                        indx1 = argwhereList(source1, s)
                        indx1g, indx2g = np.meshgrid(indx1, indx1, indexing='ij')
                        indx1g = indx1g.flatten().tolist()
                        indx2g = indx2g.flatten().tolist()
                        if d == 1:
                            K[indx1g, indx2g] += self.f[s].evaluateCovariancePrior(x1[indx1], []).flatten()
                        else:
                            K[indx1g, indx2g] += self.f[s].evaluateCovariancePrior(x1[:, indx1], []).flatten()
        
        return K
    

    def evaluateVariance(self, source, x):
        V = self.evaluateVariancePrior(source, x)
        if len(self.y) == 0:
            return V
        else:
            Kxt = self.evaluateCovariancePrior(source, x, self.source, self.x)
            aux = self.applyKinverse(Kxt.T)
            for i in range(V.size):
                V[i] -= np.dot(Kxt[i, :], aux[:, i])
            
            return V


    def evaluateVariancePrior(self, source, x):
        if not type(source) == list:
            source = [source]
            
        x = np.array(x)
        d = self.dimension()
        nx = len(source)
        if nx == 1 and d > 1:
            x = np.reshape(x, (d, nx))

        V = self.f[0].evaluateVariancePrior(x)
        if any(z > 0 for z in source):
            for s in range(1, self.nIS):
                if any(z == s for z in source):
                    indx = argwhereList(source, s)
                    if d == 1:
                        V[indx] += self.f[s].evaluateVariancePrior(x[indx])
                    else:
                        V[indx] += self.f[s].evaluateVariancePrior(x[:, indx])
        
        return V


    def getHyperparameter(self):
        pm = []
        pk = []
        for s in range(self.nIS):
            pms, pks = self.f[s].getHyperparameter()
            pm += [pms]
            pk += [pks]
        
        return pm, pk

        
    def getKernel(self):
        kernel = []
        for s in range(self.nIS):
            kernel += [self.f[s].getKernel()]

        return kernel


    def getMean(self):
        mean = []
        for s in range(self.nIS):
            mean += [self.f[s].getMean()]

        return mean


    def getRankTolerance(self):
        tolr = []
        for s in range(self.nIS):
            tolr += [self.f[s].getRankTolerance()]
        
        return tolr

   
    def hyperparameterDimension(self):
        nmp = []
        nkp = []
        for s in range(self.nIS):
            nmps, nkps = self.f[s].hyperparameterDimension()
            nmp += [nmps]
            nkp += [nkps]
        
        return nmp, nkp

   
    def setHyperparameter(self, pm, pk):
        for s in range(self.nIS):
            self.f[s].setHyperparameter(pm[s], pk[s])
            
           
    def setObservations(self, source, x, y):
        source, x, y = uniqueObservations(source, x, y)
        d = self.dimension()
        self.source = source
        self.x = x
        self.y = y
                
        indx0 = argwhereList(source, 0)
        n0 = len(indx0)
        y0 = y[indx0]
        if d == 1:
            x0 = x[indx0]
        else:
            x0 = x[:, indx0]
        
        self.f[0].setObservations(x0, y0)
        for s in range(1, self.nIS):
            indxs = argwhereList(source, s)
            ns = len(indxs)
            ys = []
            xs = []
            
            for i in range(n0):
                if d == 1:
                    j = argwhereSameX(x[indxs], x0[i])
                    if len(j) > 0:
                        ys += [y[indxs[j[0]]] - y0[i]]
                        if len(xs) > 0:
                            xs = np.append(xs, x0[i])
                        else:
                            xs = np.array([x0[i]])
                else:
                    j = argwhereSameX(x[:, indxs], x0[:, i])
                    if len(j) > 0:
                        ys += [y[indxs[j[0]]] - y0[i]]
                        if len(xs) > 0:
                            xs = np.concatenate((xs, np.reshape(x0[:, i], (d, 1))), axis=1)
                        else:
                            xs = np.reshape(x0[:, i], (d, 1))
            
            ys = np.array(ys)
            self.f[s].setObservations(xs, ys)

        
    def train(self, source, x, y):
        self.setObservations(source, x, y)
        for s in range(self.nIS):
            self.f[s].train(self.f[s].x, self.f[s].y)
            
        self.decomposeCovariance()
       
    
    def update(self, source, x, y):
        if not type(source) == list:
            source = [source]
            
        sourceNew = self.source + source
            
        d = self.dimension()
        nx = len(source)
        if nx == 1:
            x = np.reshape(x, (d, nx))
            y = np.array([y])
        
        xNew = np.concatenate((self.x, x), axis=1)
        yNew = np.concatenate((self.y, y))        
        self.setObservations(sourceNew, xNew, yNew)
        if any(z == 0 for z in source):
            for s in range(self.nIS):
                self.f[s].train(self.f[s].x, self.f[s].y)
            
        self.decomposeCovariance()


# --- end of GP class ---

def argwhereList(lst, element):
    ar = np.array(lst)
    indx = np.argwhere(ar == element).flatten()

    return indx.tolist()
    
    
def argwhereSameX(array, x):
    tol = 1.e-10
    if x.ndim == 0:
        return np.argwhere(np.abs(array - x) < tol).flatten().tolist()
    else:
        d, nx = array.shape
        test = [True]*nx
        for i in range(d):
            aux = (np.abs(array[i, :] - x[i]) < 1.e-10).flatten().tolist()
            test = [test[r] and aux[r] for r in range(nx)]

        return argwhereList(test, True)
    
    
def uniqueObservations(source, x, y):
    if not type(source) == list:
        source = [source]
        
    nx = len(source)
    x = np.array(x)
    y = np.array(y)

    d = x.size/nx
    if nx > 1:
        if d == 1:
            x = np.reshape(x, (d, nx))
        sourceArray = np.reshape(np.array(source), (1, nx))
        z = np.concatenate((sourceArray, x))
        _, indx = np.unique(z, return_index=True, axis=1)
        indx = np.sort(indx.flatten()).tolist()
        
        z = z[:, indx]
        source = []
        for i in range(len(indx)):
            source += [int(z[0, i])]

        x = z[1:, :]
        y = y[indx]
        if d == 1:
            x = x.flatten()
    
    return source, x, y
