'''
This code is part of the software CLoVER.py
------------------------------------------------------------
Implementation of Gaussian process class
------------------------------------------------------------
Distributed under the MIT License (see LICENSE.md)
Copyright 2018 Alexandre Marques, Remi Lam, and Karen Willcox
'''

import nlopt
import numpy as np

class GP(object):

    def __init__(self, mean, kernel, tolr=1e-8, hyperparameterMethod='default', optimizationOptions='default'):
        dm = mean.dimension()
        dk = kernel.dimension()
        
        if dm == dk:
            self.d = dm
            self.mean = mean.copy()
            self.kernel = kernel.copy()
            self.tolr = tolr
            self.hyper = hyperparameterMethod
            self.nlopt = optimizationOptions
            self.x = []
            self.y = []
            self.U = []
            self.S = []
            self.KiY = []
        else:
            raise ValueError('Mean function and covariance kernel must be defined over the same number of dimensions')
       
       
    def __add__(self, other):
        if isinstance(other, GP):
            if self.hyper == 'default':
                hyper = other.getHyperparameterMethod()
            else:
                hyper = self.getHyperparameterMethod()
            
            if self.nlopt == 'default':
                opt = other.getOptimizationOptions()
            else:
                opt = self.getOptimizationOptions()
            
            mean = self.getMean() + other.getMean()
            kernel = self.getKernel() + other.getKernel()
            tolr = min(self.getRankTolerance(), other.getRankTolerance())
            plus = GP(mean, kernel, tolr, hyper, opt)
            x, y = uniqueObservations(self.x, self.y, other.x, other.y)
            plus.train(x, y)
        else:
            raise ValueError('GP object can only be added to other GP object')
            
        return plus
  
            
    def __mul__(self, other):
        if type(other) is int or type(other) is float or callable(other):
            if self.hyper == 'default':
                hyper = other.getHyperparameterMethod()
            else:
                hyper = self.getHyperparameterMethod()
            
            if self.nlopt == 'default':
                opt = other.getOptimizationOptions()
            else:
                opt = self.getOptimizationOptions()
            
            mean = other*self.getMean()
            kernel = other*self.getKernel()
            tolr = self.getRankTolerance()
            mul = GP(mean, kernel, tolr, hyper, opt)
            mul.train(self.x, self.y)
        else:
            raise ValueError('GP object can only be multiplied by integer, float, or method')
        
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
        tolr = self.getRankTolerance()
        hyper = self.getHyperparameterMethod()
        opt = self.getOptimizationOptions()
        pm, pk = self.getHyperparameter()

        cp = GP(mean, kernel, tolr, hyper, opt)
        cp.setObservations(self.x.copy(), self.y.copy())
        cp.setHyperparameter(pm, pk)
        if len(self.U) > 0:
            cp.U = self.U.copy()
            cp.S = self.S.copy()
            cp.V = self.V.copy()
            cp.KiY = self.KiY.copy()

        return cp


    def decomposeCovariance(self):
        K = self.evaluateCovariancePrior(self.x, [])
        U, S, V = np.linalg.svd(K, full_matrices=0, compute_uv=1)
        r = np.sum(S > self.tolr)
        self.U = U[:, :r]
        self.S = S[:r]
        self.V = V[:r, :]
        self.KiY = self.applyKinverse(self.y - self.evaluateMeanPrior(self.x))
        
    
    def dimension(self):
        return self.d
        

    def estimateHyperparameter(self):
        temp = self.copy()
        nmp, nkp = self.hyperparameterDimension()
        ntp = nmp + nkp
        
        if ntp == 0:
            return
        
        def logLikelihoodParameter(p, grad):
            pm = p[:nmp]
            pk = p[nmp:]
            temp.setHyperparameter(pm, pk)
            temp.decomposeCovariance()

            if grad.size > 0:
                grad[:] = temp.gradientLogLikelihood()

            return temp.logLikelihood()
        
        p0 = []
        pl = []
        pu = []
        if nmp > 0:
            pm0 = self.mean.getHyperparameter()
            p0 = pm0
            pl = self.mean.hyperparameterLowerBound()
            pu = self.mean.hyperparameterUpperBound()
        
        if nkp > 0:   
            pk0 = self.kernel.getHyperparameter()
            pkl = self.kernel.hyperparameterLowerBound()
            pku = self.kernel.hyperparameterUpperBound()

            p0 = np.concatenate((p0, pk0))
            pl = np.concatenate((pl, pkl))
            pu = np.concatenate((pu, pku))

        if self.nlopt == 'default':
            opt = nlopt.opt(nlopt.GN_DIRECT, ntp)
            opt.set_lower_bounds(np.maximum(pl, p0*(1. - 1./np.sqrt(2))))
            opt.set_upper_bounds(np.minimum(pu, p0*(1. + 1./np.sqrt(2))))
            opt.set_max_objective(logLikelihoodParameter)
            opt.set_xtol_rel(1.e-4)
            opt.set_maxeval(500*ntp)
            p0 = opt.optimize(p0)

            opt = nlopt.opt(nlopt.LD_MMA, ntp)
            opt.set_lower_bounds(pl)
            opt.set_upper_bounds(pu)
            opt.set_max_objective(logLikelihoodParameter)
            opt.set_xtol_rel(1.e-4)
            opt.set_maxeval(500*ntp)
            p = opt.optimize(p0)
            
        else:
            opt = self.nlopt
            p = opt.optimize(p0)

        
        pm = p[:nmp]
        pk = p[nmp:]
        
        self.setHyperparameter(pm, pk)

        
    def evaluateMean(self, x):
        x = np.array(x)
        if x.size > 0:
            m = self.evaluateMeanPrior(x)
        else:
            m = self.evaluateMeanPrior(self.x)

        if self.y.size == 0:
            return m
        else:
            if x.size > 0:
                K = self.kernel.evaluate(x, self.x)
            else:
                K = self.kernel.evaluate(self.x, [])

            return m + np.dot(K, self.KiY)
    

    def evaluateMeanPrior(self, x):
        x = np.array(x)
        return self.mean.evaluate(x)


    def evaluateCovariance(self, x1, x2):
        K = self.evaluateCovariancePrior(x1, x2)
        if len(self.y) == 0:
            return K
        else:
            K1t = self.evaluateCovariancePrior(x1, self.x)
            x2 = np.array(x2)
            if x2.size > 0:
                Kt2 = self.evaluateCovariancePrior(self.x, x2)
                return K - np.dot(K1t, self.applyKinverse(Kt2))
            else:
                return K - np.dot(K1t, self.applyKinverse(K1t.T))


    def evaluateCovariancePrior(self, x1, x2):
        return self.kernel.evaluate(x1, x2)
    

    def evaluateVariance(self, x):
        V = self.evaluateVariancePrior(x)
        if len(self.y) == 0:
            return V
        else:
            Kxt = self.evaluateCovariancePrior(x, self.x)
            return V - np.diagonal(np.dot(Kxt, self.applyKinverse(Kxt.T)))
        

    def evaluateVariancePrior(self, x):
        if hasattr(self.kernel, 'evaluateDiagonal'):
            return self.kernel.evaluateDiagonal(x)
        else:
            return np.diagonal(self.kernel.evaluate(x, []))


    def getHyperparameter(self):
        pm = self.mean.getHyperparameter()
        pk = self.kernel.getHyperparameter()
        
        return pm, pk
        

    def getHyperparameterMethod(self):
        return self.hyper
        
        
    def getKernel(self):
        return self.kernel.copy()

        
    def getMean(self):
        return self.mean.copy()
                

    def getOptimizationOptions(self):
        return self.nlopt

                
    def getRankTolerance(self):
        return self.tolr

        
    def gradientLogLikelihood(self):
        y = self.y - self.mean.evaluate(self.x)
        nx = y.size
        nmp, nkp = self.hyperparameterDimension()
        grad = np.zeros(nmp+nkp)
        if nmp > 0:
            dm = self.mean.gradient(self.x)
            for i in range(nmp):
                grad[i] = 0.5*np.dot(y.T, self.applyKinverse(dm[:, i]))
                grad[i] += 0.5*np.dot(dm[:, i].T, self.KiY)

        if nkp > 0:
            dK = self.kernel.gradient(self.x, self.x)
            for i in range(nkp):
                aux = np.dot(dK[:, :, i], self.KiY)
                aux = self.applyKinverse(aux)
                grad[i+nmp] = 0.5*np.dot(y.T, aux)
                for j in range(nx):
                    trKidK = np.dot(self.U.T, dK[:, j, i])
                    trKidK = np.linalg.solve(np.diag(self.S), trKidK)
                    trKidK = np.dot(self.V[:, j].T, trKidK)
                    grad[i+nmp] -= 0.5*trKidK
                
        return grad
        

    def hyperparameterDimension(self):
        nmp = self.mean.hyperparameterDimension()
        nkp = self.kernel.hyperparameterDimension()
        
        return nmp, nkp
        
        
    def logLikelihood(self):
        y = self.y.copy() - self.mean.evaluate(self.x)
        logDetK = np.sum(np.log(self.S))
        n = np.size(y)
        return -0.5*(np.dot(y.T, self.KiY) + logDetK + n*np.log(2.*np.pi))
        
        
    def setHyperparameter(self, pm, pk):
        self.mean.setHyperparameter(pm)
        self.kernel.setHyperparameter(pk)
        

    def setObservations(self, x, y):
        x, y = uniqueObservations(x, y)
        d = self.dimension()
        nx = y.size
        if nx == 1:
            x = np.reshape(x, (d, nx))
            y = np.array([y])
            
        self.x = x
        self.y = y

        
    def train(self, x, y):
        self.setObservations(x, y)
        if self.hyper == 'default':
            self.estimateHyperparameter()
        else:
            pm, pk = self.hyper(self)
            self.setHyperparameter(pm, pk)
            
        self.decomposeCovariance()
       
    
    def update(self, x, y):
        d = self.dimension()
        x = np.array(x)
        y = np.array(y)
        nx = y.size
        if nx == 1:
            x = np.reshape(x, (d, nx))
            y = np.array([y])
        
        xNew = np.concatenate((self.x, x), axis=1)
        yNew = np.concatenate((self.y, y))        
        self.train(xNew, yNew)


# --- end of GP class ---

def uniqueObservations(x, y):
    x = np.array(x)
    y = np.array(y)
    nx = y.size
    d = x.size/nx
    
    if nx > 1:
        if d == 1:
            x = np.reshape(x, (d, nx))

        _, indx = np.unique(x, return_index=True, axis=1)
        indx = np.sort(indx.flatten()).tolist()
        
        x = x[:, indx]
        y = y[indx]
        if d == 1:
            x = x.flatten()
    
    return x, y
