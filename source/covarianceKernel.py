'''
This code is part of the software CLoVER.py
------------------------------------------------------------
Implementation of covariance kernel class
------------------------------------------------------------
Distributed under the MIT License (see LICENSE.md)
Copyright 2018 Alexandre Marques, Remi Lam, and Karen Willcox
'''

import numpy as np
import copy
import scipy.spatial.distance as dist

small = 1.e-8

class covarianceKernel(object):

    def __init__(self, d, p=None, pmin=None, pmax=None, variableHyperparameter=None):
        self.d = d
        self.p = p
        self.pmin = pmin
        self.pmax = pmax
        self.variableHyperparameter = variableHyperparameter


    def __add__(self, other):
        if isinstance(other, covarianceKernel):
            return kernelAdd(self, other)
        else:
            raise ValueError('Covariance kernel object can only be added to other covariance kernel object')
    
    
    def __radd__(self, other):
        if other == 0 or other == 0.:
            return self
        else:
            return self + other
            
            
    def __mul__(self, other):
        if type(other) is int or type(other) is float or callable(other):    
            return kernelMul(self, other)
        else:
            raise ValueError('Covariance kernel object can only be multiplied by integer, float, or method')
    
    
    def __rmul__(self, other):
        if other == 1 or other == 1.:
            return self
        else:
            return self * other

    
    def dimension(self):
        return self.d
    
    
    def getHyperparameter(self):
        return copy.deepcopy(self.p)
    
    
    def getHyperparameterLowerLimit(self):
        return copy.deepcopy(self.pmin)


    def getHyperparameterUpperLimit(self):
        return copy.deepcopy(self.pmax)


    def getVariableHyperparameter(self):
        if self.variableHyperparameter is None:
            return None
        else:
            nvp = self.variableHyperparameterDimension()
            if nvp == 0:
                return None
            else:
                p = []
                for i in range(len(self.variableHyperparameter)):
                    if self.variableHyperparameter[i]:
                        p += [self.p[i]]

                return np.array(p) 


    def setHyperparameter(self, p):
        self.p = np.array(p)


    def setHyperparameterLowerLimit(self, pmin):
        self.pmin = np.array(pmin)


    def setHyperparameterUpperLimit(self, pmax):
        self.pmax = np.array(pmax)
        

    def setVariableHyperparameter(self, p):
        if self.variableHyperparameter is not None:
            p = np.array(p)
            for i in range(len(self.variableHyperparameter)):
                if self.variableHyperparameter[i]:
                    self.p[i] = p[i]
                    

    def variableHyperparameterDimension(self):
        if self.variableHyperparameter is None:
            return 0
        else:
            return sum(self.variableHyperparameter)


    def variableHyperparameterLowerBound(self):
        return copy.deepcopy(self.pmin)


    def variableHyperparameterUpperBound(self):
        return copy.deepcopy(self.pmax)


class kernelAdd(covarianceKernel):

    def __init__(self, f1, f2):
    
        d1 = f1.dimension()
        d2 = f2.dimension()
        
        if d1 == d2:
            self.f1 = copy.deepcopy(f1)
            self.f2 = copy.deepcopy(f2)
        else:
            raise ValueError('Added covariance kernel must be defined over the same number of dimensions')

        if self.f1.variableHyperparameter is None:
            self.variableHyperameter = self.f2.variableHyperparameter
        elif self.f2.variableHyperparameter is None:
            self.variableHyperameter = self.f1.variableHyperparameter
        else:
            self.variableHyperameter = self.f1.variableHyperparameter + self.f2.variableHyperparameter
            
    
    def evaluate(self, x1, x2=None):
        return self.f1.evaluate(x1, x2) + self.f2.evaluate(x1, x2)
       

    def evaluateDiagonal(self, x):
        if hasattr(self.f1, 'evaluateDiagonal'):
            f1 = self.f1.evaluateDiagonal(x)
        else:
            f1 = np.diagonal(self.f1.evaluate(x))

        if hasattr(self.f2, 'evaluateDiagonal'):
            f2 = self.f2.evaluateDiagonal(x)
        else:
            f2 = np.diagonal(self.f2.evaluate(x))
            
        return f1 + f2
       

    def gradient(self, x1, x2=None):
        dK = None
        if self.f1.variableHyperparameterDimension() > 0:
            dK = self.f1.gradient(x1, x2)

        if self.f2.variableHyperparameterDimension() > 0:
            dK2 = self.f2.gradient(x1, x2)
            if dK is None:
                dK = dK2
            else:
                dK = np.concatenate((dK, dK2), axis=2)
                
        if dK is None:
            x1l = np.array(x1)
            n1d = x1l.size/self.d
            if x2 is None:
                dK = np.zeros((nx1, nx1))
            else:
                x2l = np.array(x2)
                n2d = x2l.size/self.d
                dK = np.zeros((nx1, nx2))
        
        return dK

            
    def dimension(self):
        return self.f1.dimension()
       
        
    def getHyperparameter(self):
        p = self.f1.getHyperparameter()
        p2 = self.f2.getHyperparameter()
        if p is None:
            p = p2
        else:
            if p2 is not None:
                p = np.concatenate((p, p2))
        
        return p
        

    def getHyperparameterLowerLimit(self):
        n1 = self.f1.hyperparameterDimension()
        n2 = self.f2.hyperparameterDimension()
        if n1 > 0:
            pmin = self.f1.getHyperparameterLowerLimit();
            if n2 > 0:
                pmin2 = self.f2.getHyperparameterLowerLimit();
                pmin = np.concatenate((pmin, pmin2))
        elif n2 > 0:
            pmin = self.f2.getHyperparameterLowerLimit();
        else:
            pmin = None
        
        return pmin


    def getHyperparameterUpperLimit(self):
        n1 = self.f1.hyperparameterDimension()
        n2 = self.f2.hyperparameterDimension()
        if n1 > 0:
            pmax = self.f1.getHyperparameterUpperLimit();
            if n2 > 0:
                pmax2 = self.f2.getHyperparameterUpperLimit();
                pmax = np.concatenate((pmax, pmax2))
        elif n2 > 0:
            pmax = self.f2.getHyperparameterUpperLimit();
        else:
            pmax = None
        
        return pmax


    def getVariableHyperparameter(self):
        p = self.f1.getVariableHyperparameter()
        p2 = self.f2.getVariableHyperparameter()
        if p is None:
            p = p2
        else:
            if p2 is not None:
                p = np.concatenate((p, p2))
        
        return p
        

    def setHyperparameter(self, p):
        n1 = self.f1.hyperparameterDimension()
        n2 = self.f2.hyperparameterDimension()
        if n1 > 0:
            self.f1.setHyperparameter(p[:n1])
            if n2 > 0:
                self.f2.setHyperparameter(p[n1:])
        elif n2 > 0:
            self.f2.setHyperparameter = p

            
    def setHyperparameterLowerLimit(self, pmin):
        n1 = self.f1.hyperparameterDimension()
        n2 = self.f2.hyperparameterDimension()
        if n1 > 0:
            self.f1.setHyperparameterLowerLimit(pmin[:n1])
            if n2 > 0:
                self.f2.setHyperparameterLowerLimit(pmin[n1:])
        elif n2 > 0:
            self.f2.setHyperparameterLowerLimit = pmin

            
    def setHyperparameterUpperLimit(self, pmax):
        n1 = self.f1.hyperparameterDimension()
        n2 = self.f2.hyperparameterDimension()
        if n1 > 0:
            self.f1.setHyperparameterUpperLimit(pmax[:n1])
            if n2 > 0:
                self.f2.setHyperparameterUpperLimit(pmax[n1:])
        elif n2 > 0:
            self.f2.setHyperparameterUpperLimit = pmax

            
    def setVariableHyperparameter(self, p):
        n1 = self.f1.variableHyperparameterDimension()
        n2 = self.f2.variableHyperparameterDimension()
        if n1 > 0:
            self.f1.setVariableHyperparameter(p[:n1])
            if n2 > 0:
                self.f2.setVariableHyperparameter(p[n1:])
        elif n2 > 0:
            self.f2.setVariableHyperparameter = p

            
    def hyperparameterDimension(self):
        return self.f1.hyperparameterDimension() + self.f2.hyperparameterDimension()
    

    def variableHyperparameterDimension(self):
        return self.f1.variableHyperparameterDimension() + self.f2.variableHyperparameterDimension()
    
    
    def variableHyperparameterLowerBound(self):
        lb = None
        if self.f1.variableHyperparameterDimension() > 0:
            lb = self.f1.variableHyperparameterLowerBound()
            
        if self.f2.variableHyperparameterDimension() > 0:
            lb2 = self.f2.variableHyperparameterLowerBound()
            if lb is None:
                lb = lb2
            else:
                lb = np.concatenate((lb, lb2))
        
        return lb


    def variableHyperparameterUpperBound(self):
        ub = None
        if self.f1.variableHyperparameterDimension() > 0:
            ub = self.f1.variableHyperparameterUpperBound()
            
        if self.f2.variableHyperparameterDimension() > 0:
            ub2 = self.f2.variableHyperparameterUpperBound()
            if ub is None:
                ub = ub2
            else:
                ub = np.concatenate((ub, ub2))
        
        return ub


class kernelMul(covarianceKernel):

    def __init__(self, f, c):
        self.f = copy.deepcopy(f)
        self.c = copy.deepcopy(c)
            
            
    def evaluate(self, x1, x2=None):
        if callable(self.c):
            return self.c(x1, x2)*self.f.evaluate(x1, x2)
        else:
            return self.c*self.f.evaluate(x1, x2)
       
        
    def evaluateDiagonal(self, x):
        if callable(self.c):
            c = self.c(x)
        else:
            c = self.c
            
        if hasattr(self.f, 'evaluateDiagonal'):
            return c*self.f.evaluateDiagonal(x)
        else:
            return c*np.diagonal(self.f.evaluate(x))

            
    def gradient(self, x1, x2=None):
        if callable(self.c):
            return self.c(x1, x2)*self.f.gradient(x1, x2)
        else:
            return self.c*self.f.gradient(x1, x2)


    def dimension(self):
        return self.f.dimension()
       
        
    def getHyperparameter(self):
        return self.f.getHyperparameter()
        

    def getHyperparameterLowerLimit(self):
        return self.f.getHyperparameterLowerLimit()


    def getHyperparameterUpperLimit(self):
        return self.f.getHyperparameterUpperLimit()


    def getVariableHyperparameter(self):
        return self.f.getVariableHyperparameter()


    def setHyperparameter(self, p):
        self.f.setHyperparameter(p)


    def setHyperparameterLowerLimit(self, pmin):
        self.f.setHyperparameterLowerLimit(pmin)


    def setHyperparameterUpperLimit(self, pmax):
        self.f.setHyperparameterLowerLimit(pmax)


    def setVariableHyperparameter(self, p):
        self.f.setVariableHyperparameter(p)


    def hyperparameterDimension(self):
        return self.f.hyperparameterDimension()
    

    def variableHyperparameterDimension(self):
        return self.f.variableHyperparameterDimension()
    
    
    def variableHyperparameterLowerBound(self):
        return self.f.variableHyperparameterLowerBound()

        
    def variableHyperparameterUpperBound(self):
        return self.f.variableHyperparameterUpperBound()


class kernelSquaredExponential(covarianceKernel):


    def __init__(self, d, p=None, pmin=None, pmax=None, M=None, variableHyperparameter=None):
        self.d = d
        hd = d+1
        if variableHyperparameter is None:
            self.variableHyperparameter = hd*[True]
        else:
            self.variableHyperparameter = variableHyperparameter
        if p is None:
            raise ValueError('Hyperparameters are needed')
        else:
            self.p = np.array(p)
        if pmin is None:
            self.pmin = np.array([small]*hd)
        else:
            self.pmin = np.array(pmin)
        if pmax is None:
            self.pmax = np.array([float('Inf')]*hd)
        else:
            self.pmax = np.array(pmax)
        self.M = M
            

    def evaluate(self, x1, x2=None):
        d = self.dimension()
        x1l = np.array(x1)
        nx1 = x1l.size/d
        x1l = np.reshape(x1l, (d, nx1))
        if self.M is not None:
            x1l = np.matmul(self.M, x1l)
        
        if x2 is None:
            for i in range(d):
                x1l[i] = x1l[i]*self.p[i+1]

            r2 = dist.pdist(x1l.T, 'sqeuclidean')
            r2 = dist.squareform(r2)
        else:
            x2l = np.array(x2)
            nx2 = x2l.size/d
            x2l = np.reshape(x2l, (d, nx2))
            if self.M is not None:
                x2l = np.matmul(self.M, x2l)
            
            for i in range(d):
                x1l[i] = x1l[i]*self.p[i+1]
                x2l[i] = x2l[i]*self.p[i+1]
                
            r2 = dist.cdist(x1l.T, x2l.T, 'sqeuclidean')

        return self.p[0]*np.exp(-0.5*r2)


    def evaluateDiagonal(self, x):
        d = self.dimension()
        nx = x.size/d
        
        return self.p[0]*np.ones(nx)
        
    
    def gradient(self, x1, x2=None):
        d = self.dimension()
        nv = self.variableHyperparameterDimension()
        
        if nv > 0:
            x1l = np.array(x1)
            nx1 = x1l.size/d
            x1l = np.reshape(x1l, (d, nx1))
            if self.M is not None:
                x1l = np.matmul(self.M, x1l)
         
            if x2 is None:
                dK = np.zeros((nx1, nx1, nv))
                if d == 1:
                    r2 = dist.pdist(x1l.T, 'sqeuclidean')
                    r2 = dist.squareform(r2)
                else:
                    x1s = np.zeros(np.shape(x1l))
                    for i in range(d):
                        x1s[i] = x1l[i]*self.p[i+1]
                    r2s = dist.pdist(x1s.T, 'sqeuclidean')
                    r2s = dist.squareform(r2s)
            else:
                x2l = np.array(x2)
                nx2 = x2l.size/d
                x2l = np.reshape(x2l, (d, nx2))
                if self.M is not None:
                    x2l = np.matmul(self.M, x2l)

                dK = np.zeros((nx1, nx2, nv))
                
                if d == 1:
                    r2 = dist.cdist(x1l.T, x2l.T, 'sqeuclidean')
                else:
                    x1s = np.zeros(np.shape(x1l))
                    x2s = np.zeros(np.shape(x2l))
                    for i in range(d):
                        x1s[i] = x1l[i]*self.p[i+1]
                        x2s[i] = x2l[i]*self.p[i+1]
                    r2s = dist.cdist(x1s.T, x2s.T, 'sqeuclidean')

            if d == 1:
                aux = np.exp(-0.5*r2*(self.p[1]**2))
                ct = 0
                if self.variableHyperparameter[0]:
                    dK[:, :, 0] = aux
                    ct += 1
                if self.variableHyperparameter[1]:
                    dK[:, :, ct] = -self.p[0]*self.p[1]*r2*aux
            else:
                aux = np.exp(-0.5*r2s)
                ct = 0
                if self.variableHyperparameter[0]:
                    dK[:, :, 0] = aux
                    ct += 1
                for i in range(d):
                    if self.variableHyperparameter[i+1]:
                        x1i = np.reshape(x1l[i, :], (1, nx1))
                        x2i = np.reshape(x2l[i, :], (1, nx2))
                        r2i = dist.cdist(x1i.T, x2i.T, 'sqeuclidean')
                        dK[:, :, ct] = -self.p[0]*self.p[i+1]*r2i*aux
                        ct += 1
           
            return dK
        else:
            x1l = np.array(x1)
            nx1 = x1l.size/d
            if x2 is None:
                dK = np.zeros((nx1, nx1))
            else:
                x2l = np.array(x2)
                nx2 = x2l.size/d
                dK = np.zeros((nx1, nx2))

        return dK


    def hyperparameterDimension(self):
        return self.dimension() + 1
                

class kernelMatern32(covarianceKernel):


    def __init__(self, d, p=None, pmin=None, pmax=None, M=None, variableHyperparameter=None):
        self.d = d
        hd = d+1
        if variableHyperparameter is None:
            self.variableHyperparameter = hd*[True]
        else:
            self.variableHyperparameter = variableHyperparameter
        if p is None:
            raise ValueError('Hyperparameters are needed')
        else:
            self.p = np.array(p)
        if pmin is None:
            self.pmin = np.array([small]*hd)
        else:
            self.pmin = np.array(pmin)
        if pmax is None:
            self.pmax = np.array([float('Inf')]*hd)
        else:
            self.pmax = np.array(pmax)
        self.M = M
            

    def evaluate(self, x1, x2=None):
        d = self.dimension()
        x1l = np.array(x1)
        nx1 = x1l.size/d
        x1l = np.reshape(x1l, (d, nx1))
        if self.M is not None:
            x1l = np.matmul(self.M, x1l)         

        if x2 is None:
            for i in range(d):
                x1l[i] = x1l[i]*self.p[i+1]

            r = dist.pdist(x1l.T, 'euclidean')
            r = dist.squareform(r)
        else:
            x2l = np.array(x2)
            nx2 = x2l.size/d
            x2l = np.reshape(x2l, (d, nx2))
            if self.M is not None:
                x2l = np.matmul(self.M, x2l)
            
            for i in range(d):
                x1l[i] = x1l[i]*self.p[i+1]
                x2l[i] = x2l[i]*self.p[i+1]
                
            r = dist.cdist(x1l.T, x2l.T, 'euclidean')


        return self.p[0]*(1. + np.sqrt(3.)*r)*np.exp(-np.sqrt(3.)*r)


    def evaluateDiagonal(self, x):
        d = self.dimension()
        nx = x.size/d
        
        return self.p[0]*np.ones(nx)
        
    
    def gradient(self, x1, x2=None):
        d = self.dimension()
        nv = self.variableHyperparameterDimension()
        
        if nv > 0:
            x1l = np.array(x1)
            nx1 = x1l.size/d
            x1l = np.reshape(x1l, (d, nx1))
            if self.M is not None:
                x1l = np.matmul(self.M, x1l)
            
            if x2 is None:
                dK = np.zeros((nx1, nx1, nv))
                if d == 1:
                    r = dist.pdist(x1l.T, 'euclidean')
                    r = dist.squareform(r)
                else:
                    x1s = np.zeros(np.shape(x1l))
                    for i in range(d):
                        x1s[i] = x1l[i]*self.p[i+1]
                    rs = dist.pdist(x1s.T, 'euclidean')
                    rs = dist.squareform(rs)
            else:
                x2l = np.array(x2)
                nx2 = x2l.size/d
                x2l = np.reshape(x2l, (d, nx2))
                if self.M is not None:
                    x2l = np.matmul(self.M, x2l)
                
                dK = np.zeros((nx1, nx2, nv))

                if d == 1:
                    r = dist.cdist(x1l.T, x2l.T, 'euclidean')
                else:
                    x1s = np.zeros(np.shape(x1l))
                    x2s = np.zeros(np.shape(x2l))
                    for i in range(d):
                        x1s[i] = x1l[i]*self.p[i+1]
                        x2s[i] = x2l[i]*self.p[i+1]
                    rs = dist.cdist(x1s.T, x2s.T, 'euclidean')

            if d == 1:
                aux = np.exp(-np.sqrt(3.)*r*self.p[1])
                ct = 0
                if self.variableHyperparameter[0]:
                    dK[:, :, 0] = (1. + np.sqrt(3.)*r*self.p[1])*aux
                    ct += 1
                if self.variableHyperparameter[1]:
                    dK[:, :, ct] = -3.*self.p[0]*self.p[1]*np.power(r, 2)*aux
            else:
                aux = np.exp(-np.sqrt(3.)*rs)
                ct = 0
                if self.variableHyperparameter[0]:
                    dK[:, :, 0] = (1. + np.sqrt(3.)*rs)*aux
                    ct += 1
                for i in range(d):
                    if self.variableHyperparameter[i+1]:
                        x1i = np.reshape(x1l[i, :], (1, nx1))
                        x2i = np.reshape(x2l[i, :], (1, nx2))
                        r2i = dist.cdist(x1i.T, x2i.T, 'sqeuclidean')
                        dK[:, :, ct] = -3.*self.p[0]*self.p[i+1]*r2i*aux
                        ct += 1
        else:
            x1l = np.array(x1)
            nx1 = x1l.size/d
            if x2 is None:
                dK = np.zeros((nx1, nx1))
            else:
                x2l = np.array(x2)
                nx2 = x2l.size/d
                dK = np.zeros((nx1, nx2))

        return dK       
    

    def hyperparameterDimension(self):
        return self.dimension() + 1


class kernelMatern52(covarianceKernel):


    def __init__(self, d, p=None, pmin=None, pmax=None, M=None, variableHyperparameter=None):
        self.d = d
        hd = d+1
        if variableHyperparameter is None:
            self.variableHyperparameter = hd*[True]
        else:
            self.variableHyperparameter = variableHyperparameter
        if p is None:
            raise ValueError('Hyperparameters are needed')
        else:
            self.p = np.array(p)
        if pmin is None:
            self.pmin = np.array([small]*hd)
        else:
            self.pmin = np.array(pmin)
        if pmax is None:
            self.pmax = np.array([float('Inf')]*hd)
        else:
            self.pmax = np.array(pmax)
        self.M = M
            

    def evaluate(self, x1, x2=None):
        d = self.dimension()
        x1l = np.array(x1)
        nx1 = x1l.size/d
        x1l = np.reshape(x1l, (d, nx1))
        if self.M is not None:
            x1l = np.matmul(self.M, x1l)         
         
        if x2 is None:
            for i in range(d):
                x1l[i] = x1l[i]*self.p[i+1]

            r = dist.pdist(x1l.T, 'euclidean')
            r = dist.squareform(r)
        else:
            x2l = np.array(x2)
            nx2 = x2l.size/d
            x2l = np.reshape(x2l, (d, nx2))
            if self.M is not None:
                x2l = np.matmul(self.M, x2l)
            
            for i in range(d):
                x1l[i] = x1l[i]*self.p[i+1]
                x2l[i] = x2l[i]*self.p[i+1]
                
            r = dist.cdist(x1l.T, x2l.T, 'euclidean')


        return self.p[0]*(1. + np.sqrt(5.)*r + (5./3.)*np.power(r, 2))*np.exp(-np.sqrt(5.)*r)


    def evaluateDiagonal(self, x):
        d = self.dimension()
        nx = x.size/d
        
        return self.p[0]*np.ones(nx)
        
    
    def gradient(self, x1, x2=None):
        d = self.dimension()
        nv = self.variableHyperparameterDimension()
        
        if nv > 0:
            x1l = np.array(x1)
            nx1 = x1l.size/d
            x1l = np.reshape(x1l, (d, nx1))
            if self.M is not None:
                x1l = np.matmul(self.M, x1l)         
             
            if x2 is None:
                dK = np.zeros((nx1, nx1, nv))
                if d == 1:
                    r = dist.pdist(x1l.T, 'euclidean')
                    r = dist.squareform(r)
                else:
                    x1s = np.zeros(np.shape(x1l))
                    for i in range(d):
                        x1s[i] = x1l[i]*self.p[i+1]
                    rs = dist.pdist(x1s.T, 'euclidean')
                    rs = dist.squareform(rs)
            else:
                x2l = np.array(x2)
                nx2 = x2l.size/d
                x2l = np.reshape(x2l, (d, nx2))
                if self.M is not None:
                    x2l = np.matmul(self.M, x2l)
                
                dK = np.zeros((nx1, nx2, nv))

                if d == 1:
                    r = dist.cdist(x1l.T, x2l.T, 'euclidean')
                else:
                    x1s = np.zeros(np.shape(x1l))
                    x2s = np.zeros(np.shape(x2l))
                    for i in range(d):
                        x1s[i] = x1l[i]*self.p[i+1]
                        x2s[i] = x2l[i]*self.p[i+1]
                    rs = dist.cdist(x1s.T, x2s.T, 'euclidean')

            if d == 1:
                aux = np.exp(-np.sqrt(5.)*r*self.p[1])
                r2 = np.power(r, 2)
                ct = 0
                if self.variableHyperparameter[0]:
                    dK[:, :, 0] = (1. + np.sqrt(5.)*r*self.p[1] + (5./3.)*r2*(self.p[1]**2))*aux
                    ct += 1
                if self.variableHyperparameter[1]:
                    dK[:, :, ct] = -(5./3.)*(1. + np.sqrt(5.)*r*self.p[1])*self.p[0]*self.p[1]*r2*aux
            else:
                aux = np.exp(-np.sqrt(5.)*rs)
                ct = 0
                if self.variableHyperparameter[0]:
                    dK[:, :, 0] = (1. + np.sqrt(5.)*rs + (5./3.)*np.power(rs, 2))*aux
                    ct += 1
                aux = -(5./3.)*(1. + np.sqrt(5.)*rs)*self.p[0]*aux
                for i in range(d):
                    if self.variableHyperparameter[i+1]:
                        x1i = np.reshape(x1l[i, :], (1, nx1))
                        x2i = np.reshape(x2l[i, :], (1, nx2))
                        r2i = dist.cdist(x1i.T, x2i.T, 'sqeuclidean')
                        dK[:, :, ct] = self.p[i+1]*r2i*aux
                        ct += 1
        else:
            x1l = np.array(x1)
            nx1 = x1l.size/d
            if x2 is None:
                dK = np.zeros((nx1, nx1))
            else:
                x2l = np.array(x2)
                nx2 = x2l.size/d
                dK = np.zeros((nx1, nx2))

        return dK       
    

    def hyperparameterDimension(self):
        if self.variableHyperparameter:
            return self.dimension() + 1
        else:
            return 0

                
class kernelNoise(covarianceKernel):
    

    def __init__(self, d, p=None, pmin=small, pmax=float('Inf'), variableHyperparameter=[True]):
        self.d = d
        self.variableHyperparameter = variableHyperparameter
        if p is None:
            raise ValueError('Hyperparameter is needed')
        else:
            self.p = np.array([p])
        
        self.pmin = np.array([pmin])
        self.pmax = np.array([pmax])


    def evaluate(self, x1, x2=None):
        d = self.dimension()
        x1l = np.array(x1)
        nx1 = x1l.size/d
        if x2 is None:
            return self.p*np.identity(nx1)
        else:
            x2l = np.array(x2)
            nx2 = x2l.size/d
            if nx1 == nx2:
                if np.all(np.equal(x1l, x2l)):
                    return self.p*np.identity(nx1)
                else:
                    return np.zeros((nx1, nx2))
            else:
                return np.zeros((nx1, nx2))
            

    def gradient(self, x1, x2=None):
        d = self.dimension()
        x1l = np.array(x1)
        nx1 = x1l.size/d
        if self.variableHyperparameter[0]:
            if x2 is None:
                dK = np.zeros((nx1, nx1, 1))
                dK[:, :, 0] = np.identity(nx1)
            else:
                x2l = np.array(x2)
                nx2 = x2l.size/d
                dK = np.zeros((nx1, nx2, 1))
                if nx1 == nx2:
                    if np.all(np.equal(x1l, x2l)):
                        dK[:, :, 0] = np.identity(nx1)
        else:
            if x2 is None:
                dK = np.zeros((nx1, nx1))
            else:
                x2l = np.array(x2)
                nx2 = x2l.size/d
                dK = np.zeros((nx1, nx2))
        
        return dK
        
        
    def hyperparameterDimension(self):
        return 1
