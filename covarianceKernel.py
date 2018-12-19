'''
This code is part of the software CLoVER.py
------------------------------------------------------------
Implementation of covariance kernel class
------------------------------------------------------------
Distributed under the MIT License (see LICENSE.md)
Copyright 2018 Alexandre Marques, Remi Lam, and Karen Willcox
'''

import numpy as np
import scipy.spatial.distance as dist

small = 1.e-8

class covarianceKernel(object):

    def __init__(self, d):
        self.d = d;
        self.p = []
    

    def __add__(self, other):
        if isinstance(other, covarianceKernel):
            return kernelAdd(self, other)
        else:
            raise ValueError('Covariance kernel object can only be added to other covariance kernel object')
    
    
    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self + other
            
            
    def __mul__(self, other):
        if type(other) is int or type(other) is float or callable(other):    
            return kernelMul(self, other)
        else:
            raise ValueError('Covariance kernel object can only be multiplied by integer, float, or method')
    
    
    def __rmul__(self, other):
        if other == 1:
            return self
        else:
            return self * other


    def copy(self):
        return 1*self

    
    def dimension(self):
        return self.d
    
    
    def getHyperparameter(self):
        if len(self.p) == 0:
            return []
        else:
            return self.p.copy()
    
    
    def setHyperparameter(self, p):
        self.p = np.array(p)


class kernelAdd(covarianceKernel):

    def __init__(self, f1, f2):
    
        d1 = f1.dimension()
        d2 = f2.dimension()
        
        if d1 == d2:
            self.f1 = f1
            self.f2 = f2
        else:
            raise ValueError('Added covariance kernel must be defined over the same number of dimensions')
            
    
    def evaluate(self, x1, x2):
        return self.f1.evaluate(x1, x2) + self.f2.evaluate(x1, x2)
       

    def evaluateDiagonal(self, x):
        if hasattr(self.f1, 'evaluateDiagonal'):
            f1 = self.f1.evaluateDiagonal(x)
        else:
            f1 = np.diagonal(self.f1.evaluate(x, []))

        if hasattr(self.f2, 'evaluateDiagonal'):
            f2 = self.f2.evaluateDiagonal(x)
        else:
            f2 = np.diagonal(self.f2.evaluate(x, []))
            
        return f1 + f2
       

    def gradient(self, x1, x2):
        dK = []
        if self.f1.hyperparameterDimension() > 0:
            dK = self.f1.gradient(x1, x2)

        if self.f2.hyperparameterDimension() > 0:
            dK2 = self.f2.gradient(x1, x2)
            dK = np.concatenate((dK, dK2), axis=2)
        
        return dK

            
    def dimension(self):
        return self.f1.dimension()
       
        
    def getHyperparameter(self):
        p = []
        if self.f1.hyperparameterDimension() > 0:
            p = self.f1.getHyperparameter()

        if self.f2.hyperparameterDimension() > 0:
            p2 = self.f2.getHyperparameter()
            if len(p) == 0:
                p = p2
            else:
                p = np.concatenate((p, p2))
        
        return p
        

    def setHyperparameter(self, p):
        n1 = self.f1.hyperparameterDimension()
        n2 = self.f2.hyperparameterDimension()
        if n1 > 0:
            self.f1.setHyperparameter(p[:n1])
            self.f2.setHyperparameter(p[n1:])
        elif n2 > 0:
            self.f2.setHyperparameter = p;


    def hyperparameterDimension(self):
        return self.f1.hyperparameterDimension() + self.f2.hyperparameterDimension()
    

    def hyperparameterGuess(self, x, y):
        p = []        
        if self.f1.hyperparameterDimension() > 0:
            p = self.f1.hyperparameterGuess(x, y)
            
        if self.f2.hyperparameterDimension() > 0:
            p2 = self.f2.hyperparameterGuess(x, y)
            p = np.concatenate((p, p2))
        
        return p
    
    
    def hyperparameterLowerBound(self):
        lb = []
        if self.f1.hyperparameterDimension() > 0:
            lb = self.f1.hyperparameterLowerBound()
            
        if self.f2.hyperparameterDimension() > 0:
            lb2 = self.f2.hyperparameterLowerBound()
            lb = np.concatenate((lb, lb2))
        
        return lb


    def hyperparameterUpperBound(self):
        ub = []
        if self.f1.hyperparameterDimension() > 0:
            ub = self.f1.hyperparameterUpperBound()
            
        if self.f2.hyperparameterDimension() > 0:
            ub2 = self.f2.hyperparameterUpperBound()
            ub = np.concatenate((ub, ub2))
        
        return ub


class kernelMul(covarianceKernel):

    def __init__(self, f, c):
    
        d = f.dimension()
        self.f = f
        self.c = c
            
    
    def evaluate(self, x1, x2):
        if callable(self.c):
            return self.c(x1, x2)*self.f.evaluate(x1, x2)
        else:
            return self.c*self.f.evaluate(x1, x2)
       
        
    def evaluateDiagonal(self, x):
        if callable(self.c):
            c = self.c(x1, x2)
        else:
            c = self.c
            
        if hasattr(self.f, 'evaluateDiagonal'):
            return c*self.f.evaluateDiagonal(x)
        else:
            return c*np.diagonal(self.f.evaluate(x, []))

            
    def gradient(self, x1, x2):
        if callable(self.c):
            return self.c(x1, x2)*self.f.gradient(x1, x2)
        else:
            return self.c*self.f.gradient(x1, x2)


    def dimension(self):
        return self.f.dimension()
       
        
    def getHyperparameter(self):
        return self.f.getHyperparameter()
        

    def setHyperparameter(self, p):
        self.f.setHyperparameter(p);


    def hyperparameterDimension(self):
        return self.f.hyperparameterDimension()
    

    def hyperparameterGuess(self, x, y):
        return self.f.hyperparameterGuess(x, y)
    
    
    def hyperparameterLowerBound(self):
        return self.f.hyperparameterLowerBound()

        
    def hyperparameterUpperBound(self):
        return self.f.hyperparameterUpperBound()


class kernelSquaredExponential(covarianceKernel):

    def evaluate(self, x1, x2):
        d = self.dimension()
        x1 = np.array(x1)
        nx1 = x1.size/d
        x1 = np.reshape(x1, (d, nx1))
        x2 = np.array(x2)
         
        scl = np.diag(self.p[1:])
        if x2.size > 0:
            x2 = np.array(x2)
            nx2 = x2.size/d
            x2 = np.reshape(x2, (d, nx2))
            
            for i in range(d):
                x1[i] = x1[i]/self.p[i+1]
                x2[i] = x2[i]/self.p[i+1]
                
            dist2 = dist.cdist(x1.T, x2.T, 'sqeuclidean')
        else:
            for i in range(d):
                x1[i] = x1[i]/self.p[i+1]

            dist2 = dist.pdist(x1.T, 'sqeuclidean')
            dist2 = dist.squareform(dist2)

        return (self.p[0]**2)*np.exp(-0.5*dist2)


    def evaluateDiagonal(self, x):
        d = self.dimension()
        nx = x.size/d
        
        return (self.p[0]**2)*np.ones(nx)
        
    
    def gradient(self, x1, x2):
        x1 = np.array(x1)
        x2 = np.array(x2)
        d = self.dimension()
        nx1 = x1.size/d
        nx2 = x2.size/d
        if d > 1:
            x1 = np.reshape(x1, (d, nx1))
            x2 = np.reshape(x2, (d, nx2))        

        dK = np.zeros((nx1, nx2, d + 1))
        for i in range(nx1):
            for j in range(nx2):
                if d == 1:
                    dist2 = ((x1[i] - x2[j])/self.p[1])**2
                else:
                    dist2 = 0.
                    for k in range(d):
                        dist2 += ((x1[k, i] - x2[k, j])/self.p[k+1])**2

                aux = np.exp(-0.5*dist2)
                dK[i, j, 0] = 2.*self.p[0]*aux
                if d == 1:
                    dK[i, j, 1] = (self.p[0]**2)*aux*((x1[i] - x2[j])**2)/(self.p[1]**3)
                else:
                    for k in range(d):
                        dK[i, j, k+1] = (self.p[0]**2)*aux*((x1[k, i] - x2[k, j])**2)/(self.p[k+1]**3)
       
        return dK
    

    def hyperparameterDimension(self):
        return self.dimension() + 1
        
        
    def hyperparameterGuess(self, x, y):
        d = self.dimension()
        x = np.array(x)
        if d > 1:
            nx = x.size/d
            x = np.reshape(x, (d, nx))
            
        p = np.zeros(d+1)
        p[0] = np.std(y)
        if d == 1:
            p[1] = np.max(x) - np.min(x)
        else:
            p[1:] = 0.1*(np.max(x, axis=1) - np.min(x, axis=1))
            
        return p
    
    
    def hyperparameterLowerBound(self):
        return np.array([small] + self.dimension()*[1.e-2])
        

    def hyperparameterUpperBound(self):
        return np.array([float('Inf')]*(self.dimension() + 1))
        
        
class kernelNoise(covarianceKernel):
    
    def evaluate(self, x1, x2):
        x1 = np.array(x1)
        x2 = np.array(x2)
        d = self.dimension()
        nx1 = x1.size/d
        nx2 = x2.size/d
        if nx2 > 0:
            if nx1 == nx2:
                if np.all(np.equal(x1, x2)):
                    return (self.p**2)*np.identity(nx1)
                else:
                    return np.zeros((nx1, nx2))
            else:
                return np.zeros((nx1, nx2))
        else:
            return (self.p**2)*np.identity(nx1)
            

    def gradient(self, x1, x2):
        x1 = np.array(x1)
        x2 = np.array(x2)
        d = self.dimension()
        nx1 = x1.size/d
        nx2 = x2.size/d
        dK = np.zeros((nx1, nx2, 1))
        if nx1 == nx2:
            if np.all(np.equal(x1, x2)):
                dK[:, :, 0] = 2*self.p*np.identity(nx1)
        
        return dK
        
        
    def hyperparameterDimension(self):
        return 1
        
        
    def hyperparameterGuess(self, x, y):
        return np.array([0.1*np.std(y)])
    
    
    def hyperparameterLowerBound(self):
        return np.array([small])


    def hyperparameterUpperBound(self):
        return np.array([float('Inf')])

