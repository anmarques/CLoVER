'''
This code is part of the software CLoVER.py
------------------------------------------------------------
Implementation of mean function class
------------------------------------------------------------
Distributed under the MIT License (see LICENSE.md)
Copyright 2018 Alexandre Marques, Remi Lam, and Karen Willcox
'''

import numpy as np

class meanFunction(object):

    def __init__(self, d, p=[]):
        self.d = d;
        hd = self.hyperparameterDimension()
        if hd == 0:
            self.p = []
        elif hd == 1:
            self.p = np.array([p])
        else:
            self.p = np.array(p)
    

    def __add__(self, other):
        if isinstance(other, meanFunction):
            return meanAdd(self, other)
        else:
            raise ValueError('Mean function object can only be added to other mean function object')
    
    
    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self + other
            
            
    def __mul__(self, other):
        if type(other) is int or type(other) is float or callable(other):    
            return meanMul(self, other)
        else:
            raise ValueError('Mean function object can only be multiplied by integer, float, or method')
    
    
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


class meanAdd(meanFunction):

    def __init__(self, f1, f2):
    
        d1 = f1.dimension()
        d2 = f2.dimension()
        
        if d1 == d2:
            self.f1 = f1
            self.f2 = f2
        else:
            raise ValueError('Added mean function must be defined over the same number of dimensions')
            
    
    def evaluate(self, x):
        return self.f1.evaluate(x) + self.f2.evaluate(x)
       
        
    def gradient(self, x):
        dm = []
        if self.f1.hyperparameterDimension() > 0:
            dm = self.f1.gradient(x)

        if self.f2.hyperparameterDimension() > 0:
            dm2 = self.f2.gradient(x)
            dm = np.concatenate((dm, dm2), axis=1)
        
        return dm

            
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
            self.f1.setHyperparameter(p[:n1]);
            self.f2.setHyperparameter(p[n1:]);
        elif n2 > 0:
            self.f2.setHyperparameter = p;


    def hyperparameterDimension(self):
        return self.f1.hyperparameterDimension() + self.f2.hyperparameterDimension()
    

    def hyperparameterGuess(self, x, y):
        p = []
        if self.f1.hyperparameterDimension() > 0:
            p1 = self.f1.hyperparameterGuess(x, y)
            f = self.f1.copy()
            f.setHyperparameter(p1)
            y = y - f.evaluate(x)
            p = p1
            
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


class meanMul(meanFunction):

    def __init__(self, f, c):
    
        d = f.dimension()
        self.f = f
        self.c = c
            
    
    def evaluate(self, x):
        if callable(self.c):
            return self.c(x, *args)*self.f.evaluate(x)
        else:
            return self.c*self.f.evaluate(x)
       

    def gradient(self, x):
        if callable(self.c):
            return self.c(x)*self.f.gradient(x)
        else:
            return self.c*self.f.gradient(x)
    
       
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


class meanZero(meanFunction):
    
    def evaluate(self, x):
        nx = x.size/self.dimension()
        return np.zeros(nx)
                
        
    def hyperparameterDimension(self):
        return 0


class meanConstant(meanFunction):

    def evaluate(self, x):
        nx = x.size/self.dimension()
        return self.p*np.ones(nx)
        
        
    def gradient(self, x):
        nx = x.size/self.dimension()
        return np.ones((nx, 1)) 
        
        
    def hyperparameterDimension(self):
        return 1
        
    
    def hyperparameterLowerBound(self):
        return np.array([-float('inf')])
        

    def hyperparameterUpperBound(self):
        return np.array([float('inf')])

        
class meanLinear(meanFunction):

    def evaluate(self, x):
        d = self.dimension()
        if d > 1:
            nx = x.size/d
            x = np.reshape(x, (d, nx))
            
        if d == 1:
            return self.p[0] + x*self.p[1]
        else:
            return self.p[0] + np.dot(x.T, self.p[1:]).flatten()
       
       
    def gradient(self, x):
        d = self.dimension()
        nx = x.size/d
        return np.ones((nx, d+1))

    
    def hyperparameterDimension(self):
        return self.dimension() + 1
        
    
    def hyperparameterLowerBound(self):
        return np.array([-float('inf')]*(self.dimension() + 1))
        

    def hyperparameterUpperBound(self):
        return np.array([float('inf')]*(self.dimension() + 1))
    
