'''
This code is part of the software CLoVER.py
------------------------------------------------------------
Implementation of mean function class
------------------------------------------------------------
Distributed under the MIT License (see LICENSE.md)
Copyright 2018 Alexandre Marques, Remi Lam, and Karen Willcox
'''

import numpy as np
import copy

class meanFunction(object):

    def __init__(self, d, p=None, pmin=None, pmax=None, variableHyperparameter=None):
        self.d = d
        self.p = p
        self.pmin = pmin
        self.pmax = pmax
        self.variableHyperparameter = variableHyperparameter
    

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

    
    def dimension(self):
        return self.d
    
    
    def getHyperparameter(self):
        if self.p is None:
            return None
        else:
            return copy.deepcopy(self.p)
    

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


class meanAdd(meanFunction):

    def __init__(self, f1, f2):
    
        d1 = f1.dimension()
        d2 = f2.dimension()
        
        if d1 == d2:
            self.f1 = copy.deepcopy(f1)
            self.f2 = copy.deepcopy(f2)
        else:
            raise ValueError('Added mean function must be defined over the same number of dimensions')
            
        if self.f1.variableHyperparameter is None:
            self.variableHyperameter = self.f2.variableHyperparameter
        elif self.f2.variableHyperparameter is None:
            self.variableHyperameter = self.f1.variableHyperparameter
        else:
            self.variableHyperameter = self.f1.variableHyperparameter + self.f2.variableHyperparameter


    def evaluate(self, x):
        return self.f1.evaluate(x) + self.f2.evaluate(x)
       
        
    def gradient(self, x):
        dm = []
        if self.f1.variableHyperparameterDimension() > 0:
            dm = self.f1.gradient(x)

        if self.f2.variableHyperparameterDimension() > 0:
            dm2 = self.f2.gradient(x)
            dm = np.concatenate((dm, dm2), axis=1)
        
        return dm

            
    def dimension(self):
        return self.f1.dimension()
       
        
    def getHyperparameter(self):
        p = None
        if self.f1.hyperparameterDimension() > 0:
            p = self.f1.getHyperparameter()

        if self.f2.hyperparameterDimension() > 0:
            p2 = self.f2.getHyperparameter()
            if p is None:
                p = p2
            else:
                p = np.concatenate((p, p2))
        
        return p
        
        
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
            self.f1.setHyperparameter(p[:n1]);
            if n2 > 0:
                self.f2.setHyperparameter(p[n1:]);
        elif n2 > 0:
            self.f2.setHyperparameter = p


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


class meanMul(meanFunction):

    def __init__(self, f, c):
        self.f = copy.deepcopy(f)
        self.c = copy.deepcopy(c)
        
    
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
        

    def getVariableHyperparameter(self):
        return self.f.getVariableHyperparameter()


    def setHyperparameter(self, p):
        self.f.setHyperparameter(p);


    def setVariableHyperparameter(self, p):
        self.f.setVariableHyperparameter(p);


    def hyperparameterDimension(self):
        return self.f.hyperparameterDimension()
    

    def variableHyperparameterDimension(self):
        return self.f.variableHyperparameterDimension()
    
    
    def variableHyperparameterLowerBound(self):
        return self.f.variableHyperparameterLowerBound()


    def variableHyperparameterUpperBound(self):
        return self.f.variableHyperparameterUpperBound()


class meanZero(meanFunction):

    def __init__(self, d):
        self.d = d
        self.p = None
        self.variableHyperparameter = None
    
    def evaluate(self, x):
        nx = x.size/self.dimension()
        return np.zeros(nx)
                
        
    def hyperparameterDimension(self):
        return 0


class meanConstant(meanFunction):

    def __init__(self, d, p=None, pmin=None, pmax=None, variableHyperparameter=[True]):
        self.d = d
        if p is None:
            raise ValueError('Hyperparameter is needed')
        else:
            self.p = np.array([p])
        
        if pmin is None:
            self.pmin = np.array([-float('Inf')])
        else:
            self.pmin = np.array(pmin)
        
        if pmax is None:
            self.pmax = np.array([float('Inf')])
        else:
            self.pmax = np.array(pmax)
        self.variableHyperparamer = variableHyperparameter
        
        
    def evaluate(self, x):
        nx = x.size/self.dimension()
        return self.p*np.ones(nx)
        
        
    def gradient(self, x):
        if self.variableHyperparamer[0]:
            nx = x.size/self.dimension()
            return np.ones((nx, 1)) 
        else:
            return np.zeros(nx) 
        
        
    def hyperparameterDimension(self):
        return 1
        
    
    def variableHyperparameterLowerBound(self):
        return np.array([-float('inf')])
        

    def variableHyperparameterUpperBound(self):
        return np.array([float('inf')])

        
class meanLinear(meanFunction):

    
    def __init__(self, d, p=None, pmin=None, pmax=None, variableHyperparameter=None):
        self.d = d
        hd = d + 1
        if variableHyperparameter is None:
            self.variableHyperparameter = hd*[True]
        else:
            self.variableHyperparameter = variableHyperparameter

        if p is None:
            raise ValueError('Hyperparameters are needed')
        else:
            self.p = np.array(p)
        
        if pmin is None:
            self.pmin = np.array([-float('Inf')]*hd)
        else:
            self.pmin = np.array(pmin)
        
        if pmax is None:
            self.pmax = np.array([float('Inf')]*hd)
        else:
            self.pmax = np.array(pmax)
            
    
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
        nv = self.variableHyperparameterDimension()
        if nv > 0:
            d = self.dimension()
            nx = x.size/d
            return np.ones((nx, nv))
        else:
            return np.zeros(nx)

    
    def hyperparameterDimension(self):
        return self.dimension() + 1
            
    
    def variableHyperparameterLowerBound(self):
        return np.array([-float('inf')]*(self.dimension() + 1))
        

    def variableHyperparameterUpperBound(self):
        return np.array([float('inf')]*(self.dimension() + 1))
    
