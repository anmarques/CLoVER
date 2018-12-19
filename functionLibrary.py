'''
This code is part of the software CLoVER.py
------------------------------------------------------------
Library of test functions
------------------------------------------------------------
Distributed under the MIT License (see LICENSE.md)
Copyright 2018 Alexandre Marques, Remi Lam, and Karen Willcox
'''

import numpy as np

def branin(x):
    a = 1.
    b = 1.25/(np.pi**2)
    c = 5./np.pi
    r = 6.
    s = 10.
    t = 0.125/(np.pi)
    nx = x.size/2
    if nx == 1:
        return a*np.power(x[1] - b*np.power(x[0], 2) + c*x[0] - r, 2) + s*(1. - t)*np.cos(x[0]) + s
    else:
        return a*np.power(x[1, :] - b*np.power(x[0, :], 2) + c*x[0, :] - r, 2) + s*(1. - t)*np.cos(x[0, :]) + s


def multiModal(x):
    nx = x.size/2
    if nx == 1:
        return (np.power(x[0], 2) + 4.)*(x[1] - 1.)/20. - np.sin(2.5*x[0]) - 2.
    else:
        return (np.power(x[0, :], 2) + 4.)*(x[1, :] - 1.)/20. - np.sin(2.5*x[0, :]) - 2.
