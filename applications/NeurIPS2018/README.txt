This folder contains an example of application of the software
CLoVER.py. This example corresponds to the multi-modal example
described in the paper

A.N. Marques, R.R. Lam, and K.E. Willcox,
Contour location via entropy reduction leveraging multiple information
sources,
Advances in Neural Information Processing Systems 31, 2018,
pp. 5222-5232.

The script multiModalExample_NeurIPS2018.py contains all the
definitions of the problem, and executes CLoVER 100 times. The results
are stored in the folder "runs". 

The script visualize.py allows the user to visualize how estimates
made by CLoVER evolve over the course of iterations. This script takes
one inline argument (integer between 0 and 99) corresponding to the
index of the execution to be visualized. The script creates plots that
show evaluations of the different information sources, contours of the
mean surrogate model of IS0, and the zero contour. The plots are
stored as png figures in the folder "visualization". One plot is
generated for every iteration. NOTE: the script deletes all content of
folder "visualization" before creating new plots.

The script computeArea.py postprocesses all 100 executions of CLoVER
to estimate the area in the region IS0 is positive. It records the
results in the folder "area". Each file in this folder contains as
many lines as iterations, and each line has 3 columns: iteration
number, cost, area estimate. This script also computes the error
with respect to a Monte Carlo estimate based on the true function.
The error statistics are shown in the figure "areaError.png".

----------------------------------------------------------------------
Distributed under the MIT License (see LICENSE.md)
Copyright 2018 Alexandre Marques, Remi Lam, and Karen Willcox
'''

