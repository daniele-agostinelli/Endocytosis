# ObstacleProblem

This repository contains a Python code for solving an obstacle problem via the FEniCS project (version 2019.1.0). 

The obstacle problem consists in determining the shape of a cell membrane that needs to accommodate to a spherical obstacle, while being clamped to an underlying 
spherical surface. The code minimizes the Canham-Helfrich functional by using the penalty method and adopting a Monge parameterization. 
The problem is formulated in one dimension, by assuming axial symmetry.

The shape of the obstacle is defined in the class "Obstacle", and the variational formulation is defined and solved by means of the function SolveObstacleProblem().
Plor3D() and Plot() are used to plot and save the solutions to the obstacle problem. The former makes use of additional functions such as drawSphere() and drawHalfSphere().

Users can find a description of the addressed problem in the paper "Morphological control of receptor-mediated endocytosis" by Agostinelli et al. (2021).
