Implicit steepest descent algorithm for optimization with orthogonality constraints

-------------------------------------------------------------------------
1. Problems and solvers

The package contains codes for the following two problems:
(1) The heterogeneous quadratics minimization problem

   min F(X) = \sum_{i=1}^{N} Tr[X_i'*A_i*X_i], s.t., X'*X = I, where X is a n-by-p matrix

   where X_i denotes the i–th column of X, and the Ai’s are n-by-n given real symmetric matrices.

     Solver: ImplicitSD_stiefel.m, OptStiefelCGC.m

     Solver demo: demo_HeterogeneousQuadratics.m

(2) The Rayleigh quotient maximization problem:
   	
   max F(x) = x'*A*x, s.t., ||x||_2 = 1, 
    where x is a vector with real entries.

     Solver: ImplicitSd_sphere.m, power_method.m

     Solver demo: demo_Rayleigh.m

-------------------------------------------------------------------------
2. Reference

 Harry Oviedo. Implicit steepest descent algorithm for optimization 
 with orthogonality constraints, 2020. 

-------------------------------------------------------------------------
3. The Author

 We hope that the package is useful for your application.  If you have
 any bug reports or comments, please feel free to email one of the
 toolbox authors:

   Harry Oviedo, harry.oviedo@cimat.mx

 Enjoy!
 Harry Oviedo

-------------------------------------------------------------------------
Copyright (C) 2020, Harry Oviedo
 
  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
 
  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

