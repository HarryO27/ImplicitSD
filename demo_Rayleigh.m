function demo_Rayleigh(n)
%-------------------------------------------------------------------------
% A demo for solving the Rayleigh quotient maximization:
%   max F(x) = x'*A*x, s.t., ||x||_2 = 1, where x is a vector with real
%   entries.
%
%  This demo solves the eigenvalue problem by letting
%  F(x) = -0.5*(x'*A*x); where A is a given n-by-n symmetric and positive 
%  definite matrix.
%-------------------------------------------------------------------------
% 
% Example1:
% demo_eigenspace ()
% 
% Example2:
% demo_eigenspace (100)
% 
%
% Reference:
% ----------
% Harry Oviedo 
% "Implicit steepest descent algorithm for optimization with orthogonality constraints"
%
% Authors:
% ----------
% Harry Oviedo <harry.oviedo@cimat.mx>
% Date: 15-March-2019
%-------------------------------------------------------------------------

clc
if nargin < 1
    n = 1000;   
end

%-------------------------------------------------------------------------
% Generating the problem 
%-------------------------------------------------------------------------
A = randn(n,n);     A = A'*A;  
mu = rand;
A = A + mu*eye(n); 

% Generating the starting point:
x0 = ones(n,1);     x0 = x0/norm(x0);


%-------------------------------------------------------------------------
% Solving with MATLAB eigs
%-------------------------------------------------------------------------
t_ini = tic;
[v, D] = eigs(A,1);
tf = toc(t_ini);
[F, G] = fun(v, A);
NrmG = norm(G - v*(G'*v));
D = diag(D); feig = sum(D(1:1));

%-------------------------------------------------------------------------
% Solving the problem with our solver 
%-------------------------------------------------------------------------
opts.record = 0;
opts.mxitr  = 5000;
opts.gtol = 1e-5;

%--- Our solver ---
% Implicit Steepest Descent Method:
[X1, out1] = ImplicitSD_sphere(x0, @fun, opts, A);
out1.fval = -2*out1.fval; % convert the function value to the sum of eigenvalues

% Power Method
tolg = opts.gtol;
[lambda,v,out2] = power_method(A,x0,tolg);


%-------------------------------------------------------------------------
% Printing the results
%-------------------------------------------------------------------------
fprintf('---------------------------------------------------\n')
fprintf('Results for eigs matlab fucntion\n') 
fprintf('---------------------------------------------------\n')
fprintf('   Obj. function = %7.6e\n',  feig);
fprintf('   Gradient norm = %7.6e \n', NrmG);
fprintf('   abs(||x||_2 - 1) = %3.2e\n',  norm(v) )
fprintf('   Cpu time (secs) = %3.4f  \n',  tf);
%-------------------------------------------------------------------------


%-------------------------------------------------------------------------
% Printing the results
%-------------------------------------------------------------------------
fprintf('---------------------------------------------------\n')
fprintf('Results for Implicit Steepest Descent Method\n') 
fprintf('---------------------------------------------------\n')
fprintf('   Obj. function = %7.6e\n',  out1.fval);
fprintf('   Gradient norm = %7.6e \n', out1.nrmG);
fprintf('   abs(||x||_2 - 1) = %3.2e\n',  out1.feasi )
fprintf('   Iteration number = %d\n',  out1.itr); 
fprintf('   Cpu time (secs) = %3.4f  \n',  out1.time);
fprintf('   Number of evaluation(Obj. func) = %d\n',  out1.nfe); 
%-------------------------------------------------------------------------

%-------------------------------------------------------------------------
% Printing the results
%-------------------------------------------------------------------------
fprintf('---------------------------------------------------\n')
fprintf('Results for Power Method\n') 
fprintf('---------------------------------------------------\n')
fprintf('   Obj. function = %7.6e\n',  out2.fval);
fprintf('   Gradient norm = %7.6e \n', out2.nrmG);
fprintf('   abs(||x||_2 - 1) = %3.2e\n',  out2.feasi )
fprintf('   Iteration number = %d\n',  out2.itr); 
fprintf('   Cpu time (secs) = %3.4f  \n',  out2.time); 
%-------------------------------------------------------------------------


% Objective Function (F) and Gradient (G) at X  
    function [F, G] = fun(X, A)
        AX = A*X;
        F = -0.5*(X'*AX);
        G = -AX;
    end
end


