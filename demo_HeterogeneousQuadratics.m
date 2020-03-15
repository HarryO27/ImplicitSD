function demo_HeterogeneousQuadratics(n, p)
%-------------------------------------------------------------------------
% A demo for solving the heterogeneous quadratics minimization problem
%
%   min F(X) = \sum_{i=1}^{N} Tr[X_i'*A_i*X_i], s.t., X'*X = I, where X is a n-by-p matrix
%
%  where X_i denotes the i–th column of X, and the Ai’s are n-by-n given real symmetric matrices.
%-------------------------------------------------------------------------
% 
% Example1:
% demo_HeterogeneousQuadratics ()
% 
% Example2:
% demo_HeterogeneousQuadratics (300)
% 
% Example3:
% demo_HeterogeneousQuadratics (1000,10)
%
% Reference:
% ----------
% Harry Oviedo
% "Implicit steepest descent algorithm for optimization with orthogonality constraints"
%
% Authors:
% ----------
% Harry Oviedo <harry.oviedo@cimat.mx>
% Date: 15-March-2020
%-------------------------------------------------------------------------

clc
if nargin < 2  
    p = 10;
end
if nargin < 1
    n = 1000;   
end

%-------------------------------------------------------------------------
% Generating the problem 
%-------------------------------------------------------------------------
% This example corresponds to Experiment 2 of subsection 5.2 in 
% "Implicit steepest descent algorithm for optimization with orthogonality 
%  constraints"

A = zeros(n,n,p);
for j = 1:p
    Bj = 0.1*randn(n);
    A(:,:,j) = diag(((j-1)*n + 1)/p : 1/p : (j*n)/p) + Bj + Bj'; 
end

% Generamos al iterado Inicial X_0:
X_0 = randn(n,p);
[X0,~] = qr(X_0,0);


%-------------------------------------------------------------------------
% Solving the problem with our solver 
%-------------------------------------------------------------------------
opts.record = 0;
opts.mxitr  = 3000;
opts.gtol = 1e-5;

%--- Our solver ---
% Implicit Steepest Descent Method:
[X1, out1, Grad1, F_eval1] = ImplicitSD_stiefel(X0, @fun, opts, A);

% Solving with the Riemannian Conjugate Gradient method of Zhu(CG):
% Algoritmo 1.a developed in "A riemannian conjugate gradient method for 
% optimization on the stiefel manifold. Computational Optimization and 
% Applications, 67(1):73–110, 2017." This solver is available in  
% http://www. optimization-online.org/DB_HTML/2016/09/5617.html
opts.vt = 1;
[X2, out2, Grad2, F_eval2] = OptStiefelCGC(X0, @fun, opts, A);

% Solving with the Riemannian Conjugate Gradient method of Zhu(CG):
% Algoritmo 1.b developed in "A riemannian conjugate gradient method for 
% optimization on the stiefel manifold. Computational Optimization and 
% Applications, 67(1):73–110, 2017." This solver is available in  
% http://www. optimization-online.org/DB_HTML/2016/09/5617.html
opts.vt = 2;
[X3, out3, Grad3, F_eval3] = OptStiefelCGC(X0, @fun, opts, A);

%-------------------------------------------------------------------------
% Printing the results
%-------------------------------------------------------------------------
fprintf('---------------------------------------------------\n')
fprintf('Results for Implicit--SD method 1\n') 
fprintf('---------------------------------------------------\n')
fprintf('   Obj. function = %7.6e\n',  out1.fval);
fprintf('   Gradient norm = %7.6e \n', out1.nrmG);
fprintf('   ||X^T*X-I||_F = %3.2e\n',  out1.feasi )
fprintf('   Iteration number = %d\n',  out1.itr); 
fprintf('   Cpu time (secs) = %3.4f  \n',  out1.time);
fprintf('   Number of evaluation(Obj. func) = %d\n',  out1.nfe); 
%-------------------------------------------------------------------------

%-------------------------------------------------------------------------
% Printing the results
%-------------------------------------------------------------------------
fprintf('---------------------------------------------------\n')
fprintf('Results for Riemannian CG method (Algoritmo 1.a of Zhu)\n') 
fprintf('---------------------------------------------------\n')
fprintf('   Obj. function = %7.6e\n',  out2.fval);
fprintf('   Gradient norm = %7.6e \n', out2.nrmG);
fprintf('   ||X^T*X-I||_F = %3.2e\n',  out2.feasi )
fprintf('   Iteration number = %d\n',  out2.itr); 
fprintf('   Cpu time (secs) = %3.4f  \n',  out2.time);
fprintf('   Number of evaluation(Obj. func) = %d\n',  out2.nfe); 
%-------------------------------------------------------------------------

%-------------------------------------------------------------------------
% Printing the results
%-------------------------------------------------------------------------
fprintf('---------------------------------------------------\n')
fprintf('Results for Riemannian CG method (Algoritmo 1.b of Zhu)\n') 
fprintf('---------------------------------------------------\n')
fprintf('   Obj. function = %7.6e\n',  out3.fval);
fprintf('   Gradient norm = %7.6e \n', out3.nrmG);
fprintf('   ||X^T*X-I||_F = %3.2e\n',  out3.feasi )
fprintf('   Iteration number = %d\n',  out3.itr); 
fprintf('   Cpu time (secs) = %3.4f  \n',  out3.time);
fprintf('   Number of evaluation(Obj. func) = %d\n',  out3.nfe); 
%-------------------------------------------------------------------------


hold on
plot(log(Grad1),'b','linewidth',2);
plot(log(Grad2),'r','linewidth',2);
plot(log(Grad3),'k','linewidth',2);
hold off
legend('Implicit-SD','CG1','CG2')
title('Iterations vs Gradient Norm','FontSize',15);
xlabel('Iterations','FontSize',15);
ylabel('Gradient Norm','FontSize',15);
set(gca,'FontSize',15)


% Objective Function (F) and Gradient (G) at X  
    function [F, G] = fun(X, A)
        F = 0;
        for i = 1:p
            Ai  = A(:,:,i);
            xi  = X(:,i);
            Axi = Ai*xi;
            F   = F + xi'*Axi;
            G(:,i) = 2*Axi;
        end
    end
end


