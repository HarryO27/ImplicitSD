function [X, out, Grad, F_eval]= ImplicitSD_stiefel(X, fun, opts, varargin)
%-------------------------------------------------------------------------
% Implicit steepest descent algorithm for optimization with orthogonality constraints
%
%   min F(X), S.t., X'*X = I_k, where X \in R^{n,k}
%
%   H = [G, X]*[X -G]'
%   U = 0.5*tau*[G, X];    V = [X -G]
%   X(tau) = X - 2*U * inv( I + V'*U ) * V'*X
%
%   -------------------------------------
%   U = -[G,X];  V = [X -G];  VU = V'*U;
%   X(tau) = X - tau*U * inv( I + 0.5*tau*VU ) * V'*X
%
%
% Input:
%           X --- n by k matrix such that X'*X = I
%         fun --- objective function and its gradient:
%                 [F, G] = fun(X,  data1, data2)
%                 F, G are the objective function value and gradient, repectively
%                 data1, data2 are addtional data, and can be more
%                 Calling syntax:
%                   [X, out]= ImplicitSD_stiefel(X0, @fun, opts, data1, data2);
%
%        opts --- option structure with fields:
%                 record = 0, no print out
%                 mxitr       max number of iterations
%                 gtol        stop control for the projected gradient
%   
% Output:
%           X --- solution
%         Out --- output information
%
% -------------------------------------
% For example, consider the eigenvalue problem F(X) = -0.5*Tr(X'*A*X);
%
% function demo
% 
% function [F, G] = fun(X,  A)
%   G = -(A*X);
%   F = 0.5*sum(dot(G,X,1));
% end
% 
% n = 1000; k = 6;
% A = randn(n); A = A'*A;
% opts.record = 0; %
% opts.mxitr  = 1000;
% opts.gtol = 1e-5;;
% 
% X0 = randn(n,k);    X0 = orth(X0);
% tic; [X, out]= ImplicitSD_stiefel(X0, @fun, opts, A); tsolve = toc;
% out.fval = -2*out.fval; % convert the function value to the sum of eigenvalues
% fprintf('\nOptM: obj: %7.6e, itr: %d, nfe: %d, cpu: %f, norm(XT*X-I): %3.2e \n', ...
%             out.fval, out.itr, out.nfe, tsolve, norm(X'*X - eye(k), 'fro') );
% 
% end
% -------------------------------------
%
% Reference: 
%  ZHarry Oviedo
%  "Implicit steepest descent algorithm for optimization with orthogonality constraints"
%
% Author: Harry Oviedo
%   Version 1.0 .... 2020/3
%-------------------------------------------------------------------------


%% Size information
if isempty(X)
    error('input X is an empty matrix');
else
    [n, k] = size(X);
end

if isfield(opts, 'gtol')
    if opts.gtol < 0 || opts.gtol > 1
        opts.gtol = 1e-4;
    end
else
    opts.gtol = 1e-4;
end

% parameters for control the linear approximation in line search
if isfield(opts, 'rho')
   if opts.rho < 0 || opts.rho > 1
        opts.rho = 1e-4;
   end
else
    opts.rho = 1e-4;
end

% factor for decreasing the step size in the backtracking line search
if isfield(opts, 'eta')
   if opts.eta < 0 || opts.eta > 1
        opts.eta = 0.1;
   end
else
    opts.eta = 0.2;
end

% parameters for updating C by HongChao, Zhang
if isfield(opts, 'gamma')
   if opts.gamma < 0 || opts.gamma > 1
        opts.gamma = 0.85;
   end
else
    opts.gamma = 0.85;
end

if isfield(opts, 'tau')
   if opts.tau < 0 || opts.tau > 1e3
        opts.tau = 1e-3;
   end
else
    opts.tau = 1e-3;
end

% parameters for the  nonmontone line search by Raydan
if ~isfield(opts, 'STPEPS')
    opts.STPEPS = 1e-10;
end

if isfield(opts, 'nt')
    if opts.nt < 0 || opts.nt > 100
        opts.nt = 5;
    end
else
    opts.nt = 5;
end

if isfield(opts, 'projG')
    switch opts.projG
        case {1,2}; otherwise; opts.projG = 1;
    end
else
    opts.projG = 1;
end

if isfield(opts, 'mxitr')
    if opts.mxitr < 0 || opts.mxitr > 2^20
        opts.mxitr = 1000;
    end
else
    opts.mxitr = 1000;
end

if ~isfield(opts, 'record')
    opts.record = 0;
end


%-------------------------------------------------------------------------------
% copy parameters
gtol = opts.gtol;
rho  = opts.rho;
eta   = opts.eta;
gamma = opts.gamma;

nt = opts.nt;   
invA = true; if k < n/2; invA = false;  eye2k = eye(2*k); end

%% Initial function value and gradient
% prepare for iterations
[F,  G] = feval(fun, X , varargin{:});  out.nfe = 1;  
GX = G'*X;

if invA
    GXT = G*X';  A = (GXT - GXT');    
else
    if opts.projG == 1
        U =  [G, X];    V = [X, -G];       VU = V'*U;
    elseif opts.projG == 2
        GB = G - X*(0.5*(X'*G));
        U =  [GB, X];    V = [X, -GB];       VU = V'*U;
    end
    VX = V'*X;
end
dtX = G - X*GX;     nrmG  = norm(dtX, 'fro');
Q = 1; Cval = F;    tau = opts.tau;

Grad  = zeros(opts.mxitr+1,1);      Grad(1)   = nrmG;
F_eval = zeros(opts.mxitr+1,1);     F_eval(1) = F; 

%% Print iteration header if debug == 1
if (opts.record == 1)
    fid = 1;
    fprintf(fid, '----------- Gradient Method with Line search ----------- \n');
    fprintf(fid, '%4s %8s %8s %10s %10s\n', 'Iter', 'tau', 'F(X)', 'nrmG', 'XDiff');
end

%% main iteration
tstart = tic;
for itr = 1 : opts.mxitr
    XP = X;      dtXP = dtX; 
    
    % scale step size
    nls = 1; deriv = rho*nrmG^2; 
    
    while 1
        % Update Scheme
        if invA
           X = linsolve(eye(n) + tau*A, XP); 
        else
           [aa, ~] = linsolve(eye2k + tau*VU, VX);
           X = XP - U*(tau*aa); 
        end
        L = chol(X'*X);
        X = X/L;
               
        % calculate G, F
        [F,G] = feval(fun, X, varargin{:});
        out.nfe = out.nfe + 1;
        
        if F <= Cval - tau*deriv || nls >= nt
            break;
        end
        tau = eta*tau;          nls = nls + 1;
    end  
    
    GX = G'*X;
    dtX = G - X*GX;         nrmG  = norm(dtX, 'fro');
    
    Grad(itr+1)   = nrmG;
    F_eval(itr+1) = F;
    
    if invA
        GXT = G*X';  A = GXT - GXT';        
    else
        if opts.projG == 1
            U =  [G, X];    V = [X, -G];       VU = V'*U;
        elseif opts.projG == 2
            GB = G - X*(0.5*GX');
            U  =  [GB, X];    V = [X, -GB];     VU = V'*U; 
        end
        VX = V'*X;
    end
    
    % ABB step-size:
    S = X - XP;             
    Y = dtX - dtXP;         SY = abs(sum(sum(S.*Y)));
   
    if mod(itr,2)==0; tau = sum(sum(S.*S))/SY;
    else tau  = SY/sum(sum(Y.*Y)); end
    tau = max(min(tau, 1e20), 1e-20);
    
    % Stopping Rules
    if nrmG < gtol  
        if itr <= 2
            ftol = 0.1*ftol;
            xtol = 0.1*xtol;
            gtol = 0.1*gtol;
        else
            out.msg = 'converge';
            break;
        end
    end
    Qp = Q; Q = gamma*Qp + 1; Cval = (gamma*Qp*Cval + F)/Q;
 end
tiempo = toc(tstart);
Grad = Grad(1:itr+1,1);
F_eval = F_eval(1:itr+1,1);


if itr >= opts.mxitr
    out.msg = 'exceed max iteration';
end

out.feasi = norm(X'*X-eye(k),'fro');
out.nrmG = nrmG;
out.fval = F;
out.itr = itr;
out.time = tiempo;
end
