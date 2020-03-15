function [X, out, Grada, F_eval]= OptStiefelCGC(X, fun, opts, varargin)
%-------------------------------------------------------------------------
% Riemannian conjugate gradient algorithm for optimization on Stiefel manifold
%
%   min F(X), S.t., X'*X = I_p, where X \in R^{n,p}
%
%
% Input:
%           X --- n by p matrix such that X'*X = I
%         fun --- objective function and its gradient:
%                 [F, G] = fun(X,  data1, data2)
%                 F, G are the objective function value and gradient, repectively
%                 data1, data2 are addtional data, and can be more
%                 Calling syntax:
%                   [X, out]= OptStiefelGBB(X0, @fun, opts, data1, data2);
%
%        opts --- option structure with fields:
%                 record = 0, no print out
%                 mxitr       max number of iterations
%                 xtol        stop control for ||X_k - X_{k-1}||
%                 gtol        stop control for the projected gradient
%                 ftol        stop control for |F_k - F_{k-1}|/(1+|F_{k-1}|)
%                             usually, max{xtol, gtol} > ftol
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
% opts.xtol = 1e-5;
% opts.gtol = 1e-5;
% opts.ftol = 1e-8;
% 
% X0 = randn(n,k);    X0 = orth(X0);
% tic; [X, out]= OptStiefelRCG(X0, @fun, opts, A); tsolve = toc;
% out.fval = -2*out.fval; % convert the function value to the sum of eigenvalues
% fprintf('\nOptM: obj: %7.6e, itr: %d, nfe: %d, cpu: %f, norm(XT*X-I): %3.2e \n', ...
%             out.fval, out.itr, out.nfe, tsolve, norm(X'*X - eye(k), 'fro') );
% 
% end
% -------------------------------------
%
% Reference: 
%  X. Zhu
%  A Riemannian conjugate gradient method for optimization on the Stiefel manifold
%
% Author: Xiaojing Zhu
%   Version 1.0 .... 2016/04
%-------------------------------------------------------------------------


%% Size information
if isempty(X)
    error('input X is an empty matrix');
else
    [n, p] = size(X);
end

if isfield(opts, 'xtol')
    if opts.xtol < 0 || opts.xtol > 1
        opts.xtol = 1e-6;
    end
else
    opts.xtol = 1e-6;
end

if isfield(opts, 'gtol')
    if opts.gtol < 0 || opts.gtol > 1
        opts.gtol = 1e-6;
    end
else
    opts.gtol = 1e-6;
end

if isfield(opts, 'ftol')
    if opts.ftol < 0 || opts.ftol > 1
        opts.ftol = 1e-12;
    end
else
    opts.ftol = 1e-12;
end

% parameters for control the linear approximation in line search
if isfield(opts, 'delta')
   if opts.delta < 0 || opts.delta > 1
        opts.delta = 1e-4;
   end
else
    opts.delta = 1e-4;
end

% factor for decreasing the step size in the backtracking line search
if isfield(opts, 'lambda')
   if opts.lambda < 0 || opts.lambda > 1
        opts.lambda = 0.1;
   end
else
    opts.lambda = 0.2;
end


if isfield(opts, 'alpha')
   if opts.alpha < 0 || opts.alpha > 1e3
        opts.alpha = 1e-3;
   end
else
    opts.alpha = 1e-3;
end

% backward integer for nonmonotone line search 
if isfield(opts, 'm')
   if opts.m < 0 || opts.m > 100
        opts.m = 2;
   end
else
    opts.m = 2;
end

% choice for vector transports
if isfield(opts, 'vt')
   if opts.vt ~= 1 && opts.vt ~= 2
        opts.vt = 1;
   end
else
    opts.vt = 1;
end

if isfield(opts, 'nt')
    if opts.nt < 0 || opts.nt > 100
        opts.nt = 5;
    end
else
    opts.nt = 5;
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
xtol = opts.xtol;
gtol = opts.gtol;
ftol = opts.ftol;
delta  = opts.delta;
lambda   = opts.lambda;
m=opts.m;
vt=opts.vt;
alpha = opts.alpha;
record = opts.record;
nt = opts.nt;   crit = ones(nt, 3);

eye2p = eye(2*p);

%% Initial function value and gradient
% prepare for iterations
[F,  G] = feval(fun, X , varargin{:});  out.nfe = 1; 
XG=X'*G;
Grad=G-0.5*X*(XG+XG');
Z=-Grad;
PZ=-G+0.5*X*XG;
U=[PZ,X]; V=[X,-PZ];
VX=V'*X;
VU=V'*U;
%VU=[-0.5*XG,eye(p);-G'*G+0.5*(XG)^2,0.5*XG'];
%VX=VU(:,p+1:end);
prodGZ=sum(dot(Grad,Z,1));
nrmGrad=norm(Grad,'fro');
nrmG0 = norm(G-X*XG','fro');
Fm=F;


Grada  = zeros(opts.mxitr+1,1);      Grada(1)   = nrmG0;
F_eval = zeros(opts.mxitr+1,1);     F_eval(1) = F; 

recG=[];
%ratio=[];

%% Print iteration header if debug == 1
if (opts.record == 1)
    fid = 1;
    fprintf(fid, '----------- Gradient Method with Line search ----------- \n');
    fprintf(fid, '%4s %8s %8s %10s %10s\n', 'Iter', 'tau', 'F(X)', 'nrmG', 'XDiff');
    %fprintf(fid, '%4d \t %3.2e \t %3.2e \t %5d \t %5d	\t %6d	\n', 0, 0, F, 0, 0, 0);
end

%% main iteration
tini = tic();
for itr = 1 : opts.mxitr
    XP = X;     FP = F;   nrmGradP = nrmGrad; GradP = Grad; ZP=Z; 
     % scale step size
     
    nls = 1; 
    while 1
        
        M=linsolve(eye2p-0.5*alpha*VU,VX);        
        
        X=XP+alpha*U*M;
        
        [F,G] = feval(fun, X, varargin{:});
        out.nfe = out.nfe + 1;
        
        if F <= max(Fm) + alpha*delta*prodGZ || nls >= 5
            break;
        end
        alpha = lambda*alpha;          nls = nls+1;
    end
    
    if vt==1   % Algoritmo1.a (18)
        VUM=VU*M;
        TZ=U*(VX+0.5*alpha*VUM+0.5*alpha*linsolve(eye2p-0.5*alpha*VU,VUM));
    else   % Algoritmo1.b (23)
        TZ=U*(VX+alpha*VU*M);
    end
    
    %ratio=[ratio,norm(TZ,'fro')/norm(ZP,'fro')]; 
    
    XG=X'*G;
    Grad = G-0.5*X*(XG+XG');
    nrmGrad = norm(Grad, 'fro');
    nrmG = norm(G-X*XG','fro');
    rnrmG=nrmG/nrmG0;
    recG=[recG,rnrmG];
    betaD = nrmGrad^2/max([sum(dot(Grad,TZ,1))-prodGZ, -prodGZ]);
    betaFR = nrmGrad^2/nrmGradP^2;
    beta = min(betaD,betaFR);
    % betaDY=nrmGrad^2/(dot(Grad,TZ,1)-prodGZ); beta=betaDY;
    Z=-Grad+beta*TZ;
    XZ=X'*Z;
    PZ=Z-0.5*X*XZ;
    U=[PZ,X]; V=[X,-PZ];
    VX=V'*X;
    VU=V'*U;
    %VU=[0.5*XZ,eye(p);-Z'*Z+0.5*(XZ)^2,-0.5*XZ'];
    %VX=VU(:,p+1:end);
    prodGZ=sum(dot(Grad,Z,1));    
     
    Grada(itr+1)   = nrmG;
    F_eval(itr+1) = F;
    
    
    S=alpha*ZP;  Y=Grad-GradP;
    
    alpha = sum(dot(S,S,1))/abs(sum(dot(S,Y,1)));
    
    alpha = max(min(alpha, 1), 1e-20);
    
    %alpha=-prodGZ/sum(dot(Z,varargin{1}*Z,1));
  
    

    
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
    
    Q=Fm;
    Fm=[F,Q];
    mk=min(m-1,itr);
    Fm=Fm(1:mk+1);
 end
time = toc(tini);

Grada  = Grada(1:itr+1,1);
F_eval = F_eval(1:itr+1,1);

if itr >= opts.mxitr
    out.msg = 'exceed max iteration';
end

out.feasi = norm(X'*X-eye(p),'fro');
if  out.feasi > 1e-13
    X = MGramSchmidt(X);
    [F,G] = feval(fun, X, varargin{:});
    out.nfe = out.nfe + 1;
    out.feasi = norm(X'*X-eye(p),'fro');
end

out.nrmG = nrmG;
out.fval = F;
out.itr = itr;
out.time = time;
out.rnrmG = rnrmG;
out.recG = recG;
%out.ratio = ratio;


