function [lambda,v,out] = power_method(A,x,tolg)
t_ini = tic;
nAx = 1;
lambda = 0;
v = x;  Av = A*v; itr = 0;
while 1   
     % Power iteration
     v = Av;
     lambda = norm(v);
     v = v/lambda;
     
     Av = A*v;      nrmG =  norm(Av - v*(v'*Av));
     nAx = nAx + 1;
     
     itr = itr + 1;
     % Stopping rules
     if nrmG < tolg || itr > 100000
         break;
     end
    
end
time = toc(t_ini);


out.itr = itr;
out.nAx   = nAx;
out.nrmG  = nrmG;
out.fval  = lambda;
out.feasi = abs(v'*v - 1);
out.time  = time;
end

