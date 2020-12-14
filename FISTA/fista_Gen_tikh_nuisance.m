function [res,f_vec] = fista_Gen_tikh_nuisance(A,D,b,u,delta,R,options)
%This function solves the Generalized Tikhonov Minimization Problem

%           min_x 1/2 ||RAx-b||_2^2+delta/2 ||Dx-u||_2^2
%           s.t.  x>=0 (optional)

%using the FISTA method with non-negative projections

%INPUT:
%A: System Matrix
%D: Prior Matrix
%b: data
%u: Prior mean shift
%delta: Regularization parameter
%R: inverse cholesky of covariance matrix
%options: struct containing the fields:
    
    %maxiters: Maxiumum number of iterations (def. 50)
    %x0: Initial guess (def. zero vector)
    %iid: Boolean variable denoting if prior structure is identity (def. 1)
    %nonneg: Boolean variable denoting if we want non-negativity (def. 1)
    %epsilon: stopping tolerance (def. 10^(-3))

%Output:
%res: minimizer
%f_vec: vector of function values for different iterations

n = size(D,1);

if nargin>6
    x0 = options.x0;
    maxiters = options.maxiters;
    nonneg = options.nonneg;
    epsilon = options.epsilon;
    iid = options.iid;
    h = options.h;
else
    x0 = zeros(n,1);
    maxiters = 50;
    nonneg = 1;
    epsilon = 10^(-8);
    iid = 1;
    h = 10^(-3);
end

%Determine Lipschitz constant using power iteration
n_lip = 20;
x = randn(n,1);
for k=1:n_lip
    z = (R*(A*x))/norm(x,2);
    x = A'*(R'*z);
end

if iid == 1
    L = norm(x) + delta;
else
    L = norm(x) + 8*delta/h^2;
end

%Initialize algorithm
xold = x0;
y = x0;
told = 1;
k = 1;
converged = 0;

%Compute initial objective value
if nargout>1
    f_vec = lambda/2*norm(R*(A*x0)-b,2)^2+delta/2*norm(x0-u,2)^2;
end


while k<maxiters && ~converged
    k = k+1;
    %Evaluate gradient
    grad = A'*(R'*(R*(A*y)-b))+delta*D'*(D*y-u);
    
    %Take gradient step
    x = y-1/L*grad;
    
    %Implement non-negativity
    if nonneg
        x = max(0,x(:));
    end
    
    %Update t and y
    tnew = (1+sqrt(1+4*told^2))/2;
    y = x + (told-1)/tnew*(x-xold);
    
    %Compute relative difference in of norms between the iterates to
    %determine convergence
    relDif = norm(x-xold,2)/norm(xold,2);
    if relDif<epsilon
        converged = 1;
        disp(['The algorithm stopped at iteration ' num2str(k) ' because the relative change was below the specified tolerance'])
    end
    
    told = tnew;
    xold = x;
    
    %Compute objective value
    if nargout>1
        fval = 1/2*norm(R*A*x-b,2)^2+delta/2*norm(D*x-u,2)^2;
        f_vec = [f_vec fval];
    end
end

if k==maxiters
    disp('The algorithm stopped because the number of iterations reached the specified maximum')
end
res = x;
end