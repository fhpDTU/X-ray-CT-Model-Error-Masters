function res = COR_Gibbs_Sampler(setup)
%This function samples from the full posterior with model errors using a
%Hierarchial Gibbs sampler.

%INPUT:
%setup: struct including the following fields

N_samples = setup.N_samples;            %Number of samples
N_iter = setup.N_iter;                  %Number of iterations for x-sample
b_noise = setup.b;                      %Projection data
theta = setup.theta;                    %Projection angles (degrees)
p = setup.p;                            %Number of detectors
N_metro = setup.N_metro;                %Number of metropolis hastings samples each iteration

alpha_delta = setup.alpha_delta;        %Shape parameter for delta prior
beta_delta = setup.beta_delta;          %Rate parameter for delta prior
alpha_lambda = setup.alpha_lambda;      %Shape parameter for lambda prior
beta_lambda = setup.beta_lambda;        %Rate parameter for lambda prior

lambda0 = setup.lambda0;                %Initial value for lambda parameter
delta0 = setup.delta0;                  %Initial value for delta parameter
c0 = setup.c0;                          %Initial value for COR parameter
x0 = setup.x0;                          %Initial value for x

sigma_prior = setup.sigma_prior;        %std. for Gaussian Prior
mean_prior = setup.mean_prior;          %mean for Gaussian Prior
sigma_proposal = setup.sigma_proposal;  %std. for proposal density

nonneg = setup.nonneg;                  %Specify if nonnegativity
iid =    setup.iid;                     %Specify if identity matrix for Gaussian Prior

%Set up FISTA algorithm details
N = sqrt(length(x0));
if iid == 1
    D = speye(N^2);
    L = D'*D;
else  
    e_vec = ones(N,1);
    L_1D = spdiags([-e_vec 2*e_vec -e_vec], [-1 0 1],N,N);
    L = kron(speye(N),L_1D) + kron(L_1D,speye(N));
    D = chol(L);
end

options.nonneg = nonneg;
options.maxiters = N_iter;
options.h = 1;
options.epsilon = 10^(-6);
options.iid = iid;

%Preallocate arrays for samples
x_samps =       zeros(N^2,N_samples);
delta_samps =   zeros(1,N_samples);
lambda_samps =  zeros(1,N_samples);
c_samps =       zeros(1,N_samples);

%Set starting x equal to regularized MAP estimate
lambda = lambda0;
delta = delta0;
c = c0;
x = x0;

%Generate initial forward operator
vol_geom = astra_create_vol_geom(N,N);

angles = theta*pi/180;

M = length(theta)*p;

vectors = zeros(length(theta),1);
vectors(:,1) = -sin(angles);
vectors(:,2) = cos(angles);
vectors(:,3) = c*cos(angles);
vectors(:,4) = c*sin(angles);
vectors(:,5) = cos(angles);
vectors(:,6) = sin(angles);

proj_geom = astra_create_proj_geom('parallel_vec',p,vectors);
A = opTomo('line',proj_geom,vol_geom);

n_accept = 0; %calculate acceptance rate

%Start sampling
for k=1:N_samples
    waitbar(k/N_samples);
    
    %Sample lambda and delta from Gamma congugate distributions
    
    if nonneg == 1
        N_nonzero = nnz(x);
        lambda = gamrnd(M/2+alpha_lambda,1/(1/2*norm(A*x-b_noise)^2+beta_lambda));
        delta = gamrnd(N_nonzero/2+alpha_delta,1/(1/2*x'*L*x+beta_delta));
    else
        lambda = gamrnd(M/2+alpha_lambda,1/(1/2*norm(A*x-b_noise)^2+beta_lambda));
        delta = gamrnd(N^2/2+alpha_delta,1/(1/2*x'*L*x+beta_delta));
    end
    
    %Sample COR parameter using Metropolis-Hastings
    
    for j = 1:N_metro
        %Get candidate sample from proposal
        c_prop = normrnd(c,sigma_proposal);
    
        %Check if proposed c is inside prior interval. If not then
        %we automatically reject it, and use the previous value.
    
        %Compute proposed system matrix
        vectors(:,3) = c_prop*cos(angles);
        vectors(:,4) = c_prop*sin(angles);
        
        proj_geom = astra_create_proj_geom('parallel_vec',p,vectors);
        A_prop = opTomo('line',proj_geom,vol_geom);
        
        %Ensure more numerical stability by doing logpdf
        lp_old = -lambda/2*norm(A*x-b_noise,2)^2-1/2*(c-mean_prior)/sigma_prior^2;
        lp_prop = -lambda/2*norm(A_prop*x-b_noise,2)^2-1/2*(c_prop-mean_prior)/sigma_prior^2;
        
        %Check if we accept
        if lp_prop-lp_old>log(rand())
            disp(['The sample was accepted - c = ' num2str(c_prop)])
            n_accept = n_accept + 1;
            A = A_prop;
            c = c_prop;
        end
    end
    
    %sample x using FISTA
    e_N = randn(N^2,1);
    e_M = randn(length(theta)*p,1);
    
    b_tilde = b_noise + lambda^(-1/2)*e_M;
    u = delta^(-1/2)*e_N;
    
    options.x0 = x;
    x = fista_Gen_tikh(A,D,b_tilde,u,lambda,delta,options);
    
    %save samples    
    delta_samps(k) = delta;
    lambda_samps(k) = lambda;
    c_samps(k) = c;
    x_samps(:,k) = x;
end

astra_clear;

%Save results
res.delta_samps = delta_samps;
res.lambda_samps = lambda_samps;
res.c_samps = c_samps;
res.x_samps = x_samps;
res.setup = setup;
res.acceptrate = n_accept/N_samples;
end