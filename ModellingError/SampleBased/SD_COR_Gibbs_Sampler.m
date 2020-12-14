function res = SD_COR_Gibbs_Sampler(setup)
%This function samples from the full posterior with Source Distance and COR Model
%Error using a Hierarchial Gibbs sampler with embedded Metropolis-Hastings sampling
%for the model parameters

%INPUT:
%setup: struct including the following fields

N = setup.N;                                    %Grid Size
N_samples = setup.N_samples;                    %Number of Gibbs samples
N_iter = setup.N_iter;                          %Number of FISTA iterations for x-sample
b_noise = setup.b;                              %Noisy sinogram
theta = setup.theta;                            %Projection angles (degrees)
p = setup.p;                                    %Number of detectors
N_metro = setup.N_metro;                        %Number of Metropolis-Hastings samples each iteration

DIST = setup.DIST;                              %Boolean specifying if SD parameter should be estimated
COR = setup.COR;                                %Boolean specifying if COR parameter should be estimated
TILT = setup.TILT;                              %Boolean specifying if TILT parameter should be estimated

alpha_delta = setup.alpha_delta;                %Shape parameter for delta (x-prior parameter) prior
beta_delta = setup.beta_delta;                  %Rate parameter for delta (x-prior parameter) prior
alpha_lambda = setup.alpha_lambda;              %Shape parameter for lambda (noise precision) prior
beta_lambda = setup.beta_lambda;                %Rate parameter for lambda (noise precision) prior

s_true = setup.s_true;                          %True value for SD parameter (physical length)
c_true = setup.c_true;                          %True value for COR parameter (physical length)
t_true = setup.t_true;                          %True value for Tilt parameter (degrees)

s0 = setup.s0;                                  %Initial value for SD parameter (physical length)
c0 = setup.c0;                                  %Initial value for COR parameter (physical length)
t0 = setup.t0;                                  %Initial value for TILT parameter (degrees)
x0 = setup.x0;                                  %Initial value for x

cor_sigma_prior = setup.cor_sigma_prior;        %std. for Gaussian COR Prior
cor_mean_prior = setup.cor_mean_prior;          %mean for Gaussian COR Prior
s_sigma_prior = setup.s_sigma_prior;            %std. for Gaussian SD Prior
s_mean_prior = setup.s_mean_prior;              %mean for Gaussian SD Prior
t_sigma_prior = setup.t_sigma_prior;            %std. for Gaussian Tilt Prior
t_mean_prior = setup.t_mean_prior;              %mean for Gaussian Tilt Prior

cor_sigma_proposal = setup.cor_sigma_proposal;  %std. for COR metropolis proposal density
s_sigma_proposal = setup.s_sigma_proposal;      %std. for SD metropolis proposal density
t_sigma_proposal = setup.t_sigma_proposal;      %std for Tilt metropolis proposal density

nonneg = setup.nonneg;                          %Specify if nonnegativity
iid =    setup.iid;                             %Specify if identity matrix for Gaussian precision matrix

M = length(theta)*p;  %"Rows" in forward operator

%Set up FISTA algorithm options
%Precision Matrix
if iid == 1
    D = speye(N^2);
    L = D;
else  
    e_vec = ones(N,1);
    L_1D = spdiags([-e_vec 2*e_vec -e_vec], [-1 0 1],N,N);
    L = kron(speye(N),L_1D) + kron(L_1D,speye(N));
    D = chol(L);
end
options.nonneg = nonneg;     %Set non-negativity options
options.maxiters = N_iter;   %Set maximum number of iterations
options.h = 1;               %Set discretization parameter
options.epsilon = 10^(-8);   %Heuristic tolerance parameter
options.iid = iid;           %Specify if identity precision matrix

%Preallocate arrays for samples
x_samps =       zeros(N^2,N_samples);
delta_samps =   zeros(1,N_samples);
lambda_samps =  zeros(1,N_samples);

%Preallocate array samples and set initial model parameters
%SOURCE DISTANCE
if DIST == 1
    s_samps = zeros(1,N_samples);   %Array for SD Gibbs samples
    s = s0;                         %Initial SD parameter    
    n_accept_SD = 0;                %Metropolis acceptance rate tracker
else
    s = s_true;
end
%CENTER OF ROTATION
if COR == 1
    c_samps = zeros(1,N_samples);   %Array for COR Gibbs samples
    c = c0;                         %Initial COR parameter
    n_accept_COR = 0;               %Metropolis acceptance rate tracker
else
    c = c_true;
end
%TILT PARAMETER
if TILT == 1
    t_samps = zeros(1,N_samples);   %Array for TILT Gibbs samples
    t = t0;                         %Initial TILT parameter
    n_accept_TILT = 0;              %Metropolis Acceptance rate tracker
else
    t = t_true;
end

%Set initial x-sample
x = x0;

%Convert projection angles to radians
angles = theta*pi/180;

%Set volume geometry
vol_geom = astra_create_vol_geom(N,N,-1,1,-1,1);

%Specify vectorized geometry
vectors = zeros(length(theta),1);
vectors(:,1) = sin(angles)*s;                          %First component for source location
vectors(:,2) = -cos(angles)*s;                         %Second component for source location
vectors(:,3) = c*cos(angles);                          %First component for detector center
vectors(:,4) = c*sin(angles);                          %Second component for detection center
vectors(:,5) = cos(angles + t/180*pi)*3/p;             %First component of detector basis
vectors(:,6) = sin(angles + t/180*pi)*3/p;             %Second component of detector basis

%Generate forward operator
proj_geom = astra_create_proj_geom('fanflat_vec',p,vectors);
A = opTomo('line_fanflat',proj_geom,vol_geom);

%Start sampling using Hierarchial Gibbs
for k=1:N_samples
    waitbar(k/N_samples);
    disp(['Gibbs iteration number: ' num2str(k)])
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Sample lambda and delta from conjugate Gamma distributions
    if nonneg == 1
        N_nonzero = nnz(x);
        lambda = gamrnd(M/2+alpha_lambda,1/(1/2*norm(A*x-b_noise)^2+beta_lambda));
        delta = gamrnd(N_nonzero/2+alpha_delta,1/(1/2*x'*L*x+beta_delta));
    else
        lambda = gamrnd(M/2+alpha_lambda,1/(1/2*norm(A*x-b_noise)^2+beta_lambda));
        delta = gamrnd(N^2/2+alpha_delta,1/(1/2*x'*L*x+beta_delta));
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if COR == 1
        count = 0;
        %Sample COR parameter using Metropolis-Hastings
    
        for j = 1:N_metro
            %Get candidate sample from proposal
            c_prop = normrnd(c,cor_sigma_proposal);
        
            vectors = zeros(length(angles),6);
    
            %Vectorized geometry
            vectors(:,1) = s*sin(angles);
            vectors(:,2) = -s*cos(angles);
            vectors(:,3) = c_prop*cos(angles);
            vectors(:,4) = c_prop*sin(angles);
            vectors(:,5) = cos(angles + t/180*pi)*3/p;
            vectors(:,6) = sin(angles + t/180*pi)*3/p;
            
            %Generate proposed forward operator
            proj_geom = astra_create_proj_geom('fanflat_vec',p,vectors);
            A_prop = opTomo('line_fanflat',proj_geom,vol_geom);
        
            %Calculate acceptance rate (we do log for numerical stability)
            lp_old = -lambda/2*norm(A*x-b_noise,2)^2-1/2*(c-cor_mean_prior)/cor_sigma_prior^2;
            lp_prop = -lambda/2*norm(A_prop*x-b_noise,2)^2-1/2*(c_prop-cor_mean_prior)/cor_sigma_prior^2;
        
            %Check if we accept
            if lp_prop-lp_old>log(rand())
                count = count + 1;
                n_accept_COR = n_accept_COR + 1;
                A = A_prop;
                c = c_prop;
            end
        end
        disp(['New c-sample: ' num2str(c)])
        disp(['Number of accepted Metropolis for c-sampling: ' num2str(count)])
        c_samps(k) = c;
    end
    
    if DIST == 1
        count = 0;
        %Sample SD parameter using Metropolis-Hastings 
        for j = 1:N_metro
            %Get candidate sample from proposal
            s_prop = normrnd(s,s_sigma_proposal);
        
            vectors = zeros(length(angles),6);
        
            %Specify vectorized geometry
            vectors(:,1) = s_prop*sin(angles);
            vectors(:,2) = -s_prop*cos(angles);
            vectors(:,3) = c*cos(angles);
            vectors(:,4) = c*sin(angles);
            vectors(:,5) = cos(angles + t/180*pi)*3/p;
            vectors(:,6) = sin(angles + t/180*pi)*3/p;
            
            %Get proposed forward operator
            proj_geom = astra_create_proj_geom('fanflat_vec',p,vectors);
            A_prop = opTomo('line_fanflat',proj_geom,vol_geom);
        
            %Ensure more numerical stability by doing logpdf
            lp_old = -lambda/2*norm(A*x-b_noise,2)^2-1/2*(s-s_mean_prior)/s_sigma_prior^2;
            lp_prop = -lambda/2*norm(A_prop*x-b_noise,2)^2-1/2*(s_prop-s_mean_prior)/s_sigma_prior^2;
        
            %Check if we accept
            if lp_prop-lp_old>log(rand())
                count = count + 1;
                n_accept_SD = n_accept_SD + 1;
                A = A_prop;
                s = s_prop;
            end
        end
        disp(['New s sample: ' num2str(s)])
        disp(['Number of accepted Metropolis for s sampling: ' num2str(count)])
        s_samps(k) = s;
    end
    
    if TILT == 1
        count = 0;
        %Sample SD parameter using Metropolis-Hastings 
        for j = 1:N_metro
            %Get candidate sample from proposal
            t_prop = normrnd(t,t_sigma_proposal);
        
            vectors = zeros(length(angles),6);
        
            %Specify vectorized geometry
            vectors(:,1) = s*sin(angles);
            vectors(:,2) = -s*cos(angles);
            vectors(:,3) = c*cos(angles);
            vectors(:,4) = c*sin(angles);
            vectors(:,5) = cos(angles + t_prop/180*pi)*3/p;
            vectors(:,6) = sin(angles + t_prop/180*pi)*3/p;
            
            %Get proposed forward operator
            proj_geom = astra_create_proj_geom('fanflat_vec',p,vectors);
            A_prop = opTomo('line_fanflat',proj_geom,vol_geom);
        
            %Ensure more numerical stability by doing logpdf
            lp_old = -lambda/2*norm(A*x-b_noise,2)^2-1/2*(t-t_mean_prior)/t_sigma_prior^2;
            lp_prop = -lambda/2*norm(A_prop*x-b_noise,2)^2-1/2*(t_prop-t_mean_prior)/t_sigma_prior^2;
        
            %Check if we accept
            if lp_prop-lp_old>log(rand())
                count = count + 1;
                n_accept_TILT = n_accept_TILT + 1;
                A = A_prop;
                t = t_prop;
            end
        end
        disp(['New t sample: ' num2str(t)])
        disp(['Number of accepted Metropolis for t sampling: ' num2str(count)])
        t_samps(k) = t;
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Sample x from multivariate normal distribution using FISTA
    
    %Get i.i.d normal samples
    e_N = randn(N^2,1);
    e_M = randn(length(theta)*p,1);
    
    %Compute modified rhs and mean deviation
    b_tilde = b_noise + lambda^(-1/2)*e_M;
    u = delta^(-1/2)*e_N;
    
    %Run FISTA starting at previous sample for N_iter iterations
    options.x0 = x;
    x = fista_Gen_tikh(A,D,b_tilde,u,lambda,delta,options);
    
    %save samples    
    delta_samps(k) = delta;
    lambda_samps(k) = lambda;
    x_samps(:,k) = x;
end

%Clear all astra objects
astra_clear;

%Save results
res.delta_samps = delta_samps;
res.lambda_samps = lambda_samps;
if DIST == 1
    res.s_samps = s_samps;
    res.acceptrate_DIST = n_accept_SD/(N_samples*N_metro);
end
if COR == 1
    res.c_samps = c_samps;
    res.acceptrate_COR = n_accept_COR/(N_samples*N_metro);
end
if TILT == 1
    res.t_samps = t_samps;
    res.acceptrate_TILT = n_accept_TILT/(N_samples*N_metro);
end
res.x_samps = x_samps;
res.setup = setup;
end

function [sample,n_accept] = Metropolis_Gaussian_Proposal(N_metro,mean_proposal,sigma_proposal,mean_prior,sigma_prior,p,theta,vol_geom,mod_param,lambda,x,A)
%This function samples model parameters using Metropolis-Hastings with a
%Gaussian proposal.

%INPUT:
%N_metro: Number of Metropolis-Hastings iterations
%mean_proposal: Mean of Gaussian Proposal
%sigma_proposal: std. of Gaussian Proposal (step-size)
%mean_prior: Mean of Gaussian prior
%sigma_prior: std. of Gaussian prior
%p: Number of detectors
%angles: Projection angles (radians)
%vol_geom: Astra object containing the volume geometry
%mod_param: String specifying model parameter. The following are possible:
            %COR: Center of Rotation
            %SD: Source-Origin Distance
            %DD: Detector-Origin Distance
            %TILT: Detector Tilt 
%lambda: Current lambda-sample
%x: Current x-sample
%A: Current forward operator

%OUTPUT:
%sample: Resulting sample after N_metro iterations
%n_accept: Number of accepted steps

%Precompute forward projection of current sample
Ax = A*x;

count = 0;
    
for j = 1:N_metro
    %Get candidate sample from proposal
    prop = normrnd(mean_proposal,sigma_proposal);
    
    switch mod_param
        case COR
            vectors = zeros(length(angles),6);
    
            %Vectorized geometry
            vectors(:,1) = s*sin(angles);
            vectors(:,2) = -s*cos(angles);
            vectors(:,3) = c_prop*cos(angles);
            vectors(:,4) = c_prop*sin(angles);
            vectors(:,5) = cos(angles)*3/p;
            vectors(:,6) = sin(angles)*3/p;
            
            %Generate proposed forward operator
            proj_geom = astra_create_proj_geom('fanflat_vec',p,vectors);
            A_prop = opTomo('line_fanflat',proj_geom,vol_geom);
        
            %Calculate acceptance probability (we do log for numerical stability)
            lp_old = -lambda/2*norm(A*x-b_noise,2)^2-1/2*(c-cor_mean_prior)/cor_sigma_prior^2;
            lp_prop = -lambda/2*norm(A_prop*x-b_noise,2)^2-1/2*(c_prop-cor_mean_prior)/cor_sigma_prior^2;
        
            %Check if we accept
            if lp_prop-lp_old>log(rand())
                count = count + 1;
                n_accept_COR = n_accept_COR + 1;
                A = A_prop;
                c = c_prop;
            end
    end
end
end