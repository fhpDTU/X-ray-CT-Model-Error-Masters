close all
clc
clear
%%
rng(100);

N = 100; %Grid size
theta = 0:180; %angles
p = 2*N; %Number of rays
d = p-1; %Width of detector
dxoffset = 10.75;
da = 0; %Tilt of detector (degrees)

phantomstring = 'SheppLogan';
x_true = phantom(N);

%Create analytical sinogram using modified air-tools
[~,b_true] = paralleltomo_mod(N,theta,p,[d,dxoffset,da]);

%Add gaussian additive noise to sinogram
rho = 0.05; %noise level
e = randn(size(b_true)); %standard normal distributed noise

%Normalize the noise and add to sinogram
e = rho*norm(b_true)*e/(norm(e));
b_noise = b_true + e;

%Gaussian iid prior
L = speye(N^2);
D = speye(N^2);
precisionstring = 'iid';

%Gaussian Laplacian Prior
%e = ones(N,1);
%L_1D = spdiags([-e 2*e -e], [-1 0 1],N,N);
%L = kron(speye(N),L_1D) + kron(L_1D,speye(N));
%D = chol(L);
%precisionstring = 'laplacian';
%%
%Determine initial estimate of COR using the cross-covariance approach
b_noise = reshape(b_noise,p,length(theta));

[c,lags] = xcorr(flipud(b_noise(:,end)),b_noise(:,1));
[~,ind] = max(c);
opt_lag = lags(ind);
c0 = opt_lag/2;

%Remove the projection at 180 degrees, and generate system-matrix with
%estimated COR-parameter
b_noise(:,end) = [];
b_noise = reshape(b_noise',p*length(0:179),1);

theta = 0:179;
M = p*length(theta);

%Create volume geometry
vol_geom = astra_create_vol_geom(N,N);

%Create projection geometry manually
angles = pi/180*theta;
vectors = zeros(length(theta),6);

%Set ray direction
vectors(:,1) = -sin(angles);
vectors(:,2) = cos(angles);

%Set Detector center
vectors(:,3) = c0*cos(angles);
vectors(:,4) = c0*sin(angles);

%Set Detector basis-vector
vectors(:,5) = cos(angles);
vectors(:,6) = sin(angles);

%Create projection geometry
proj_geom = astra_create_proj_geom('parallel_vec',2*N,vectors);

%Create Spot operator
A = opTomo('line',proj_geom,vol_geom);

%Precompute matrices
ATA = A'*A;
ATb = A'*b_noise;

%Minimize GCV to find regularization parameter
maxit = 200;
options = optimset('Display','iter','MaxIter',10);
alpha = fminbnd(@GCV_CT,0,10,options,A,b_noise,ATA,ATb,L,maxit);

%Compute MAP estimate
x_alpha = pcg(ATA+alpha*L,ATb,[],1000);

%Compute estimates sigma_0 and delta_0 for the heck of it (to get an idea of where we
%start)
lambda_0 = 1/var(A*x_alpha-b_noise);
delta_0 = alpha*lambda_0;
%%
%Implement the Hierarchial Gibbs sampler
N_samples = 1000;
N_burn = 100;
t0 = 2; %Time when we switch from using the initial covariance structure to using the
          %adapted covariance structure

%Preallocate arrays for samples
x_samps = zeros(N^2,N_samples-N_burn);
delta_samps = zeros(1,N_samples-N_burn);
lambda_samps = zeros(1,N_samples-N_burn);
c_samps = zeros(1,N_samples-N_burn);

%Parameters for precision variable priors
alpha_delta = 1; beta_delta = 10^(-4);
alpha_sigma = 1; beta_sigma = 10^(-4);

%Set initial parameters
lambda = lambda_0;
delta = delta_0;
c = c0;
x = x_alpha;
xc = [x;c];
sigma_c = 1/3;
xc_mean = xc; %Mean of all samples

%Set initial empirical covariance structure
epsilon = 10^(-8); %Factor to ensure SPD for initial iterations
sd = 2.4^2/(N^2+1); %Scaling dimension factor (see article)
%sd = 1;
R = sqrt(epsilon)*eye(N^2+1); %Cholesky factorization of covariance matrix

%Evaluate l_pdf for chain initialization
lpdf_old = -lambda/2*norm(A*x-b_noise,2)^2-delta/2*x'*L*x;
n_accept = 0;
%Start sampling
for k=1:N_samples
    h = waitbar(k/N_samples);
    
    %sample sigma and delta from gamma distribution
    lambda = gamrnd(M/2+alpha_sigma,1/(1/2*norm(A*x-b_noise)^2+beta_sigma));
    delta = gamrnd(N^2/2+alpha_delta,1/(1/2*x'*L*x+beta_delta));
    
    %Sample from the joint distribution (x,c)|(sigma,delta) using Adapted
    %Metropolis-Hastings algorithm.
    
    %Compute proposed sample from adaptive proposal density (For the
    %initial runs, we assume that $x and c$ and independent!)
    if k<t0
        %Sample x using CG
        e_N = randn(N^2,1);
        e_M = randn(length(0:179)*p,1);
    
        reg_param = delta/lambda;
        perturb = lambda^(-1/2)*A'*e_M+sqrt(delta)*D'*e_N/(lambda);
        rhs_CG = ATb+perturb;
        x_prop = pcg(ATA+reg_param*L,rhs_CG,[],1000,[],[],zeros(N^2,1));
        
        %Sample c from a normal distribution
        c_prop = c + sigma_c*randn(1,1);
    else
        xc_prop = xc + R'*randn(N^2+1,1);
        x_prop = xc_prop(1:end-1);
        c_prop = xc_prop(end);
    end
    
    %Check that c is within prior interval
    if c_prop>c0-0.5 && c_prop<c0+0.5
        %Compute system matrix for proposal
        vectors(:,3) = c_prop*cos(angles);
        vectors(:,4) = c_prop*sin(angles);
        
        proj_geom = astra_create_proj_geom('parallel_vec',2*N,vectors);
        A_prop = opTomo('line',proj_geom,vol_geom);
        
        %Compute logpdf
        lpdf_prop = -lambda/2*norm(A_prop*x_prop-b_noise,2)^2-delta/2*x_prop'*L*x_prop;
        lpdf_prop-lpdf_old
        %Flip alpha-coin to see if we accept
        if lpdf_prop-lpdf_old>log(rand)
            n_accept = n_accept + 1;
            x = x_prop;
            c = c_prop;
            A = A_prop;
            xc = [x;c];
            lpdf_old = lpdf_prop;
        end
    end
        
    %Update Covariance structure (rank 1 update)
    xc_mean = xc_mean + (xc - xc_mean)/k;
    %R = cholupdate(sqrt((k-1)/k)*R,sqrt(sd)*xc_mean,'+');
    if k>2
        R = cholupdate(sqrt((k-2)/(k-1))*R,1/(sqrt(k-1))*(xc-xc_mean));
    end
        
    %save samples if after Burn period    
    if k>N_burn
        delta_samps(k-N_burn) = delta;
        lambda_samps(k-N_burn) = lambda;
        c_samps(k-N_burn) = c;
        
        x_samps(:,k-N_burn) = x;
    end
end

astra_clear;

%Do Geweke test to asses convergence
[z_delta,pval_delta] = geweke(delta_samps');
[z_lambda,pval_lambda] = geweke(lambda_samps');
[z_c,pval_c] = geweke(c_samps');
%%
%Create data-structure with all information
data.xsamps = x_samps;
data.deltasamps = delta_samps;
data.sigmasamps = lambda_samps;
data.CORsamps = c_samps;
data.noiselevel = rho;
data.theta = theta;
data.rays = p;
data.DetectorDistance = d;
data.regParam = alpha;
data.noise = e;
data.precisionMatrix = precisionstring;
data.Nsamps = N_samples;
data.method = 'Adapted Metropolis-Hastings';
data.rhs = 'Analytical';
data.GridSize = N;
data.phantom = x_true;
data.noisysino = b_noise;
data.COR = dxoffset;
data.Acceptancerate = n_accept/N_samples;
data.proposalStd = sigma_proposal;
data.pval_delta = pval_delta;
data.pval_sigma = pval_lambda;
data.pval_COR = pval_c;


filename = strcat(phantomstring,'Gaussian',precisionstring,'Noise',num2str(rho*100),'%','Analytical','COR','astra','AdaptedMetro','.mat');

save(filename,'data');
%%
%Plot the results
filename = 'Mystring.mat';
mydata = load(filename);
GaussianSamplingPlotCOR(mydata);