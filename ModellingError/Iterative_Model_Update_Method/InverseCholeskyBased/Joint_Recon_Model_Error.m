function out = Joint_Recon_Model_Error(setup)
%This function solves the joint-reconstruction problem with modelling
%errors using algorithm 5 from the thesis. The method is akin to the one
%proposed by Nicolai in his paper only for other types of modelling error.

%OUTPUT
%out: struct containing the final reconstruction and the parameter estimates

%INPUT
%setup: struct containing the following fields

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%UNPACK THE VARIABLES

%GEOMETRY
N =                 setup.N;                %GridSize (N x N)
theta =             setup.theta;            %Projection angles in degrees
p =                 setup.p;                %Number of rays
ray_config =        setup.ray_config;       %Specify the ray configuration (parallel or fanbeam)
c_true =            setup.c_true;           %The true COR parameter (pixels)
d_true =            setup.d_true;           %The true detector-origin distance (Some unit)
s_true =            setup.s_true;           %The true source-origin distance (Some unit)

%MODELLING ERROR
c0 =                setup.c0;               %Initial guess for COR parameter (pixels)
d0 =                setup.d0;               %Initial guess for detector-origin parameter (some unit)
s0 =                setup.s0;               %Initial guess for source-origin distance (some unit)
sigma_COR_0 =       setup.sigma_COR_0;      %Initial standard deviation of COR parameter
sigma_d_0 =         setup.sigma_d_0;        %Initial standard devaiation of Tilt parameter
sigma_s_0 =         setup.sigma_s_0;        %Initial standard deviation of source distance parameter
COR =               setup.COR;              %Specify if COR is to be estimated
DETECTOR_DIST =     setup.DETECTOR_DIST;    %Specify if Detector-Origin Distance is to be estimated
SOURCE_DIST =       setup.SOURCE_DIST;      %Specify if Source-Origin Distance is to be estimated

%REGULARIZATION TERM AND SOLVER PARAMETERS
reg_term =          setup.reg_term;         %Specify regularization term (tikh, gentikh or TV)
nonneg =            setup.nonneg;           %Specify if nonnegativity constraints(1 or 0)
alpha =             setup.alpha;            %Regularization parameter for inner problem
maxiters =          setup.maxiters;         %Maximum number of iterations within outer loop

%AUXILLARY PARAMETERS
N_out =             setup.N_out;            %Number of outer iterations
S =                 setup.S;                %Number of samples for modelerror reconstruction step
S_update =          setup.S_update;         %Number of samples for parameter update step
gamma =             setup.gamma;            %Variance relaxation parameter
gpu =               setup.gpu;              %Specify if gpu (1 if gpu 0 if not)
fig_show =          setup.fig_show;         %Specify if we want to show figures

b_noise =           setup.b_noise;          %noisy sinogram
lambda =            setup.lambda;           %noise precision of measurement errors
x_ini =             setup.x_ini;            %Reconstruction with initial geometric parameter

%Set seed
rng(setup.seed);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Set algorithm parameters for the specific regularization term
if strcmp(reg_term,'tikh')
    %Gaussian iid prior
    h = 1;
    options.nonneg = nonneg;
    options.iid = 1;                        %Specify if i.i.d precision matrix
    options.h = h;                          %Set discretization step
    options.maxiters = maxiters;            %Set maximum number of iterations
    options.epsilon = 10^(-8);              %Set stopping tolerance
    options.x0 = zeros(N^2,1);              %Set initial guess
    
    %Built precision matrix
    L = speye(N^2);
    D = speye(N^2);
    
    options.D = D;
    options.L = L;
elseif strcmp(reg_term,'gentikh')
    h = 1;
    options.nonneg = nonneg;
    options.iid = 0;                        %Specify if i.i.d precision matrix
    options.h = h;                          %Set discretization step
    options.maxiters = maxiters;            %Set maximum number of iterations
    options.epsilon = 10^(-8);              %Set stopping tolerance
    options.x0 = zeros(N^2,1);              %Set Initial guess
    
    %Build precision matrix  
    e_vec = ones(N,1);
    L_1D = spdiags([-e_vec 2*e_vec -e_vec], [-1 0 1],N,N);
    L = kron(speye(N),L_1D) + kron(L_1D,speye(N));
    D = chol(L);
    D = 1/h*D;
    options.D = D;
    options.L = L;
elseif strcmp(reg_term,'TV')
    options.nonneg = nonneg;                %Set nonnegativity
    options.h = 10^(-3);                    %Set discretization step
    options.rho = 10^(-2);                  %Set TV-smoothing parameter
    options.epsilon = 10^(-6);              %Set stopping parameter
    options.maxiters = maxiters;            %Set Max iterations
    options.x0 = zeros(N^2,1);              %Set starting guess
end

if COR == 0
    c0 = c_true;
end

if DETECTOR_DIST == 0
    d0 = d_true;
end

if SOURCE_DIST == 0
    s0 = s_true;
end

M = p*length(theta);
angles = theta*pi/180;
vol_geom = astra_create_vol_geom(N,N);

%Compute initial forward operator
A = Build_Forward_Operator(ray_config,vol_geom,gpu,angles,p,c0,d0,s0);

%Preallocate arrays for samples
x_samps = zeros(N^2,N_out);
c_samps = zeros(1,N_out);
d_samps = zeros(1,N_out);
s_samps = zeros(1,N_out);

%Set initial parameters
if COR == 1
    c_mean = c0;                        %Initial estimate of COR - parameter
else
    c_mean = c_true;
end
if DETECTOR_DIST == 1
    d_mean = d0;                        %Initial estimate of Tilt - parameter
else
    d_mean = d_true;
end
if SOURCE_DIST == 1
    s_mean = s0;                        %Initial estimate of DIST - parameter
else
    s_mean = s_true;
end

sigma_COR = sigma_COR_0;                %Initial standard deviation of COR
sigma_d = sigma_d_0;             %Initial standard deviation of Tilt
sigma_s = sigma_s_0;                    %Initial standard deviation of DIST

x = x_ini;                              %Initial reconstruction estimate                   

%Run the algorithm
for k=1:N_out
    loading_bar = waitbar(k/N_out);
    
    %PARAMETER UPDATE STEP
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Preallocate space for samples
    model_error_vec = zeros(M,S_update);
    c_vec = zeros(1,S_update);
    d_vec = zeros(1,S_update);
    s_vec = zeros(1,S_update);
    
    Ax = A*x;
    
    %Sample S_update instances of model error
    for i = 1:S_update
        if COR == 1
            %Sample c from prior distribution
            c = normrnd(c_mean,sigma_COR);
            c_vec(i) = c;
        else
            c = c_true;
        end
        
        if DETECTOR_DIST == 1
            %Sample detector tilt from prior distribution
            d = normrnd(d_mean,sigma_d);
            d_vec(i) = d;
        else
            d = d_true;
        end
        
        if SOURCE_DIST == 1
            %Sample source distance from prior distribution
            s = normrnd(s_mean,sigma_s);
            s_vec(i) = s;
        else
            s = s_true;
        end
        
        %Build sampled forward operator
        A_samp = Build_Forward_Operator(ray_config,vol_geom,gpu,angles,p,c,d,s);
        
        %Compute sample Model Error
        model_error_vec(:,i) = A_samp*x-Ax;
    end
    
    %Compute mean of model error samples
    model_error_mean = mean(model_error_vec,2);
    
    %Compute cholesky factorization of (C+1/lambda I)^(-1) using rank
    %1 updates (C is empirical covariance matrix).
    R = sqrt(lambda)*eye(M);
    for i=1:S_update
        u = (model_error_vec(:,i)-model_error_mean)/sqrt(S_update-1);
        w = R'*(R*u);
        perturb = w/sqrt(1+u'*w);
        R = cholupdate(R,perturb,'-');
    end
    
    %Update Model Error Parameters
    if COR == 1
        cov_full = cov([model_error_vec;c_vec]');
        C_21_COR = cov_full(end,1:end-1);
        C_12_COR = cov_full(1:end-1,end);
        
        %Compute conditional expectation and covariance
        c_mean = c_mean+C_21_COR*R'*(R*(b_noise-Ax-model_error_mean));
        sigma_COR = sqrt(sigma_COR^2 - gamma*C_21_COR*(R'*(R*C_12_COR)));
    else
        c_mean = c_true;
    end
    
    if DETECTOR_DIST == 1
        cov_full = cov([model_error_vec;d_vec]');
        C_21_Tilt = cov_full(end,1:end-1);
        C_12_Tilt = cov_full(1:end-1,end);
        
        d_mean = d_mean+C_21_Tilt*R'*(R*(b_noise-Ax-model_error_mean));
        sigma_d = sqrt(sigma_d^2 - gamma*C_21_Tilt*(R'*(R*C_12_Tilt)));
    else
        d_mean = d_true;
    end
    
    if SOURCE_DIST == 1
        cov_full = cov([model_error_vec;s_vec]');
        C_21_DIST = cov_full(end,1:end-1);
        C_12_DIST = cov_full(1:end-1,end);
        
        s_mean = s_mean+C_21_DIST*R'*(R*(b_noise-Ax-model_error_mean));
        sigma_s = sqrt(sigma_s^2 - gamma*C_21_DIST*(R'*(R*C_12_DIST)));
    else
        s_mean = s_true;
    end
    
    %Update approximated forward model and projector id
    A = Build_Forward_Operator(ray_config,vol_geom,gpu,angles,p,c_mean,d_mean,s_mean);
    proj_id = Build_proj_id(ray_config,vol_geom,gpu,angles,p,c_mean,d_mean,s_mean);
    
    Ax = A*x;
    
    %RECONSTRUTION STEP
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Preallocate model parameter samples
    model_error_vec = zeros(M,S);
    c_vec = zeros(1,S);
    d_vec = zeros(1,S);
    s_vec = zeros(1,S);
    
    %Sample model parameters from updated priors
    for i = 1:S
        if COR == 1
            %Sample c from prior distribution
            c = normrnd(c_mean,sigma_COR);
            c_vec(i) = c;
        else
            c = c_true;
        end
        
        if DETECTOR_DIST == 1
            %Sample detector tilt from prior distribution
            d = normrnd(d_mean,sigma_d);
            d_vec(i) = d;
        else
            d = d_true;
        end
        
        if SOURCE_DIST == 1
            %Sample source distance from prior distribution
            s = normrnd(s_mean,sigma_s);
            s_vec(i) = s;
        else
            s = s_true;
        end
        
        %Get forward operator
        A_samp = Build_Forward_Operator(ray_config,vol_geom,gpu,angles,p,c,d,s);
        
        %Compute Model Error
        model_error_vec(:,i) = A_samp*x-Ax;
    end
    
    %Compute mean of model error samples
    model_error_mean = mean(model_error_vec,2);
    
    %Compute cholesky factorization of (C+1/lambda I)^(-1) using rank
    %1 updates (C is empirical covariance matrix).
    R = sqrt(lambda)*eye(M);
    for i=1:S
        u = (model_error_vec(:,i)-model_error_mean)/sqrt(S-1);
        w = R'*(R*u);
        perturb = w/sqrt(1+u'*w);
        R = cholupdate(R,perturb,'-');
    end
    
    %Create modified system matrix and modifed b-vector
    b_tilde = R*(b_noise-model_error_mean);
    
    %Update MAP solution
    x = Update_MAP_Recon(A,proj_id,b_tilde,alpha,lambda,R,p,angles,x,maxiters,nonneg,fig_show,reg_term,options);
    
    %Display parameter updates
    if COR == 1
        disp(['COR estimate: ' num2str(c_mean) '; std: ' num2str(sigma_COR)])
    end
    if DETECTOR_DIST == 1
        disp(['Detector-Origin Distance estimate: ' num2str(d_mean) '; std: ' num2str(sigma_d)])
    end
    if SOURCE_DIST == 1
        disp(['Source-Origin Distance estimate: ' num2str(s_mean) '; std: ' num2str(sigma_s)])
    end
    
    %Save x and COR estimate
    x_samps(:,k) = x;
    c_samps(:,k) = c_mean;
    d_samps(:,k) = d_mean;
    s_samps(:,k) = s_mean;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%SAVE OUTPUT

out.x_final_recon = x;
out.x_samps = x_samps;

if COR == 1
    out.COR_samps = c_samps;
end

if DETECTOR_DIST == 1
    out.DETECTORDIST_samps = d_samps;
end

if SOURCE_DIST == 1
    out.SOURCEDIST_samps = s_samps;
end
end

function A = Build_Forward_Operator(ray_config,vol_geom,gpu,angles,p,c,d,s)
%Function that builds the forward operator A

%INPUT:
%ray_config: string that tells if parallel or fanbeam
%volume_geometry: astra volume geometry object
%angles: projection angles in radians
%p: number of rays
%c: COR parameter
%da: Tilt parameter
%s: Source distance parameter

%OUTPUT:
%A: forward operator

vectors = zeros(length(angles),6);

if strcmp(ray_config,'parallel')
    vectors(:,1) = -sin(angles);
    vectors(:,2) = cos(angles);
    vectors(:,3) = c*cos(angles);
    vectors(:,4) = c*sin(angles);
    vectors(:,5) = cos(angles);
    vectors(:,6) = sin(angles);
    
    proj_geom = astra_create_proj_geom('parallel_vec',p,vectors);
elseif strcmp(ray_config,'fanbeam')
    vectors(:,1) = sin(angles)*s;
    vectors(:,2) = -cos(angles)*s;
    vectors(:,3) = -sin(angles)*d + c*cos(angles);
    vectors(:,4) = cos(angles)*d + c*sin(angles);
    vectors(:,5) = cos(angles);
    vectors(:,6) = sin(angles);
    
    proj_geom = astra_create_proj_geom('fanflat_vec',p,vectors);
end
        
if gpu == 1
    A = opTomo('cuda',proj_geom,vol_geom);
else
    if strcmp(ray_config,'parallel')
        A = opTomo('linear',proj_geom,vol_geom);
    elseif strcmp(ray_config,'fanbeam')
        A = opTomo('line_fanflat',proj_geom,vol_geom);
    end
end
end

function proj_id = Build_proj_id(ray_config,vol_geom,gpu,angles,p,c,d,s)
vectors = zeros(length(angles),6);

if strcmp(ray_config,'parallel')
    vectors(:,1) = -sin(angles);
    vectors(:,2) = cos(angles);
    vectors(:,3) = c*cos(angles);
    vectors(:,4) = c*sin(angles);
    vectors(:,5) = cos(angles);
    vectors(:,6) = sin(angles);
    
    proj_geom = astra_create_proj_geom('parallel_vec',p,vectors);
elseif strcmp(ray_config,'fanbeam')
    vectors(:,1) = sin(angles)*s;
    vectors(:,2) = -cos(angles)*s;
    vectors(:,3) = -sin(angles)*d + c*cos(angles);
    vectors(:,4) = cos(angles)*d + c*sin(angles);
    vectors(:,5) = cos(angles);
    vectors(:,6) = sin(angles);
    
    proj_geom = astra_create_proj_geom('fanflat_vec',p,vectors);
end
        
if gpu == 1
    proj_id = astra_create_projector('cuda',proj_geom,vol_geom);
else
    if strcmp(ray_config,'parallel')
        proj_id = astra_create_projector('linear',proj_geom,vol_geom);
    elseif strcmp(ray_config,'fanbeam')
        proj_id = astra_create_projector('line_fanflat',proj_geom,vol_geom);
    end
end
end

function x = Update_MAP_Recon(A,proj_id,b_tilde,alpha,lambda,R,p,theta,x0,maxiters,nonneg,fig_show,reg_term,options)
%Function that updates the MAP estimate based on the oupdated model errors

%The algorithms are chosen are follows:

%TV without nonnegativity: Accelerated Gradient Descent
%TV with nonnegativity: Projected Accelerated Gradient Descent
%tikh/gentikh without nonnegativity: PCG
%tikh/gentikh with nonnegativity: Projected Accelerated Gradient Descent

%INPUT:
%A: Forward operator
%proj_id: projector id (needed for PCG)
%b_tilde: Modified sinogram
%alpha: Regularization parameter
%lambda: noise precision
%R: Inverse cholesky factorization of covariance matrix
%p: number of rays
%x0: Initial guess
%maxiters: maximum number of iterations
%fig_show: specify if MAP update should be plotted
%options: Struct containing algorithm options

if strcmp(reg_term,'TV')
    %Solve smooth TV problem using FISTA
    options.x0 = x0;
    options.maxiters = maxiters;
    x = L2TV_acc_smooth_nuisance(A,b_tilde,alpha,R,options);
    N = sqrt(length(x0));
        
    if fig_show == 1
        figure
        imagesc(reshape(x,N,N)), axis image, colorbar
        title(reg_term)
    end
else
    if nonneg == 1
        %Solve constrained problem using FISTA
        options.x0 = x0;
        options.maxiters = maxiters;
        h = options.h;
        reg_param = alpha*lambda*h^2;
        N = sqrt(length(x0));
        u = zeros(N^2,1);
        D = options.D;
        x = fista_Gen_tikh_nuisance(A,D,b_tilde,u,reg_param,R,options);
        
        if fig_show == 1
            figure
            imagesc(reshape(x,N,N)), axis image, colorbar
            title(reg_term)
        end
    else
        %Solve unconstrained problem using PCG
        h = options.h;
        reg_param = lambda*alpha*h^2;
        ATb_tilde = A'*(R'*b_tilde);
        D = options.D;
        N = sqrt(length(x0));
            
        x = pcg(@nuisance_pcg_full,ATb_tilde,[],maxiters,[],[],x0,proj_id,R,D'*D,reg_param,theta,p);
            
        if fig_show == 1
            figure
            imagesc(reshape(x,N,N)), axis image, colorbar
            title(reg_term)
        end
    end
end
end