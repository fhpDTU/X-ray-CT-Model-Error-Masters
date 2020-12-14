function out = Joint_Recon_Model_Error_v2(setup)
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
N =                 setup.N;                %Pixelgrid (N x N)
theta =             setup.theta;            %Projection angles in degrees
p =                 setup.p;                %Number of rays
ray_config =        setup.ray_config;       %Specify the ray configuration (parallel or fanbeam)

%Model Parameters
c_true =            setup.c_true;           %The true Center of Rotation (COR) parameter (physical length)
d_true =            setup.d_true;           %The true Detector-Origin distance (DD) (physical length)
s_true =            setup.s_true;           %The true Source-Origin distance (SD) (physical length)
t_true =            setup.t_true;           %The true Detector Tilt (degrees)

c0 =                setup.c0;               %Initial guess for COR parameter (physical length)
d0 =                setup.d0;               %Initial guess for DD parameter (physical length)
s0 =                setup.s0;               %Initial guess for SD parameter (physical length)
t0 =                setup.t0;               %Initial guess for TILT parameter (degrees)

sigma_COR_0 =       setup.sigma_COR_0;      %Initial standard deviation of COR prior
sigma_d_0 =         setup.sigma_d_0;        %Initial standard deviation of DD prior
sigma_s_0 =         setup.sigma_s_0;        %Initial standard deviation of SD prior
sigma_t_0 =         setup.sigma_t_0;        %Initial standard deviation of TILT prior

COR =               setup.COR;              %Boolean specifying if COR should be estimated
DETECTOR_DIST =     setup.DETECTOR_DIST;    %Boolean specifying if DD should be estimated
SOURCE_DIST =       setup.SOURCE_DIST;      %Boolean specifying if SD should be estimated
TILT =              setup.TILT;             %Boolean specifying if TILT should be estimated

%REGULARIZATION TERM AND SOLVER PARAMETERS
reg_term =          setup.reg_term;         %Regularization term (tikh, gentikh or TV)
nonneg =            setup.nonneg;           %Boolean specifying if non-negativity constraints
alpha =             setup.alpha;            %Regularization parameter for inner problem
maxiters =          setup.maxiters;         %Maximum number of FISTA iterations for inner problem

%AUXILLARY PARAMETERS
N_out =             setup.N_out;            %Number of outer iterations
S =                 setup.S;                %Number of samples for reconstruction step
S_update =          setup.S_update;         %Number of samples for parameter update step
gamma =             setup.gamma;            %Variance relaxation parameter
gpu =               setup.gpu;              %Specify if gpu (1 if gpu 0 if not)
fig_show =          setup.fig_show;         %Specify if we want to show figures for each outer iteration

b_noise =           setup.b_noise;          %noisy sinogram
lambda =            setup.lambda;           %noise precision of measurement errors
x_ini =             setup.x_ini;            %Reconstruction with initial model parameters

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

if COR == 1
    c = c0;
    c_samps = zeros(1,N_out);
else
    c = c_true;
end

if DETECTOR_DIST == 1
    d = d0;
    d_samps = zeros(1,N_out);
else
    d = d_true;
end

if SOURCE_DIST == 1
    s = s0;
    s_samps = zeros(1,N_out);
else
    s = s_true;
end

if TILT == 1
    t = t0;
    t_samps = zeros(1,N_out);
else
    t = t_true;
end

M = p*length(theta);
angles = theta*pi/180;
vol_geom = astra_create_vol_geom(N,N,-1,1,-1,1);

%Compute initial forward operator
A = Build_Forward_Operator(ray_config,vol_geom,gpu,angles,p,c,d,s,t);

%Preallocate arrays for samples
x_samps = zeros(N^2,N_out);

sigma_COR = sigma_COR_0;                %Initial standard deviation of COR
sigma_d = sigma_d_0;                    %Initial standard deviation of DD
sigma_s = sigma_s_0;                    %Initial standard deviation of SD
sigma_t = sigma_t_0;                    %Initial standard deviation for TILT

x = x_ini;                              %Initial reconstruction estimate                   

%Run the algorithm
for k=1:N_out
    loading_bar = waitbar(k/N_out);
    
    %PARAMETER UPDATE STEP
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Preallocate space for samples
    model_error_vec = zeros(M,S_update);
    
    Ax = A*x;
    c_vec = zeros(1,S_update);
    s_vec = zeros(1,S_update);
    d_vec = zeros(1,S_update);
    t_vec = zeros(1,S_update);
    
    %Sample S_update instances of model error
    for i = 1:S_update
        if COR == 1
            %Sample c from prior distribution
            c_samp = normrnd(c,sigma_COR);
            c_vec(i) = c_samp;
        else
            c_samp = c;
        end
        
        if DETECTOR_DIST == 1
            %Sample detector tilt from prior distribution
            d_samp = normrnd(d,sigma_d);
            d_vec(i) = d_samp;
        else
            d_samp = d;
        end
        
        if SOURCE_DIST == 1
            %Sample source distance from prior distribution
            s_samp = normrnd(s,sigma_s);
            s_vec(i) = s_samp;
        else
            s_samp = s;
        end
        
        if TILT == 1
            %Sample tilt parameter from prior distribution
            t_samp = normrnd(t,sigma_t);
            t_vec(i) = t_samp;
        else
            t_samp = t;
        end
        
        %Build sampled forward operator
        A_samp = Build_Forward_Operator(ray_config,vol_geom,gpu,angles,p,c_samp,d_samp,s_samp,t_samp);
        
        %Compute Sample Model Error
        model_error_vec(:,i) = A_samp*x-Ax;
    end
    
    %Compute mean of model error samples
    model_error_mean = mean(model_error_vec,2);
    u_mat = (model_error_vec - model_error_mean)/(sqrt(S_update-1));
    uTu = u_mat'*u_mat;
    %Update Model Error Parameters
    if COR == 1
        C_21_COR = 1/(S_update-1)*(model_error_vec-model_error_mean)*(c_vec-mean(c_vec))';
        
        %Compute conditional expectation and covariance
        c = c+C_21_COR'*C_inv_mult(b_noise-Ax-model_error_mean,u_mat,uTu,lambda);
        sigma_COR = sqrt(sigma_COR^2 - gamma*C_21_COR'*C_inv_mult(C_21_COR,u_mat,uTu,lambda));
        c_samps(k) = c;
    end
    
    if DETECTOR_DIST == 1
        C_21_DD = 1/(S_update-1)*(model_error_vec-model_error_mean)*(d_vec-mean(d_vec))';
        
        d = d+C_21_DD'*C_inv_mult(b_noise-Ax-model_error_mean,u_mat,uTu,lambda);
        sigma_d = sqrt(sigma_d^2 - gamma*C_21_DD'*C_inv_mult(C_21_DD,u_mat,uTu,lambda));
        d_samps(k) = d;
    end
    
    if SOURCE_DIST == 1
        C_21_SD = 1/(S_update-1)*(model_error_vec-model_error_mean)*(s_vec-mean(s_vec))';
        
        s = s+C_21_SD'*C_inv_mult(b_noise-Ax-model_error_mean,u_mat,uTu,lambda);
        sigma_s = sqrt(sigma_s^2 - gamma*C_21_SD'*C_inv_mult(C_21_SD,u_mat,uTu,lambda));
        s_samps(k) = s;
    end
    
    if TILT == 1
        C_21_TILT = 1/(S_update-1)*(model_error_vec-model_error_mean)*(t_vec-mean(t_vec))';
        
        t = t+C_21_TILT'*C_inv_mult(b_noise-Ax-model_error_mean,u_mat,uTu,lambda);
        sigma_t = sqrt(sigma_t^2 - gamma*C_21_TILT'*C_inv_mult(C_21_TILT,u_mat,uTu,lambda));
        t_samps(k) = t;
    end
    
    %Update approximated forward model
    A = Build_Forward_Operator(ray_config,vol_geom,gpu,angles,p,c,d,s,t);
    
    Ax = A*x;
    
    %RECONSTRUCTION STEP
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Preallocate model parameter samples
    model_error_vec = zeros(M,S);
    c_vec = zeros(1,S);
    d_vec = zeros(1,S);
    s_vec = zeros(1,S);
    t_vec = zeros(1,S);
    
    %Sample model parameters from updated priors
    for i = 1:S
        if COR == 1
            %Sample c from prior distribution
            c_samp = normrnd(c,sigma_COR);
            c_vec(i) = c_samp;
        else
            c_samp = c;
        end
        
        if DETECTOR_DIST == 1
            %Sample detector tilt from prior distribution
            d_samp = normrnd(d,sigma_d);
            d_vec(i) = d_samp;
        else
            d_samp = d;
        end
        
        if SOURCE_DIST == 1
            %Sample source distance from prior distribution
            s_samp = normrnd(s,sigma_s);
            s_vec(i) = s_samp;
        else
            s_samp = s;
        end
        
        if TILT == 1
            %Sample TILT from prior distribution
            t_samp = normrnd(t,sigma_t);
            t_vec(i) = t_samp;
        else
            t_samp = t;
        end
        
        %Get forward operator
        A_samp = Build_Forward_Operator(ray_config,vol_geom,gpu,angles,p,c_samp,d_samp,s_samp,t_samp);
        
        %Compute Model Error
        model_error_vec(:,i) = A_samp*x-Ax;
    end
    
    %Compute mean of model error samples
    model_error_mean = mean(model_error_vec,2);
    u_mat = (model_error_vec-model_error_mean)/sqrt(S-1);
    
    %Create modified system matrix and modifed b-vector
    b_tilde = b_noise-model_error_mean;
    
    %Update MAP solution
    x = Update_MAP_Recon(A,b_tilde,alpha,lambda,u_mat,x,maxiters,fig_show,reg_term,options);
    
    %Display parameter updates
    if COR == 1
        disp(['COR estimate: ' num2str(c) '; std: ' num2str(sigma_COR)])
    end
    if DETECTOR_DIST == 1
        disp(['Detector-Origin Distance estimate: ' num2str(d) '; std: ' num2str(sigma_d)])
    end
    if SOURCE_DIST == 1
        disp(['Source-Origin Distance estimate: ' num2str(s) '; std: ' num2str(sigma_s)])
    end
    if TILT == 1
        disp(['Tilt parameter estimate: ' num2str(t) '; std: ' num2str(sigma_t)])
    end
    
    %Save x sample
    x_samps(:,k) = x;
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

if TILT == 1
    out.TILT_samps = t_samps;
end
end

function A = Build_Forward_Operator(ray_config,vol_geom,gpu,angles,p,c,d,s,t)
%Function that builds the forward operator A

%INPUT:
%ray_config: string that tells if parallel or fanbeam
%volume_geometry: astra volume geometry object
%angles: projection angles in radians
%p: number of rays
%c: COR parameter
%d: Detector Distance parameter (Fanbeam only)
%s: Source Distance parameter (Fanbeam only)
%t: Tilt parameter

%OUTPUT:
%A: forward operator

vectors = zeros(length(angles),6);

if strcmp(ray_config,'parallel')
    vectors(:,1) = -sin(angles);
    vectors(:,2) = cos(angles);
    vectors(:,3) = c*cos(angles);
    vectors(:,4) = c*sin(angles);
    vectors(:,5) = cos(angles + t/180*pi)*3/p;
    vectors(:,6) = sin(angles + t/180*pi)*3/p;
    
    proj_geom = astra_create_proj_geom('parallel_vec',p,vectors);
elseif strcmp(ray_config,'fanbeam')
    vectors(:,1) = sin(angles)*s;
    vectors(:,2) = -cos(angles)*s;
    vectors(:,3) = -sin(angles)*d + c*cos(angles);
    vectors(:,4) = cos(angles)*d + c*sin(angles);
    vectors(:,5) = cos(angles + t/180*pi)*3/p;
    vectors(:,6) = sin(angles + t/180*pi)*3/p;
    
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
    vectors(:,5) = cos(angles)*3/p;
    vectors(:,6) = sin(angles)*3/p;
    
    proj_geom = astra_create_proj_geom('parallel_vec',p,vectors);
elseif strcmp(ray_config,'fanbeam')
    vectors(:,1) = sin(angles)*s;
    vectors(:,2) = -cos(angles)*s;
    vectors(:,3) = -sin(angles)*d + c*cos(angles);
    vectors(:,4) = cos(angles)*d + c*sin(angles);
    vectors(:,5) = cos(angles)*3/p;
    vectors(:,6) = sin(angles)*3/p;
    
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

function x = Update_MAP_Recon(A,b_tilde,alpha,lambda,u_mat,x0,maxiters,fig_show,reg_term,options)
%Function that updates the MAP estimate based on the outdated model errors

%The algorithms are chosen are follows:

%TV without nonnegativity: Accelerated Gradient Descent
%TV with nonnegativity: Projected Accelerated Gradient Descent
%tikh/gentikh without nonnegativity: Accelerated Gradient Descent
%tikh/gentikh with nonnegativity: Projected Accelerated Gradient Descent

%INPUT:
%A: Forward operator
%proj_id: projector id (needed for PCG)
%b_tilde: Modified sinogram
%alpha: Regularization parameter
%lambda: noise precision
%u_mat: Matrix containing translated model error samples
%p: number of rays
%x0: Initial guess
%maxiters: maximum number of iterations
%fig_show: specify if MAP update should be plotted
%options: Struct containing algorithm options

if strcmp(reg_term,'TV')
    %Solve smooth TV problem using FISTA
    options.x0 = x0;
    options.maxiters = maxiters;
    x = L2TV_acc_smooth_nuisance_v2(A,b_tilde,u_mat,alpha,lambda,options);
    N = sqrt(length(x0));
        
    if fig_show == 1
        figure
        imagesc(reshape(x,N,N)), axis image, colorbar
        title(reg_term)
    end
else
    %Solve problem using FISTA
    options.x0 = x0;
    options.maxiters = maxiters;
    h = options.h;
    reg_param = alpha*lambda*h^2;
    N = sqrt(length(x0));
    D = options.D;
    x = fista_Gen_tikh_nuisance_v2(A,D,b_tilde,u_mat,reg_param,lambda,options);
        
    if fig_show == 1
        figure
        imagesc(reshape(x,N,N)), axis image, colorbar
        title(reg_term)
    end
end
end