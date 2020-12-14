function out = Joint_Recon_Model_Error_Param2(setup)
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
detector_width =    setup.detector_width;   %Width of detector

%Model Parameters
dx_true =           setup.dx_true;           %The true x-coordinate of detector center (dx)
dy_true =           setup.dy_true;           %The true y-coordiante of detector center (dy)
sx_true =           setup.sx_true;           %The true x-coordinate of source location (sx)
sy_true =           setup.sy_true;           %The true y-coordinate of source location (sy)

dx0 =               setup.dx0;               %Initial guess for dx
dy0 =               setup.dy0;               %Initial guess for dy
sx0 =               setup.sx0;               %Initial guess for sx
sy0 =               setup.sy0;               %Initial guess for sy

sigma_dx_0 =        setup.sigma_dx_0;        %Initial standard deviation of dx prior
sigma_dy_0 =        setup.sigma_dy_0;        %Initial standard deviation of dy prior
sigma_sx_0 =        setup.sigma_sx_0;        %Initial standard deviation of sx prior
sigma_sy_0 =        setup.sigma_sy_0;        %Initial standard deviation of sy prior

DETECTOR_X =        setup.DETECTOR_X;        %Boolean specifying if dx should be estimated
DETECTOR_Y =        setup.DETECTOR_Y;        %Boolean specifying if dy should be estimated
SOURCE_X =          setup.SOURCE_X;          %Boolean specifying if sx should be estimated
SOURCE_Y =          setup.SOURCE_Y;          %Boolean specifying if sy should be estimated

%REGULARIZATION TERM AND SOLVER PARAMETERS
reg_term =          setup.reg_term;          %Regularization term (tikh, gentikh or TV)
nonneg =            setup.nonneg;            %Boolean specifying if non-negativity constraints
alpha =             setup.alpha;             %Regularization parameter for inner problem
maxiters =          setup.maxiters;          %Maximum number of FISTA iterations for inner problem

%AUXILLARY PARAMETERS
N_out =             setup.N_out;             %Number of outer iterations
S =                 setup.S;                 %Number of samples for reconstruction step
S_update =          setup.S_update;          %Number of samples for parameter update step
gamma =             setup.gamma;             %Variance relaxation parameter
gpu =               setup.gpu;               %Specify if gpu (1 if gpu 0 if not)
fig_show =          setup.fig_show;          %Specify if we want to show figures for each outer iteration

b_noise =           setup.b_noise;           %noisy sinogram
lambda =            setup.lambda;            %noise precision of measurement errors
x_ini =             setup.x_ini;             %Reconstruction with initial model parameters

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

if DETECTOR_X == 1
    dx = dx0;
    dx_samps = zeros(1,N_out);
else
    dx = dx_true;
end

if DETECTOR_Y == 1
    dy = dy0;
    dy_samps = zeros(1,N_out);
else
    dy = dy_true;
end

if SOURCE_X == 1
    sx = sx0;
    sx_samps = zeros(1,N_out);
else
    sx = sx_true;
end

if SOURCE_Y == 1
    sy = sy0;
    sy_samps = zeros(1,N_out);
else
    sy = sy_true;
end

M = p*length(theta);
angles = theta*pi/180;
vol_geom = astra_create_vol_geom(N,N,-1,1,-1,1);

%Compute initial forward operator
A = Build_Forward_Operator(ray_config,vol_geom,gpu,angles,p,dx,dy,sx,sy,detector_width);

%Preallocate arrays for samples
x_samps = zeros(N^2,N_out);

sigma_dx = sigma_dx_0;                 %Initial standard deviation of COR
sigma_dy = sigma_dy_0;                    %Initial standard deviation of DD
sigma_sx = sigma_sx_0;                    %Initial standard deviation of SD
sigma_sy = sigma_sy_0;                    %Initial standard deviation for TILT

x = x_ini;                              %Initial reconstruction estimate                   

%Run the algorithm
for k=1:N_out
    loading_bar = waitbar(k/N_out);
    
    %PARAMETER UPDATE STEP
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Preallocate space for samples
    model_error_vec = zeros(M,S_update);
    
    Ax = A*x;
    dx_vec = zeros(1,S_update);
    dy_vec = zeros(1,S_update);
    sx_vec = zeros(1,S_update);
    sy_vec = zeros(1,S_update);
    
    %Sample S_update instances of model error
    for i = 1:S_update
        if DETECTOR_X == 1
            %Sample detector x-coordinate from prior distribution
            dx_samp = normrnd(dx,sigma_dx);
            dx_vec(i) = dx_samp;
        else
            dx_samp = dx;
        end
        
        if DETECTOR_Y == 1
            %Sample detector y-coordinate from prior distribution
            dy_samp = normrnd(dy,sigma_dy);
            dy_vec(i) = dy_samp;
        else
            dy_samp = dy;
        end
        
        if SOURCE_X == 1
            %Sample source x-coordinate from prior distribution
            sx_samp = normrnd(sx,sigma_sx);
            sx_vec(i) = sx_samp;
        else
            sx_samp = sx;
        end
        
        if SOURCE_Y == 1
            %Sample source y-coordinate from prior distribution
            sy_samp = normrnd(sy,sigma_sy);
            sy_vec(i) = sy_samp;
        else
            sy_samp = sy;
        end
        
        %Build sampled forward operator
        A_samp = Build_Forward_Operator(ray_config,vol_geom,gpu,angles,p,dx_samp,dy_samp,sx_samp,sy_samp,detector_width);
        
        %Compute Sample Model Error
        model_error_vec(:,i) = A_samp*x-Ax;
    end
    
    %Compute mean of model error samples
    model_error_mean = mean(model_error_vec,2);
    u_mat = (model_error_vec - model_error_mean)/(sqrt(S_update-1));
    uTu = u_mat'*u_mat;
    %Update Model Error Parameters
    if DETECTOR_X == 1
        C_21_dx = 1/(S_update-1)*(model_error_vec-model_error_mean)*(dx_vec-mean(dx_vec))';
        
        %Compute conditional expectation and covariance
        dx = dx+C_21_dx'*C_inv_mult(b_noise-Ax-model_error_mean,u_mat,uTu,lambda);
        sigma_dx = sqrt(sigma_dx^2 - gamma*C_21_dx'*C_inv_mult(C_21_dx,u_mat,uTu,lambda));
        dx_samps(k) = dx;
    end
    
    if DETECTOR_Y == 1
        C_21_dy = 1/(S_update-1)*(model_error_vec-model_error_mean)*(dy_vec-mean(dy_vec))';
        
        dy = dy+C_21_dy'*C_inv_mult(b_noise-Ax-model_error_mean,u_mat,uTu,lambda);
        sigma_dy = sqrt(sigma_dy^2 - gamma*C_21_dy'*C_inv_mult(C_21_dy,u_mat,uTu,lambda));
        dy_samps(k) = dy;
    end
    
    if SOURCE_X == 1
        C_21_sx = 1/(S_update-1)*(model_error_vec-model_error_mean)*(sx_vec-mean(sx_vec))';
        
        sx = sx+C_21_sx'*C_inv_mult(b_noise-Ax-model_error_mean,u_mat,uTu,lambda);
        sigma_sx = sqrt(sigma_sx^2 - gamma*C_21_sx'*C_inv_mult(C_21_sx,u_mat,uTu,lambda));
        sx_samps(k) = sx;
    end
    
    if SOURCE_Y == 1
        C_21_sy = 1/(S_update-1)*(model_error_vec-model_error_mean)*(sy_vec-mean(sy_vec))';
        
        sy = sy+C_21_sy'*C_inv_mult(b_noise-Ax-model_error_mean,u_mat,uTu,lambda);
        sigma_sy = sqrt(sigma_sy^2 - gamma*C_21_sy'*C_inv_mult(C_21_sy,u_mat,uTu,lambda));
        sy_samps(k) = sy;
    end
    
    %Update approximated forward model
    A = Build_Forward_Operator(ray_config,vol_geom,gpu,angles,p,dx,dy,sx,sy,detector_width);
    
    Ax = A*x;
    
    %RECONSTRUCTION STEP
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Preallocate model parameter samples
    model_error_vec = zeros(M,S);
    dx_vec = zeros(1,S);
    dy_vec = zeros(1,S);
    sx_vec = zeros(1,S);
    sy_vec = zeros(1,S);
    
    %Sample model parameters from updated priors
    for i = 1:S
        if DETECTOR_X == 1
            %Sample c from prior distribution
            dx_samp = normrnd(dx,sigma_dx);
            dx_vec(i) = dx_samp;
        else
            dx_samp = dx;
        end
        
        if DETECTOR_Y == 1
            %Sample detector tilt from prior distribution
            dy_samp = normrnd(dy,sigma_dy);
            dy_vec(i) = dy_samp;
        else
            dy_samp = dy;
        end
        
        if SOURCE_X == 1
            %Sample source distance from prior distribution
            sx_samp = normrnd(sx,sigma_sx);
            sx_vec(i) = sx_samp;
        else
            sx_samp = sx;
        end
        
        if SOURCE_Y == 1
            %Sample TILT from prior distribution
            sy_samp = normrnd(sy,sigma_sy);
            sy_vec(i) = sy_samp;
        else
            sy_samp = sy;
        end
        
        %Get forward operator
        A_samp = Build_Forward_Operator(ray_config,vol_geom,gpu,angles,p,dx_samp,dy_samp,sx_samp,sy_samp,detector_width);
        
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
    if DETECTOR_X == 1
        disp(['dx estimate: ' num2str(dx) '; std: ' num2str(sigma_dx)])
    end
    if DETECTOR_Y == 1
        disp(['dy estimate: ' num2str(dy) '; std: ' num2str(sigma_dy)])
    end
    if SOURCE_X == 1
        disp(['sx estimate: ' num2str(sx) '; std: ' num2str(sigma_sx)])
    end
    if SOURCE_Y == 1
        disp(['sy estimate: ' num2str(sy) '; std: ' num2str(sigma_sy)])
    end
    
    %Save x sample
    x_samps(:,k) = x;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%SAVE OUTPUT

out.x_final_recon = x;
out.x_samps = x_samps;

if DETECTOR_X == 1
    out.dx_samps = dx_samps;
end

if DETECTOR_Y == 1
    out.dy_samps = dy_samps;
end

if SOURCE_X == 1
    out.sx_samps = sx_samps;
end

if SOURCE_Y == 1
    out.sy_samps = sy_samps;
end
end

function A = Build_Forward_Operator(ray_config,vol_geom,gpu,angles,p,dx,dy,sx,sy,detector_width)
%Function that builds the forward operator A

%INPUT:
%ray_config: string that tells if parallel or fanbeam
%volume_geometry: astra volume geometry object
%angles: projection angles in radians
%p: number of rays
%dx: x-coordinate of detector center
%dy: y-coordinate of detector center (Fanbeam only)
%sx: x-coordinate of source (Fanbeam only)
%sy: y-coordinate of source (Fanbeam only)
%detector_width: Width of detector

%OUTPUT:
%A: forward operator

vectors = zeros(length(angles),6);

if strcmp(ray_config,'parallel')
    vectors(:,1) = -sin(angles);
    vectors(:,2) = cos(angles);
    vectors(:,3) = dx*cos(angles);
    vectors(:,4) = dx*sin(angles);
    vectors(:,5) = cos(angles)*detector_width/p;
    vectors(:,6) = sin(angles)*detector_width/p;
    
    proj_geom = astra_create_proj_geom('parallel_vec',p,vectors);
elseif strcmp(ray_config,'fanbeam')
    vectors(:,1) = sin(angles)*sy + cos(angles)*sx;
    vectors(:,2) = -cos(angles)*sy + sin(angles)*sx;
    vectors(:,3) = -sin(angles)*dy + dx*cos(angles);
    vectors(:,4) = cos(angles)*dy + dx*sin(angles);
    vectors(:,5) = cos(angles)*detector_width/p;
    vectors(:,6) = sin(angles)*detector_width/p;
    
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