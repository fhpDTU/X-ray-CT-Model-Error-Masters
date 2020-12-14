function res = Additive_Modelling_Error_Wrap_v2(setup)
%This function solves the joint-reconstruction problem with modelling
%errors using algorithm 5 from the thesis. The method is akin to the one
%proposed by Nicolai in his paper only for other types of modelling error.

%OUTPUT
%res: struct containing the final reconstruction, alongside the parameter
%estimates, and all the geometry parameters. Also contains the
%reconstruction with initial and true geometric parameters

%INPUT
%setup: struct containing all information on the geometry, reconstruction
%algorithms, data simulation and model parameters. See algorithm function
%or driver script for more information

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%UNPACK THE VARIABLES NECCESSARY FOR INITIALIZATION

%GEOMETRY
N =                 setup.N;                %GridSize
theta =             setup.theta;            %Projection angles in degrees
p =                 setup.p;                %Number of rays
ray_config =        setup.ray_config;       %Specify the ray configuration (parallel or fanbeam)

%MODELLING ERROR
c_true =            setup.c_true;           %True COR parameter (physical unit)
d_true =            setup.d_true;           %True DD parameter (physical unit)
s_true =            setup.s_true;           %True SD parameter (physical unit)
t_true =            setup.t_true;           %True Tilt parameter (degrees)

c0 =                setup.c0;               %Initial guess for COR parameter (physical unit)
d0 =                setup.d0;               %Initial guess for DD parameter (physical unit)
s0 =                setup.s0;               %Initial guess for SD parameter (physical unit)
t0 =                setup.t0;               %Initial guess for Tilt parameter (degrees)

%REGULARIZATION TERM AND SOLVER PARAMETERS
reg_term =          setup.reg_term;         %Specify regularization term
nonneg =            setup.nonneg;           %Specify if nonnegativity constraints
alpha_ini =         setup.alpha_ini;        %Regularization parameter for initial reconstruction
maxiters_ini =      setup.maxiters_ini;     %Maximum number of iteration for initial reconstruction

%DATA SIMULATION
sino =              setup.sino;             %Specify if analytical or discretized sinogram
noise_level =       setup.noise_level;      %Specify noise_level
inverse_factor =    setup.inverse_factor;   %Specify the fineness of data generation grid
gpu =               setup.gpu;              %Specify if gpu (1 if gpu 0 if not)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N_fine = round(inverse_factor*N);
x_fine = phantom(N_fine);

%Generate noise free sinogram with true model parameters
[~,b] = Generate_Model_Problem(ray_config,N_fine,x_fine,theta,p,gpu,sino,c_true,d_true,s_true,t_true);

%Add Gaussian Noise to sinogram
e = randn(size(b));
e = noise_level*norm(b)*e/(norm(e));
b = b + e;

%Compute exact noise precision
lambda = 1/var(e);

setup.lambda = lambda;
setup.b_noise = b;

%Generate initial forward operator
angles = theta*pi/180;
vol_geom = astra_create_vol_geom(N,N,-1,1,-1,1);
A = Build_Forward_Operator(ray_config,vol_geom,gpu,angles,p,c0,d0,s0,t0);

%Compute initial reconstruction
x_initial_recon = MAP_recon(A,b,alpha_ini,zeros(N^2,1),reg_term,maxiters_ini,nonneg);
setup.x_ini = x_initial_recon;

%Do joint reconstruction
out = Joint_Recon_Model_Error_v2(setup);

%Compute reconstruction with true model parameters
A_true = Build_Forward_Operator(ray_config,vol_geom,gpu,angles,p,c_true,d_true,s_true,t_true);
x_true_model = MAP_recon(A_true,b,alpha_ini,zeros(N^2,1),reg_term,maxiters_ini,nonneg);

out.x_initial_recon = x_initial_recon;
out.x_true_model = x_true_model;

res.out = out;
res.setup = setup;

astra_clear;
end

function A = Build_Forward_Operator(ray_config,vol_geom,gpu,angles,p,c,d,s,t)
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