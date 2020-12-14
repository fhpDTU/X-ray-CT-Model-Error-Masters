function res = Additive_Modelling_Error_Wrap(setup)
%This function solves the joint-reconstruction problem with modelling
%errors using algorithm 5 from the thesis. The method is akin to the one
%proposed by Nicolai in his paper only for other types of modelling error.

%OUTPUT
%res: struct containing the final reconstruction, alongside the parameter
%estimates, and all the geometry parameters. Also contains the
%reconstruction with initial and true geometric parameters

%INPUT
%options: struct containing the following fields

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%UNPACK THE VARIABLES

%GEOMETRY
N =                 setup.N;                %GridSize
theta =             setup.theta;            %Projection angles in degrees
p =                 setup.p;                %Number of rays
ray_config =        setup.ray_config;       %Specify the ray configuration (parallel or fanbeam)
c_true =            setup.c_true;           %The true COR parameter (pixels)
d_true =            setup.d_true;           %The true Tilt parameter (degrees)
s_true =            setup.s_true;           %The true source distance (Some unit)

%MODELLING ERROR
c0 =                setup.c0;               %Initial guess for COR parameter (pixels)
d0 =                setup.d0;               %Initial guess for Tilt parameter (degrees)
s0 =                setup.s0;               %Initial guess for source distance (some unit)

%REGULARIZATION TERM AND SOLVER PARAMETERS
reg_term =          setup.reg_term;         %Specify regularization term
nonneg =            setup.nonneg;           %Specify if nonnegativity constraints
alpha_ini =         setup.alpha_ini;        %Initial regularization parameter
maxiters_ini =      setup.maxiters_ini;     %Maximum number of iteration for initial reconstruction

%DATA SIMULATION
sino =              setup.sino;             %Specify if analytical or discretized sinogram
x_true =            setup.x_true;           %Specify the true image
noise_level =       setup.noise_level;      %Specify noise_level
seed =              setup.seed;             %Seed
gpu =               setup.gpu;              %Specify if gpu (1 if gpu 0 if not)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Set seed
rng(seed);

%Generate noise free sinogram with true model parameters
[A_true,b] = Generate_Model_Problem(ray_config,N,x_true,theta,p,gpu,sino,seed,c_true,d_true,s_true);

%Add Gaussian Noise to sinogram
e = randn(size(b));
e = noise_level*norm(b)*e/(norm(e));
b = b + e;

%Compute exact noise precision
lambda = 1/var(e);

setup.lambda = lambda;
setup.b_noise = b;

%Generate Forward Model with initial guesses for model parameters
A = Generate_Model_Problem(ray_config,N,x_true,theta,p,gpu,sino,seed,c0,d0,s0);

%Compute initial reconstruction
x_initial_recon = MAP_recon(A,b,alpha_ini,zeros(N^2,1),reg_term,maxiters_ini,nonneg);
setup.x_ini = x_initial_recon;

%Do joint reconstruction
out = Joint_Recon_Model_Error(setup);

%Compute reconstruction with true model parameters
x_true_model = MAP_recon(A_true,b,alpha_ini,zeros(N^2,1),reg_term,maxiters_ini,nonneg);

out.x_initial_recon = x_initial_recon;
out.x_true_model = x_true_model;

res.out = out;
res.setup = setup;

astra_clear;
end