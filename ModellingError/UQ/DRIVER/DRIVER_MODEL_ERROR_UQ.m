%Driver script for UQ sampling method

f = fullfile('/zhome','94','f','108663','Desktop','Masters','MatlabCode');
addpath(genpath(f));

%Geometry details
setup.N = 128;                          %Reconstruction grid size
setup.theta = 0:2:358;                  %Projection angles (degrees)
setup.p = setup.N*1.5;                  %Number of detector pixels

%Model parameters
setup.SOURCE_X = 1;                     %Boolean specifying if x-coordinate of SOURCE should be estimated
setup.SOURCE_Y = 1;                     %Boolean specifying if y-coordinate of SOURCE should be estimated
setup.DETECTOR_X = 1;                   %Boolean specifying if x-coordinate of DETECTOR should be estimated    
setup.DETECTOR_Y = 0;                   %Boolean specifying if y-coordinate of DETECTOR should be estimated

setup.sx_true = 0.1;                    %True value of x-coordinate of SOURCE
setup.sy_true = 4;                      %True value of y-coordinate of SOURCE
setup.dx_true = 0.2;                    %True value of x-coordinate of DETECTOR
setup.dy_true = 0;                      %True value of y-coordinate of DETECTOR

detector_width = (setup.sy_true + setup.dy_true)/setup.sy_true*3; %Width of detector
setup.detector_width = detector_width;           

setup.sx0 = 0;                          %Initial guess for x-coordinate of SOURCE
setup.sy0 = 3;                          %Initial guess for y-coordinate of SOURCE
setup.dx0 = 0;                          %Initial guess for x-coordinate of DETECTOR
setup.dy0 = 9;                          %Initial guess for y-coordinate of DETECTOR

%Gibbs sampler parameters
setup.N_samples = 5000;                 %Number of total Gibbs samples
setup.N_iter = 30;                      %Number of FISTA iterations for x-sample

setup.alpha_delta = 1;                  %Shape parameter for delta prior
setup.beta_delta = 10^(-4);             %Rate parameter for delta prior
setup.alpha_lambda = 1;                 %Shape parameter for lambda prior
setup.beta_lambda = 10^(-4);            %Rate parameter for lambda prior

%Gaussian
setup.sx_sigma_prior = 0.2;             %std for Gaussian Prior for sx
setup.sx_mean_prior = setup.sx0;        %mean for Gaussian Prior for sx

setup.sy_sigma_prior = 1;               %std for Gaussian Prior for sy
setup.sy_mean_prior = setup.sy0;        %mean for Gaussian Prior for sy

setup.dx_sigma_prior = 0.2;             %std for Gaussian Prior for dx
setup.dx_mean_prior = setup.dx0;        %mean for Gaussian Prior for dx

setup.dy_sigma_prior = 1;               %std for Gaussian Prior for dy
setup.dy_mean_prior = setup.dy0;        %mean for Gasussian Prior for dy

%Metropolis-Hastings parameters
setup.N_metro = 20;                     %Number of metropolis hastings samples each Gibbs iteration

%Gaussian proposal
setup.sx_sigma_proposal = 0.0001;       %Step size for MH for sx
setup.sy_sigma_proposal = 0.001;        %Step size for MH for sy
setup.dx_sigma_proposal = 0.0001;       %Step size for MH for dx
setup.dy_sigma_proposal = 0.001;        %Step size for MH for dy

%Initial x-sample parameters
setup.alpha = 0.005;                    %Regularization parameter for initial x-sample
setup.maxiters = 1000;                  %Number of iterations for initial x-sample

%Data simulation parameters
setup.noise_level = 0.02;               %Relative measurement Noise level
setup.inverse_factor = 20.74324;        %Factor determining fineness of grid for data simulation

setup.nonneg = 1;                       %Boolean Specifying if nonnegativity
setup.iid = 1;                          %Boolean Specifying if identity matrix for Gaussian Prior

res = Model_Gibbs_Sampler_Wrap(setup);

foldername = 'dx_sx_sy_128_lowprop_2_virtual';
folder_dir = fullfile('/zhome','94','f','108663','Desktop','Masters','Data','UQ','Fanbeam',foldername);
mkdir(folder_dir);

f = fullfile(folder_dir,'tikh_nonneg.mat');
save(f,'res')
