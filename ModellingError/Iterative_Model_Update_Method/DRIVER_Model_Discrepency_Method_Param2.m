%Driver script for additive modelling errors.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%ADD NECESSARY PATHS TO DIRECTORY
f = fullfile('/zhome','94','f','108663','Desktop','Masters','MatlabCode');
addpath(genpath(f));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%SET PARAMETERS FOR MODEL ERROR ALGORITHM

%GEOMETRY details
setup.N = 512;                          %GridSize (N by N)
setup.theta = 0:2:358;                  %Projection angles in degrees
setup.p = 1.5*setup.N;                  %Number of rays
setup.ray_config = 'fanbeam';           %Ray configuration (parallel or fanbeam)

%Model Parameters
setup.DETECTOR_X = 1;                   %Specify if COR should be estimated
setup.DETECTOR_Y = 1;                   %Specify if detector distance should be estimated
setup.SOURCE_X = 1;                     %Specify if source distance should be estimated
setup.SOURCE_Y = 1;                     %Specify if Detector Tilt shoul be estimated

setup.dx_true = 0.2;                       %The true COR parameter (physical)
setup.dy_true = 10;                       %The true Detector Distance (physical)
setup.sx_true = 0.1;                       %The true Source distance (physical)
setup.sy_true = 4;                       %The true Detector Tilt (degrees)

setup.detector_width = (setup.dy_true + setup.sy_true)/(setup.sy_true)*3;

setup.dx0 = 0;                           %Initial guess for COR parameter (physical)
setup.dy0 = 9;                           %Initial guess for Detector distance (physical)
setup.sx0 = 0;                           %Initial guess for Source distance
setup.sy0 = 5;                           %Initial guess for Detector Tilt

setup.sigma_dx_0 = 0.005;              %Initial guess for std. of COR parameter
setup.sigma_dy_0 = 0.1;                   %Initial guess for std. of DD parameter
setup.sigma_sx_0 = 0.005;                  %Initial guess for std. of SD parameter
setup.sigma_sy_0 = 0.1;                  %Initial guess for std. of Tilt parameter 

setup.cross_cor = 0;                    %Specify if preproccess COR using cross-correlation (parallel only)

%Data simulation details
setup.inverse_factor = 2.321;           %Factor that determines fineness of grid for data generation
setup.sino = 'disc';                    %Specify if ana. or disc. sinogram (ana or disc) (only parallel)
setup.noise_level = 0.02;               %Set measurement error noise level          

%Reconstruction method
setup.reg_term = 'TV';                  %Specify regularization term (tikh, gentikh or TV)
setup.nonneg = 1;                       %Specify if nonnegativity constraints (1 if yes, 0 if no)
setup.alpha_ini = 0.000002;             %Regularization parameter for initial reconstruction
setup.alpha = 0.01;                     %Regularization parameter for inner problem
setup.maxiters = 300;                   %Maximum number of iterations in iterative algorithms within big loop
setup.maxiters_ini = 1000;              %Maximum number of iteration for initial reconstruction
setup.gpu = 0;                          %Specify if we should use GPU (1 if yes, 0 if no)

%Algorithm parameters
setup.N_out = 50;                       %Number of outer iterations
setup.S = 500;                          %Number of samples for modelerror reconstruction step
setup.S_update = 500;                   %Number of samples for parameter update step
setup.gamma = 0;                        %Variance relaxation parameter
setup.fig_show = 0;                     %Specify if we want to show figures (1 if yes, 0 if no)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
foldername = 'dx_dy_sx_sy_512_regTest';
folder_dir = fullfile('/zhome','94','f','108663','Desktop','Masters','Data','Model_Discrepency','Fanbeam',foldername);
mkdir(folder_dir);

res = Additive_Modelling_Error_Wrap_Param2(setup);
f = fullfile(folder_dir,'TV_nonneg_001.mat');
save(f,'res')
