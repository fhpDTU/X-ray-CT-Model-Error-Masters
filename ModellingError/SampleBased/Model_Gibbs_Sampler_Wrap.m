function res = Model_Gibbs_Sampler_Wrap(setup)
%Wrapper function for Gibbs sampler with SD and COR model errors. Here we
%create the noisy sinogram and the initial reconstruction

%INPUT:
%setup: Struct containing all information about sampler and geometry. See
%driver script or sampler for more details on all the fields

%Unpack variables neccessary for simulation of data and computing initial
%x-sample

N = setup.N;                            %Grid size                                
p = setup.p;                            %Number of detectors
theta = setup.theta;                    %Projection angles (degrees)

%true model paramters
sx_true = setup.sx_true;                
sy_true = setup.sy_true;
dx_true = setup.dx_true;
dy_true = setup.dy_true;                

%initial model parameters
sx0 = setup.sx0;                    
sy0 = setup.sy0;
dx0 = setup.dx0;
dy0 = setup.dy0;

noise_level = setup.noise_level;        %Measurement noise level
inverse_factor = setup.inverse_factor;  %Fineness of data simulation grid
detector_width = setup.detector_width;

iid = setup.iid;                        %Boolean determining if identity precison prior
nonneg = setup.nonneg;                  %Boolean determining if nonnegativity constraints

%Generate Data from a higher dimensional grid to avoid inverse crime
N_fine = round(N*inverse_factor);
x_fine = phantom(N_fine);

%Volume geometry (physical domain on [-1,1] x [-1,1])
vol_geom_fine = astra_create_vol_geom(N_fine,N_fine,-1,1,-1,1);

angles = pi/180*theta;
%Geometry for data generation using true model parameters
vectors_true = zeros(length(angles),6);

vectors_true(:,1) = cos(angles)*sx_true+sin(angles)*sy_true; 
vectors_true(:,2) = sin(angles)*sx_true-cos(angles)*sy_true;  
vectors_true(:,3) = cos(angles)*dx_true-sin(angles)*dy_true;  
vectors_true(:,4) = sin(angles)*dx_true+cos(angles)*dy_true;  
vectors_true(:,5) = cos(angles)*detector_width/p; 
vectors_true(:,6) = sin(angles)*detector_width/p;

%Generate projection geometry and simulate data
proj_geom_true = astra_create_proj_geom('fanflat_vec',p,vectors_true);
A_true = opTomo('line_fanflat',proj_geom_true,vol_geom_fine);
b = A_true*x_fine(:);

%Add Gaussian i.i.d Noise to data to simulate measurement noise
e = randn(size(b));
e = noise_level*norm(b)*e/(norm(e));
b_noise = b + e;

setup.b = b_noise;

%Setup projection geometry for initial guess of forward operator
vectors = zeros(length(theta),6);
vectors(:,1) = cos(angles)*sx0+sin(angles)*sy0; 
vectors(:,2) = sin(angles)*sx0-cos(angles)*sy0;  
vectors(:,3) = cos(angles)*dx0-sin(angles)*dy0;  
vectors(:,4) = sin(angles)*dx0+cos(angles)*dy0;  
vectors(:,5) = cos(angles)*detector_width/p; 
vectors(:,6) = sin(angles)*detector_width/p;

%Compute initial projection and volume geometry geometry
vol_geom = astra_create_vol_geom(N,N,-1,1,-1,1);
proj_geom = astra_create_proj_geom('fanflat_vec',p,vectors);
A = opTomo('line_fanflat',proj_geom,vol_geom);

%Compute initial reconstruction using FISTA
%Gaussian Prior precision matrix
if iid == 1
    D = speye(N^2);
else
    e_vec = ones(N,1);
    L_1D = spdiags([-e_vec 2*e_vec -e_vec], [-1 0 1],N,N);
    L = kron(speye(N),L_1D) + kron(L_1D,speye(N));
    D = chol(L);
end

%Fista Algorithm parameters
options.maxiters = setup.maxiters;
options.x0 = zeros(N^2,1);
options.h = 1;
options.epsilon = 10^(-8);
options.iid = iid;
options.nonneg = nonneg;

alpha = setup.alpha; %Regularization parameter for initial reconstruction

x_MAP = fista_Gen_tikh(A,D,b_noise,zeros(N^2,1),1,alpha,options);
setup.x0 = x_MAP;

%Do Hierarchial Gibbs sampling
res = Model_Gibbs_Sampler(setup);
end