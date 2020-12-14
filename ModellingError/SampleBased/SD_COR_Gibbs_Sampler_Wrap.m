function res = SD_COR_Gibbs_Sampler_Wrap(setup)
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

s_true = setup.s_true;                  %True Source Distance (SD)
c_true = setup.c_true;                  %True Center of Rotation (COR)
t_true = setup.t_true;                  %True Detector Tilt

s0 = setup.s0;                          %Initial guess for SD parameter
c0 = setup.c0;                          %Initial guess for COR parameter
t0 = setup.t0;                          %Initial guess for TILT parameter

noise_level = setup.noise_level;        %Measurement noise level
inverse_factor = setup.inverse_factor;  %Fineness of data simulation grid

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

%Source position
vectors_true(:,1) = sin(angles)*s_true;
vectors_true(:,2) = -cos(angles)*s_true;

%Detector center
vectors_true(:,3) = c_true*cos(angles);
vectors_true(:,4) = c_true*sin(angles);

%Detector basis vector (Detector has "physical" length 3, and we have p detector
%pixels)
vectors_true(:,5) = cos(angles + t_true/180*pi)*3/p;
vectors_true(:,6) = sin(angles + t_true/180*pi)*3/p;

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
vectors(:,1) = sin(angles)*s0;
vectors(:,2) = -cos(angles)*s0;
vectors(:,3) = c0*cos(angles);
vectors(:,4) = c0*sin(angles);
vectors(:,5) = cos(angles + t0/180*pi)*3/p;
vectors(:,6) = sin(angles + t0/180*pi)*3/p;

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
res = SD_COR_Gibbs_Sampler(setup);
end