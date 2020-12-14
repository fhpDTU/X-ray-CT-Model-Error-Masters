function res = SD_Gibbs_Sampler_Wrap(setup)
%Wrapper function for Gibbs sampler with SD model errors

N = setup.N;
p = setup.p;
s_true = setup.s_true;
noise_level = setup.noise_level;
theta = setup.theta;
iid = setup.iid;
nonneg = setup.nonneg;
inverse_factor = setup.inverse_factor;
s0 = setup.s0;

%Generate Data from a higher dimensional grid to avoid the inverse crime
N_fine = round(N*inverse_factor);
x_fine = phantom(N_fine);

angles = pi/180*theta;
%Geometry for initial forward operator
vectors = zeros(length(angles),6);
vectors(:,1) = sin(angles)*s0;
vectors(:,2) = -cos(angles)*s0;
vectors(:,5) = cos(angles);
vectors(:,6) = sin(angles);

%Geometry for data generation
vectors_fine = zeros(length(angles),6);
vectors_fine(:,1) = sin(angles)*s_true*inverse_factor;
vectors_fine(:,2) = -cos(angles)*s_true*inverse_factor;
vectors_fine(:,5) = cos(angles)*inverse_factor;
vectors_fine(:,6) = sin(angles)*inverse_factor;

vol_geom_fine = astra_create_vol_geom(N_fine,N_fine);
proj_geom_fine = astra_create_proj_geom('fanflat_vec',p,vectors_fine);
A_fine = opTomo('line_fanflat',proj_geom_fine,vol_geom_fine);
b = A_fine*x_fine(:);
b = b*(N^2/N_fine^2);

%Add Gaussian Noise to data
e = randn(size(b));
e = noise_level*norm(b)*e/(norm(e));
b_noise = b + e;

setup.b = b_noise;

%Compute initial projection geometry
vol_geom = astra_create_vol_geom(N,N);
proj_geom = astra_create_proj_geom('fanflat_vec',p,vectors);
A = opTomo('line_fanflat',proj_geom,vol_geom);

%Compute initial reconstruction
if iid == 1
    reg_term = 'tikh';
else
    reg_term = 'gentikh';
end
x0 = zeros(N^2,1);
alpha = setup.alpha;
maxiters = setup.maxiters;

x_MAP = MAP_recon(A,b_noise,alpha,x0,reg_term,maxiters,nonneg);
setup.x0 = x_MAP;

%Do Hierarchial Gibbs sampling
res = COR_Gibbs_Sampler(setup);
end