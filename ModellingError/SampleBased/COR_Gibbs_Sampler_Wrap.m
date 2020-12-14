function res = COR_Gibbs_Sampler_Wrap(setup)
%Wrapper function for Gibbs sampler in COR

N = setup.N;
p = setup.p;
c_true = setup.c_true;
noise_level = setup.noise_level;
theta = setup.theta;
crosscorr = setup.crosscorr;
iid = setup.iid;
nonneg = setup.nonneg;

%Generate data with true COR parameter
if setup.ana == 1
    [~,b] = paralleltomo_mod(N,theta,p,[p-1,c_true,0]);
    norm(b)
else
    inverse_factor = setup.inverse_factor;
    N_fine = double(round(N*inverse_factor));
    x_fine = phantom(N_fine);

    angles = theta/180*pi;
    %Geometry for fine projections
    vectors_fine = zeros(length(angles),6);
    vectors_fine(:,1) = -sin(angles);
    vectors_fine(:,2) = cos(angles);
    vectors_fine(:,3) = N_fine/N*c_true*cos(angles);
    vectors_fine(:,4) = N_fine/N*c_true*sin(angles);
    vectors_fine(:,5) = N_fine/N*cos(angles);
    vectors_fine(:,6) = N_fine/N*sin(angles);
    
    vol_geom_fine = astra_create_vol_geom(N_fine,N_fine);
    proj_geom_fine = astra_create_proj_geom('parallel_vec',p,vectors_fine);
    A_fine = opTomo('line',proj_geom_fine,vol_geom_fine);
    b = A_fine*x_fine(:);
    b = b/inverse_factor;                    %With correct scaling
    %b = b/(N_fine.^2/(N^2)); %With incorrect scaling
end

%Add Gaussian Noise to data
e = randn(size(b));
e = noise_level*norm(b)*e/(norm(e));
b_noise = b + e;

%Obtain initial estimate for COR parameter
if crosscorr == 1
    b_noise = reshape(b_noise,p,length(theta));

    [c,lags] = xcorr(flipud(b_noise(:,end)),b_noise(:,1));
    [~,ind] = max(c);
    opt_lag = lags(ind);
    c0 = opt_lag/2;
    setup.c0 = c0;
    b_noise(:,end) = [];
    theta(end) = [];
    b_noise = reshape(b_noise',p*length(theta),1);
else
    c0 = setup.c0;
    if setup.ana == 1
        b_noise = reshape(b_noise,p,length(theta));
        b_noise = reshape(b_noise',p*length(theta),1);
    end
end
setup.b = b_noise;

angles = pi/180*theta;
vol_geom = astra_create_vol_geom(N,N);
vectors = zeros(length(angles),6);
vectors(:,1) = -sin(angles);
vectors(:,2) = cos(angles);
vectors(:,3) = c0*cos(angles);
vectors(:,4) = c0*sin(angles);
vectors(:,5) = cos(angles);
vectors(:,6) = sin(angles);

proj_geom = astra_create_proj_geom('parallel_vec',p,vectors);
A = opTomo('line',proj_geom,vol_geom);

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