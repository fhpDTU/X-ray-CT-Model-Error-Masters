%Test the implemented FISTA method
clear
clc
close all
%%
rng(100);

N = 100; %Grid size
theta = 0:179; %angles
theta = theta*pi/180; %Angles in radians
p = 2*N; %Number of detectors
d = p-1;
dxoffset = 0;
da = 0;
dvec = [d,dxoffset,da];
d_astra = 1; %Distance between detectors

M = length(theta)*p; %Number of rows in system matrix

%Create Phantom
phantomstring = 'SheppLogan';
x_true = phantom(N);

%Specify volume geometry
vol_geom = astra_create_vol_geom(N,N);

%Specify projection geometry
proj_geom = astra_create_proj_geom('parallel',d_astra,p,theta);

%Generate SPOT operator
A = opTomo('line',proj_geom,vol_geom);

%Create analytical sinogram
[~,b_true] = paralleltomo_mod(N,0:179,p,dvec);

%Modify the analytical sinogram slightly to get it into correct form for
%astra

b_true = reshape(b_true,p,length(theta))';
b_true = reshape(b_true,p*length(theta),1);

%Add gaussian additive noise to analytical sinogram
rho = 0.05; %noise level
e = randn(size(b_true)); %standard normal distributed noise

%Normalize the noise and add to sinogram
e = rho*norm(b_true)*e/(norm(e));
b_noise = b_true + e;

%Prior matrix
D = speye(N^2);
delta = 1;
lambda = 5;
u = zeros(N^2,1);
x0 = zeros(N^2,1);
%%
options_kat.x0 = zeros(N^2,1);
options_kat.lambda = lambda;
options_kat.delta = delta;
options_kat.maxiters = 100;
options_kat.nonneg = 1;

options.maxiters = 100;
options.nonneg = 1;
options.epsilon = 10^(-6);
options.x0 = zeros(N^2,1);
tic
[x_fista,fval] = fista_Gen_tikh(A,D,b_noise,u,lambda,delta,options);
toc
x_pcg = pcg(lambda*(A'*A)+delta*speye(N^2),lambda*A'*b_noise,10^(-6),50);
x_naive = (lambda*(A'*A)+delta*speye(N^2))\(lambda*A'*b_noise);
tic
[x_fista_kb,L_fista_kb] = gd_fista_weights(A,speye(N^2),lambda*A'*b_noise,ones(M,1),options_kat);
toc

%%
figure
subplot(2,2,1)
imagesc(reshape(x_fista,N,N)), colorbar, axis image
title('my Fista')
subplot(2,2,2)
imagesc(reshape(x_pcg,N,N)), colorbar, axis image
title('pcg')
subplot(2,2,3)
imagesc(reshape(x_naive,N,N)), colorbar, axis image
title('Direct Solve')
subplot(2,2,4)
imagesc(reshape(x_fista_kb,N,N)), colorbar , axis image
title('KB fista')

figure
semilogy(1:length(fval),fval)
title('Fista objective values')
xlabel('Iteration')
ylabel('Objective Value')