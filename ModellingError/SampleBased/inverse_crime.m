%Test a setup, where we can try to avoid the inverse crime by simulating
%data from a higher dimensional grid

%Use fanbeam geometry
N = 128;
x = phantom(N);
inverse_factor = 10.75;
N_fine = round(inverse_factor*N);
x_fine = phantom(N_fine);
p = 128*1.5;
theta = 0:2:358;
angles = theta/180*pi;
s = 150;
c = -10;


vectors = zeros(length(theta),6);
vectors(:,1) = sin(angles)*s;
vectors(:,2) = -cos(angles)*s;
vectors(:,3) = c*cos(angles);
vectors(:,4) = c*sin(angles);
vectors(:,5) = cos(angles);
vectors(:,6) = sin(angles);

vectors_fine = zeros(length(theta),6);
vectors_fine(:,1) = sin(angles)*s*N_fine/N;
vectors_fine(:,2) = -cos(angles)*s*N_fine/N;
vectors_fine(:,3) = N_fine/N*c*cos(angles);
vectors_fine(:,4) = N_fine/N*c*sin(angles);
vectors_fine(:,5) = N_fine/N*cos(angles);
vectors_fine(:,6) = N_fine/N*sin(angles);

vol_geom = astra_create_vol_geom(N,N);
vol_geom_fine = astra_create_vol_geom(N_fine,N_fine);
proj_geom = astra_create_proj_geom('fanflat_vec',p,vectors);
proj_geom_fine = astra_create_proj_geom('fanflat_vec',p,vectors_fine);

A = opTomo('line_fanflat',proj_geom,vol_geom);
A_fine = opTomo('line_fanflat',proj_geom_fine,vol_geom_fine);

%Create data
b = A*x(:);
b_fine = A_fine*x_fine(:)/(N_fine^2/N^2);

figure
subplot(1,2,1)
imagesc(reshape(b,length(theta),p)), axis image, colorbar
title('Coarse Grid - Fanbeam')
subplot(1,2,2)
imagesc(reshape(b_fine,length(theta),p)), axis image, colorbar
title('Fine Grid - Fanbeam')

%Compute reconstructions
x_recon = A\b;
x_fine_recon = A\b_fine;

figure
subplot(1,2,1)
imagesc(reshape(x_recon,N,N)), axis image, colorbar
title('Coarse Data - Fanbeam')
subplot(1,2,2)
imagesc(reshape(x_fine_recon,N,N)), axis image, colorbar
title('Fine Data - Fanbeam')
%%
%Test a setup, where we can try to avoid the inverse crime by simulating
%data from a higher dimensional grid

%Use fanbeam geometry
N = 128;
x = phantom(N);
inverse_factor = 20.75;
N_fine = round(N*inverse_factor);
x_fine = phantom(N_fine);
p = 128*1.5;
theta = 0:179;
angles = theta/180*pi;
s = 150;
c = 2.75;


vectors = zeros(length(theta),6);
vectors(:,1) = -sin(angles);
vectors(:,2) = cos(angles);
vectors(:,3) = c*cos(angles);
vectors(:,4) = c*sin(angles);
vectors(:,5) = cos(angles);
vectors(:,6) = sin(angles);

vectors_fine = zeros(length(theta),6);
vectors_fine(:,1) = -sin(angles);
vectors_fine(:,2) = cos(angles);
vectors_fine(:,3) = N_fine/N*c*cos(angles);
vectors_fine(:,4) = N_fine/N*c*sin(angles);
vectors_fine(:,5) = N_fine/N*cos(angles);
vectors_fine(:,6) = N_fine/N*sin(angles);

vol_geom = astra_create_vol_geom(N,N);
vol_geom_fine = astra_create_vol_geom(N_fine,N_fine);
proj_geom = astra_create_proj_geom('parallel_vec',p,vectors);
proj_geom_fine = astra_create_proj_geom('parallel_vec',p,vectors_fine);

A = opTomo('line',proj_geom,vol_geom);
A_fine = opTomo('line',proj_geom_fine,vol_geom_fine);

%Create data
b = A*x(:);
norm(b)
b_fine = A_fine*x_fine(:);
norm(b_fine,2)
b_fine = b_fine/(N_fine^2/N^2);
norm(b_fine,2)
[~,b_ana] = paralleltomo_mod(N,theta,p,[p-1,c,0]);

norm(b_ana,2)
norm(b_fine,2)
norm(b)

figure
subplot(1,3,1)
imagesc(reshape(b,length(theta),p)), axis image, colorbar
title('Coarse Grid - Parallel')
subplot(1,3,2)
imagesc(reshape(b_fine,length(theta),p)), axis image, colorbar
title('Fine Grid - Parallel')
subplot(1,3,3)
imagesc(reshape(b_ana,p,length(theta))'), axis image, colorbar
title('Analytical - Parallel')

b_ana = reshape(b_ana,p,length(theta));
b_ana = reshape(b_ana',p*length(theta),1);

%Compute reconstructions
x_recon = A\b;
x_fine_recon = A\b_fine;
x_ana = A\b_ana;

figure
subplot(1,3,1)
imagesc(reshape(x_recon,N,N)), axis image, colorbar
title('Coarse Data - Parallel')
subplot(1,3,2)
imagesc(reshape(x_fine_recon,N,N)), axis image, colorbar
title('Fine Data - Parallel')
subplot(1,3,3)
imagesc(reshape(x_ana,N,N)), axis image, colorbar
title('Analytical - Parallel')