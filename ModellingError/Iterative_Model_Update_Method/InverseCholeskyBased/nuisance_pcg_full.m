function val = nuisance_pcg_full(x,proj_id,R,L,alpha,angles,p)
%This function computes the quantity
%   (R'A'AR + alpha L)x
%for the use of PCG in the Gaussian Setting. (Due to some wierd issues it
%does not seem to be possible to pass a SPOT operator as auxillary variable
%into PCG) therefore we resort to instead pass a projector id to do the
%forward projection

%INPUT:
%x: Reconstruction
%proj_id: Astra object projector id
%R: Inverse cholesky of covariance matrix
%L: Precision matrix of Gaussian Prior
%alpha: regularization parameter
%angles: Projection angles
%p: Number of rays

%OUTPUT:
%val: (R'A'AR + alpha L)x

%Get the grid size and number of projections
N = sqrt(length(x));
n_proj = length(angles);

%Compute forward projection
[~,forward_proj] = astra_create_sino(reshape(x,N,N),proj_id);

%Reshape into vector structure
Ax = reshape(forward_proj,n_proj*p,1);

%Multiply by inverse covariance and turn into matrix structure
tmp = reshape(R'*(R*Ax),n_proj,p);

%Backproject using astra
[~,back_proj] = astra_create_backprojection(tmp,proj_id);

%Turn into vector form
tmp = reshape(back_proj,N^2,1);

%Add regularization term
val = tmp + alpha*L*x;
end
