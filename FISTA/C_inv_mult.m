function res = C_inv_mult(x,u,uTu,lambda)
%This function computes the matrix vector multiplication
%           (lambda^(-1) I + uu^T)^(-1) x
%by utilizing the Woodbury formula.

%INPUT:
%x: Input vector of length M
%u: Matrix error sample matrix (M x N_s)
%uTu: Matrix-Matrix product u^T u (N_s x N_s)
%lambda: Noise precision parameter

%Output:
%res: Resulting matrix vector multiplication

[~,Ns] = size(u);
uTu = u'*u;

uT_x = u'*x;

tmp = (eye(Ns)+lambda*uTu)\uT_x;

res = lambda*x - lambda^2*u*tmp;
end