function [A,b] = Generate_Model_Problem(ray_config,N,x,theta,p,gpu,sino,c,d,s,t)

%This function sets up the system matrix and data for the modelling error
%problem

%INPUT:
%ray_config: string specifying projection geometry (parallel or fanbeam)
%N: Gridsize (N x N)
%x_true: True image
%theta: projection angles in degrees
%noise_level: noise level of measurement errors
%p: Number of rays
%gpu: specify if we should use gpu or cpu
%sino: string specifying if analytical or discretized sinogram
%c: Center of rotation parameter
%da: Detector-Origin distance parameter
%s: Source distance parameter
%t: Detector Tilt parameter

%Output:
%A: forward operator
%b: sinogram

vol_geom = astra_create_vol_geom(N,N,-1,1,-1,1);
angles = theta*pi/180;
n_detector = p-1;

A = Build_Forward_Operator(ray_config,vol_geom,gpu,angles,p,c,d,s,t);

if strcmp(sino,'ana')
    [~,b] = paralleltomo_mod(N,theta,p,[n_detector,c,0]);
    b = reshape(b,p,length(theta));
    b = reshape(b',p*length(theta),1);
elseif strcmp(sino,'disc')
    b = A*x(:);
end
end

function A = Build_Forward_Operator(ray_config,vol_geom,gpu,angles,p,c,d,s,t)
%Function that builds the forward operator A

%INPUT:
%ray_config: string that tells if parallel or fanbeam
%volume_geometry: astra volume geometry object
%angles: projection angles in radians
%p: number of rays
%c: COR parameter
%d: DD parameter
%s: SD parameter
%t: Detector Tilt parameter

%OUTPUT:
%A: forward operator

vectors = zeros(length(angles),6);

if strcmp(ray_config,'parallel')
    vectors(:,1) = -sin(angles);
    vectors(:,2) = cos(angles);
    vectors(:,3) = c*cos(angles);
    vectors(:,4) = c*sin(angles);
    vectors(:,5) = cos(angles + t/180*pi)*3/p;
    vectors(:,6) = sin(angles + t/180*pi)*3/p;
    
    proj_geom = astra_create_proj_geom('parallel_vec',p,vectors);
elseif strcmp(ray_config,'fanbeam')
    vectors(:,1) = sin(angles)*s;
    vectors(:,2) = -cos(angles)*s;
    vectors(:,3) = -sin(angles)*d + c*cos(angles);
    vectors(:,4) = cos(angles)*d + c*sin(angles);
    vectors(:,5) = cos(angles + t/180*pi)*3/p;
    vectors(:,6) = sin(angles + t/180*pi)*3/p;
    
    proj_geom = astra_create_proj_geom('fanflat_vec',p,vectors);
end
        
if gpu == 1
    A = opTomo('cuda',proj_geom,vol_geom);
else
    if strcmp(ray_config,'parallel')
        A = opTomo('linear',proj_geom,vol_geom);
    elseif strcmp(ray_config,'fanbeam')
        A = opTomo('line_fanflat',proj_geom,vol_geom);
    end
end
end