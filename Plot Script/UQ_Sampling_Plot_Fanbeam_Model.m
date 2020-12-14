function UQ_Sampling_Plot_Fanbeam_Model(res,N_burn)

close all

%Unpack variables
x_samps = res.x_samps;
delta_samps = res.delta_samps;
lambda_samps = res.lambda_samps;

model_param_count = res.setup.SOURCE_X + res.setup.SOURCE_Y + res.setup.DETECTOR_X + res.setup.DETECTOR_Y;
figure(1)
figure(2)

figure(1)
subplot(model_param_count+2,3,1)
plot(1:length(lambda_samps), lambda_samps)
title('\lambda - chain')
subplot(model_param_count+2,3,2)
histogram(lambda_samps)
title('\lambda - histogram')
subplot(model_param_count+2,3,3)
autocorr(lambda_samps)
title('\lambda - ACF')
subplot(model_param_count+2,3,4)
plot(1:length(delta_samps), delta_samps)
title('\delta - chain')
subplot(model_param_count+2,3,5)
histogram(delta_samps)
title('\delta - histogram')
subplot(model_param_count+2,3,6)
autocorr(delta_samps)
title('\delta - ACF')

figure(2)
subplot(model_param_count+2,3,1)
plot(1:(length(lambda_samps)-N_burn),lambda_samps(N_burn+1:end))
title('\lambda - chain')
subplot(model_param_count+2,3,2)
histogram(lambda_samps(N_burn+1:end))
title('\lambda - histogram')
subplot(model_param_count+2,3,3)
autocorr(lambda_samps(N_burn+1:end))
title('\lambda - ACF')
ylabel('')
subplot(model_param_count+2,3,4)
plot(1:(length(delta_samps)-N_burn), delta_samps(N_burn+1:end))
title('\delta - chain')
subplot(model_param_count+2,3,5)
histogram(delta_samps(N_burn+1:end))
title('\delta - histogram')
subplot(model_param_count+2,3,6)
autocorr(delta_samps(N_burn+1:end))
ylabel('')
title('\delta - ACF')

mod_count = 0;

if res.setup.SOURCE_X == 1
    sx_true = res.setup.sx_true;
    sx_samps = res.sx_samps;
    mod_count = mod_count + 1;
    figure(1)
    subplot(model_param_count+2,3,6+mod_count*3-2)
    plot(1:length(sx_samps),sx_samps)
    title('sx-chain')
    subplot(model_param_count+2,3,6+mod_count*3-1)
    histogram(sx_samps)
    title('sx-histogram')
    subplot(model_param_count+2,3,6+mod_count*3)
    autocorr(sx_samps)
    ylabel('')
    title('sx-ACF')
        
    figure(2)
    subplot(model_param_count+2,3,6+mod_count*3-2)
    plot(1:(length(sx_samps)-N_burn),sx_samps(N_burn+1:end))
    title('sx-chain')
    subplot(model_param_count+2,3,6+mod_count*3-1)
    histogram(sx_samps(N_burn+1:end))
    title('sx-histogram')
    subplot(model_param_count+2,3,6+mod_count*3)
    autocorr(sx_samps(N_burn+1:end))
    ylabel('')
    title('sx-ACF')
end

if res.setup.SOURCE_Y == 1
    sy_true = res.setup.sy_true;
    sy_samps = res.sy_samps;
    mod_count = mod_count + 1;
    
    figure(1)
    subplot(model_param_count+2,3,6+mod_count*3-2)
    plot(1:length(sy_samps),sy_samps)
    title('sy-chain')
    subplot(model_param_count+2,3,6+mod_count*3-1)
    histogram(sy_samps)
    title('sy-histogram')
    subplot(model_param_count+2,3,6+mod_count*3)
    autocorr(sy_samps)
    ylabel('')
    title('sy-ACF')
        
    figure(2)
    subplot(model_param_count+2,3,6+mod_count*3-2)
    plot(1:(length(sy_samps)-N_burn),sy_samps(N_burn+1:end))
    title('sy-chain')
    subplot(model_param_count+2,3,6+mod_count*3-1)
    histogram(sy_samps(N_burn+1:end))
    title('sy-histogram')
    subplot(model_param_count+2,3,6+mod_count*3)
    autocorr(sy_samps(N_burn+1:end))
    ylabel('')
    title('sy-ACF')
end

if res.setup.DETECTOR_X == 1
    dx_true = res.setup.dx_true;
    dx_samps = res.dx_samps;
    mod_count = mod_count + 1;
    
    figure(1)
    subplot(model_param_count+2,3,6+mod_count*3-2)
    plot(1:length(dx_samps),dx_samps)
    title('dx-chain')
    subplot(model_param_count+2,3,6+mod_count*3-1)
    histogram(dx_samps)
    title('dx-histogram')
    subplot(model_param_count+2,3,6+mod_count*3)
    autocorr(dx_samps)
    ylabel('')
    title('dx-ACF')
        
    figure(2)
    subplot(model_param_count+2,3,6+mod_count*3-2)
    plot(1:(length(dx_samps)-N_burn),dx_samps(N_burn+1:end))
    title('dx-chain')
    subplot(model_param_count+2,3,6+mod_count*3-1)
    histogram(dx_samps(N_burn+1:end))
    title('dx-histogram')
    subplot(model_param_count+2,3,6+mod_count*3)
    autocorr(dx_samps(N_burn+1:end))
    ylabel('')
    title('dx-ACF')
end

if res.setup.DETECTOR_Y == 1
    dy_true = res.setup.dy_true;
    dy_samps = res.dy_samps;
    mod_count = mod_count + 1;
    
    figure(1)
    subplot(model_param_count+2,3,6+mod_count*3-2)
    plot(1:length(dy_samps),dy_samps)
    title('dy-chain')
    subplot(model_param_count+2,3,6+mod_count*3-1)
    histogram(dy_samps)
    title('dy-histogram')
    subplot(model_param_count+2,3,6+mod_count*3)
    autocorr(dy_samps)
    ylabel('')
    title('dy-ACF')
        
    figure(2)
    subplot(model_param_count+2,3,6+mod_count*3-2)
    plot(1:(length(dy_samps)-N_burn),dy_samps(N_burn+1:end))
    title('dy-chain')
    subplot(model_param_count+2,3,6+mod_count*3-1)
    histogram(dy_samps(N_burn+1:end))
    title('dy-histogram')
    subplot(model_param_count+2,3,6+mod_count*3)
    autocorr(dy_samps(N_burn+1:end))
    ylabel('')
    title('dy-ACF')
end

N = res.setup.N;

if res.setup.nonneg == 1
    con = 'non-negative';
else
    con = '';
end

if res.setup.iid == 1
    reg_term = 'Tikh.';
else
    reg_term = 'Gentikh.';
end

%Plot sample mean reconstruction and width of 95% pixelwise credibility
%Remove Burn in samples
x_samps(:,1:N_burn) = [];

x_mean = mean(x_samps,2);
lower_quant = quantile(x_samps,0.025,2);
upper_quant = quantile(x_samps,0.975,2);

figure
subplot(1,2,1)
imagesc(reshape(x_mean,N,N)), axis image, colorbar
title('Sample Mean')
subplot(1,2,2)
imagesc(reshape(upper_quant-lower_quant,N,N)), axis image, colorbar
title('Width of 95% Credibility Interval')

%Plot images of upper and lower quantiles
figure
subplot(1,2,1)
imagesc(reshape(lower_quant,N,N),[min(min(lower_quant)), max(max(upper_quant))]), axis image, colorbar
title('Lower Quantile')
subplot(1,2,2)
imagesc(reshape(upper_quant,N,N),[min(min(lower_quant)), max(max(upper_quant))]), axis image, colorbar
title('Upper Quantile')

figure
imagesc(reshape(res.setup.b,length(res.setup.theta),res.setup.p)), axis image, colorbar
title('Noisy Sinogram')

%Do geweke test
[z_lambda,p_lambda] = geweke(lambda_samps(N_burn:end)');
[z_delta,p_delta] = geweke(delta_samps(N_burn:end)');
disp(['p-value for lambda chain: ' num2str(p_lambda)])
disp(['p-value for delta chain: ' num2str(p_delta)])

if res.setup.SOURCE_X == 1
    [z_sx,p_sx] = geweke(sx_samps(N_burn:end)');
    disp(['p-value for sx-chain: ' num2str(p_sx)])
end

if res.setup.SOURCE_Y == 1
    [z_sy,p_sy] = geweke(sy_samps(N_burn:end)');
    disp(['p-value for sy-chain: ' num2str(p_sy)])
end

if res.setup.DETECTOR_X == 1
    [z_dx,p_dx] = geweke(dx_samps(N_burn:end)');
    disp(['p-value for dx-chain: ' num2str(p_dx)])
end

if res.setup.DETECTOR_Y == 1
    [z_dy,p_dy] = geweke(dy_samps(N_burn:end)');
    disp(['p-value for dy-chain: ' num2str(p_dy)])
end
end
