% Compare Tikhonv and Iterative estimators, and plot boxchart of 
% L2 relative erros for multiple simulations
% 

clear; clc; close all; restoredefaultpath;  
addpath('../')
add_mypaths;
saveON = 0;


%%% 0 setttings: R_phi[u](x)  = \sum_r phi(r) * fun_g[u](x,r) dr, x in [0,1], r= [0,R0]
% the first example----Itegral operator
dx       = 0.005;                          % space mesh size in observation. 
N        = 10;                               %  number of data pairs (u_i,f_i)
u_Type   = 'stocFourier';        % types: 'Bspline', 'Fourier'; 'stocFourier';  'stocCosine' %  Fourier   
jump_disc = 0;               % jump discontinuity to increase rank of G 

R0       = 1;                % maximal interaction range [0,R0] for radial kernel
supp_u   = [-R0 1+R0];       % data u support    >>> f(x) with x = [0,1] 
example_type = 'LinearIntOpt';   % {'LinearIntOpt','nonlocal','Aggregation_StrForm'};
kernel_type  = 'sinx_smooth';       % Gaussian, sinkx, FracLap, sinx_smooth

  
% the second example-----Nonlocal operator
% dx       = 0.005;                          % space mesh size in observation. 
% N        = 10;                               %  number of data pairs (u_i,f_i)
% u_Type   = 'stocFourier';        % types: 'Bspline', 'Fourier', 'stocFourier';  'stocCosine' %  Fourier   
% jump_disc = 1;               % jump discontinuity to increase rank of G 
% 
% R0       = 1;                % maximal interaction range [0,R0] for radial kernel
% supp_u   = [-R0 1+R0];       % data u support    >>> f(x) with x = [0,1] 
% example_type = 'nonlocal';   % {'LinearIntOpt','nonlocal','Aggregation_StrForm'};
% kernel_type  = 'sinkx';       % Gaussian, sinkx, FracLap


% the third example-----Aggregation operator
% dx       = 0.005;                          % space mesh size in observation. 
% N        = 10;                             %  number of data pairs (u_i,f_i)
% u_Type   = 'randomDensity';        % types: 'Bspline', 'Fourier', 'stocFourier';  'stocCosine', randomDensity  
% jump_disc = 0;                     % jump discontinuity to increase rank of G 
% 
% R0       = 1;                    % maximal interaction range [0,R0] for radial kernel
% supp_u   = [-R0 1+R0];           % data u support    >>> f(x) with x = [0,1] 
% example_type = 'Aggregation_StrForm';   % {'LinearIntOpt','nonlocal','Aggregation_StrForm'};
% kernel_type  = 'sinx_cubic';       % Gaussian, sinkx, FracLap, sinx_smooth, polyx, powerFn, sinx_cubic



[kernelInfo, obsInfo] = load_settings_v2(N, u_Type,jump_disc, supp_u, R0, dx,kernel_type,example_type);
noise_ratio           = 0.1;  obsInfo.noise_ratio  = noise_ratio;
integrator = 'quadgk'; %  'Riemann', 'quadgk'



%%%---------settings for multiple simulations of Tikhonov estimators------------

numSimulations = 50;
err_num = 6;
err_data = zeros(err_num,numSimulations);

rkhsType = {'auto-RKHS','Gaussian-RKHS','L2-rho'};
method = {'LC','gcv'};
showError = 0;
plotOn = 0; 

%%% Compute Tikhonov regularized solutions, auto basis for kernel methods -------------------
for i = 1:numSimulations
    % Generate data on x_mesh (adaptive to u, f) and pre-process data
    rng(i);
    obsInfo.plotON = 0; % no plots in the parallel simulations. 
    [obsInfo,ux_val,fx_val]   = generateData2(kernelInfo, obsInfo, SAVE_DIR,saveON,integrator);    % get observation data
    if strcmp(example_type, 'Aggregation_StrForm')
        noise_std_upperBd = max(round(obsInfo.noise_std_upperBd,2),0.001);
    else
        noise_std_upperBd = max(round(obsInfo.noise_std_upperBd,2),0.1);
    end
    nsr               = obsInfo.noise_ratio;
    noise_std         = nsr *noise_std_upperBd;  obsInfo.noise_std = noise_std;% noise added to fx_val
    fx_val            = fx_val + noise_std.*randn(size(fx_val));

    dx        = obsInfo.x_mesh_dx; 
    data_str  = [obsInfo.example_type,kernelInfo.kernel_type,obsInfo.u_str,obsInfo.x_mesh_str,sprintf('NSR%1.1f_dx%1.4f_',nsr,dx)];

    % get boundary width and x-mesh in use. --------                            
    bdry_width = obsInfo.bdry_width;  % boundary space for inetration range with kernel
    r_seq      = dx*(1:bdry_width);
    Index_xi_inLoss = (bdry_width+1): (length(obsInfo.u_xmesh) - bdry_width); % index that x_i of u(x_i) in use for L_phi[u]


    %%% 2. pre-process data, get all data elements for regression:   ****** a key step significantly reducing computational cost
 
    normalizeOn    = ~strcmp(obsInfo.example_type, 'classicalReg');
    fun_g_vec      = obsInfo.fun_g_vec;
    regressionData = getData4regression_auto(ux_val,fx_val,dx,obsInfo,bdry_width,Index_xi_inLoss,r_seq,data_str,normalizeOn);
    [ns, J, n0] = size(regressionData.g_ukxj);
    clear ux_val fx_val;

    % select using rho_val = uniform, rho_L1, or rho_L2
    rho_type = 'rho_L2';   % rho_L1, or rho_L2    
    switch  rho_type 
        case 'uniform'
            regressionData.rho_val = regressionData.rho_val0;  
        case 'rho_L1'
            regressionData.rho_val = regressionData.rho_val1; 
        case 'rho_L2'
        regressionData.rho_val = regressionData.rho_val2; 
    end 

    rho_val = regressionData.rho_val; 
    ind_rho = find(rho_val>0);  rho_val = rho_val(ind_rho);
    r_seq   = regressionData.r_seq(ind_rho);    % when r_seq is non-uniform, use dr = r_seq(2:end) - r_seq(1:end-1).       
    dr      = r_seq(2)-r_seq(1);   
    K_true   = kernelInfo.K_true; 
    K_true_val  = K_true(r_seq);
    xx  = K_true_val';
    % nx = norm(xx);    % norm(xx);

    % start solving
    [Tikhonov_sol, Tikhonov_lamb, Tikhonov_err] = comp_Tikh_regu(regressionData, rkhsType, method, xx, showError,plotOn);
    err_data(1,i) = Tikhonov_err.autoRKHS_LC;
    err_data(2,i) = Tikhonov_err.autoRKHS_gcv;
    err_data(3,i) = Tikhonov_err.GaussRKHS_LC;
    err_data(4,i) = Tikhonov_err.GaussRKHS_gcv;
    err_data(5,i) = Tikhonov_err.L2rho_LC;
    err_data(6,i) = Tikhonov_err.L2rho_gcv;
end


%-----------List and plot----------------------
n_methods = size(err_data,1);
n_trials =  size(err_data,2);
colors = lines(n_methods);
mean_err = mean(err_data,2);    % average of every row
std_err  = std(err_data,0,2);   % no-bias std of every row

fprintf('Tikhonov regularization relative L2 errors for %d simulations, mean and standard deviation: \n', n_trials); 
methods_all= ["auto-RKHS-LC"; "auto-RKHS-GCV"; "Gaussian-RKHS-LC"; "Gaussian-RKHS-GCV"; "L2-rho-LC"; "L2-rho-GCV"]; 
Tikh_result = table(methods_all, mean_err, std_err); 
disp(Tikh_result)


figure;
hold on;
h = gobjects(n_methods,1);
for i = 1:n_methods
    x = i*ones(1,n_trials);
    y = err_data(i,:);
    h(i) = boxchart(x, y, ...
        'BoxWidth', 0.5, ...
        'BoxFaceColor', colors(i,:), ...
        'BoxEdgeColor', colors(i,:), ...
        'WhiskerLineColor', colors(i,:), ...
        'JitterOutliers','on', ...
        'MarkerColor', colors(i,:), ...
        'MarkerStyle', '*', ...
        'MarkerSize', 5);
    plot(i, mean_err(i), 'p', ...
         'MarkerSize', 8, ...
         'MarkerFaceColor', colors(i,:), ...
         'MarkerEdgeColor', colors(i,:));
end
set(gca, 'YScale', 'log'); 
labels = {'$H_{\bar{G}}$, LC','$H_{\bar{G}}$, GCV','$H_{k}$, LC','$H_{k}$, GCV','$L^{2}(\rho)$, LC','$L^{2}(\rho)$, GCV'};
fade_ratio = 0.7;  
legendMarkers = gobjects(n_methods, 1);
for i = 1:n_methods
    legend_color = (1 - fade_ratio) * colors(i,:) + fade_ratio * [1 1 1];  
    legendMarkers(i) = plot(nan, nan, 's', ...
        'MarkerSize', 8, ...
        'MarkerFaceColor', legend_color, ...
        'MarkerEdgeColor', colors(i,:), ...
        'LineWidth', 1.2, ...
        'DisplayName', labels{i});
end
legend(legendMarkers, labels, 'Interpreter', 'latex', 'FontSize', 13, 'Location', 'best');
set(gca, 'XTick', 1:n_methods, 'XTickLabel', 1:n_methods,'FontSize', 14);
ylabel('Relative  error','Fontsize',14);
title('Tikh. Est.','FontSize',16,'interpreter','latex');
grid on;
grid minor;
set(gca, 'GridAlpha', 0.1);
set(gca, 'MinorGridAlpha', 0.01);
box on;



%%%---------settings for multiple simulations of Iterative estimators------------

numSimulations = 50;
% err_num = 9;
err_num = 6;
err_data = zeros(err_num,numSimulations);

rkhsType = {'auto-RKHS','Gaussian-RKHS','L2-rho'};
method = {'LC','hyb'};
showError = 0;
K = 40; 

%%% Compute Iterative regularized solutions, auto basis for kernel methods -------------------
for i = 1:numSimulations
    rng(i);
    [kernelInfo, obsInfo] = load_settings_v2(N, u_Type,jump_disc, supp_u, R0, dx,kernel_type,example_type);
    noise_ratio           = 0.1;  obsInfo.noise_ratio  = noise_ratio;  


    %%% 1. Generate data on x_mesh (adaptive to u, f) and pre-process data
    integrator = 'quadgk'; %  'Riemann', 'quadgk'
    obsInfo.plotON = 0; % no plots in the parallel simulations.
    [obsInfo,ux_val,fx_val]   = generateData2(kernelInfo, obsInfo, SAVE_DIR,saveON,integrator);    % get observation data
    if strcmp(example_type, 'Aggregation_StrForm')
        noise_std_upperBd = max(round(obsInfo.noise_std_upperBd,2),0.1);
    else
        noise_std_upperBd = max(round(obsInfo.noise_std_upperBd,2),0.001);
    end
    nsr               = obsInfo.noise_ratio;
    noise_std         = nsr *noise_std_upperBd;  obsInfo.noise_std = noise_std;% noise added to fx_val
    fx_val            = fx_val + noise_std.*randn(size(fx_val));

    dx        = obsInfo.x_mesh_dx; 
    data_str  = [obsInfo.example_type,kernelInfo.kernel_type,obsInfo.u_str,obsInfo.x_mesh_str,sprintf('NSR%1.1f_dx%1.4f_',nsr,dx)];

    % get boundary width and x-mesh in use. --------                            
    bdry_width = obsInfo.bdry_width;  % boundary space for inetration range with kernel
    r_seq      = dx*(1:bdry_width);
    Index_xi_inLoss = (bdry_width+1): (length(obsInfo.u_xmesh) - bdry_width); % index that x_i of u(x_i) in use for L_phi[u]


    %%% 2. pre-process data, get all data elements for regression:   ****** a key step significantly reducing computational cost
    normalizeOn    = ~strcmp(obsInfo.example_type, 'classicalReg');
    fun_g_vec      = obsInfo.fun_g_vec;
    regressionData = getData4regression_auto(ux_val,fx_val,dx,obsInfo,bdry_width,Index_xi_inLoss,r_seq,data_str,normalizeOn);
    [ns, J, n0] = size(regressionData.g_ukxj);
    clear ux_val fx_val;

    % select using rho_val = uniform, rho_L1, or rho_L2
    rho_type = 'rho_L2';   % rho_L1, or rho_L2    
    switch  rho_type 
        case 'uniform'
            regressionData.rho_val = regressionData.rho_val0;  
        case 'rho_L1'
            regressionData.rho_val = regressionData.rho_val1; 
        case 'rho_L2'
        regressionData.rho_val = regressionData.rho_val2; 
    end 

    rho_val = regressionData.rho_val; 
    ind_rho = find(rho_val>0);  rho_val = rho_val(ind_rho);
    r_seq   = regressionData.r_seq(ind_rho);    % when r_seq is non-uniform, use dr = r_seq(2:end) - r_seq(1:end-1).       
    dr      = r_seq(2)-r_seq(1);   
    K_true   = kernelInfo.K_true; 
    K_true_val  = K_true(r_seq);
    xx  = K_true_val';
    % nx = norm(xx);    % norm(xx);

    % start solving
    [Iter_sol, Iter_stop, Iter_lamb, Iter_err, Iter_all_err] = comp_Iter_regu(regressionData, rkhsType, method, K, xx, showError);
    err_data(1,i) = Iter_err.autoRKHS_LC;
    err_data(2,i) = Iter_err.autoRKHS_hyb;
    err_data(3,i) = Iter_err.GaussRKHS_LC;
    err_data(4,i) = Iter_err.GaussRKHS_hyb;
    err_data(5,i) = Iter_err.L2rho_LC;
    err_data(6,i) = Iter_err.L2rho_hyb;

    % err_data(3,i) = Iter_err.autoRKHS_opt;
    % err_data(4,i) = Iter_err.GaussRKHS_LC;
    % err_data(5,i) = Iter_err.GaussRKHS_hyb;
    % err_data(6,i) = Iter_err.GaussRKHS_opt;
    % err_data(7,i) = Iter_err.L2rho_LC;
    % err_data(8,i) = Iter_err.L2rho_hyb;
    % err_data(9,i) = Iter_err.L2rho_opt;

end


%-----------List and plot----------------------
n_methods = size(err_data,1);
n_trials =  size(err_data,2);
colors = lines(n_methods);
mean_err = mean(err_data,2);    % average of every row
std_err  = std(err_data,0,2);   % no-bias std of every row

fprintf('Iterative regularization relative L2 errors for %d simulations, mean and standard deviation: \n', n_trials); 
% methods_all= ["auto-RKHS-LC"; "auto-RKHS-hybrid"; "auto-RKHS-optimal"; "Gaussian-RKHS-LC"; "Gaussian-RKHS-hybrid"; "Gaussian-RKHS-optimal"; "L2-rho-LC"; "L2-rho-hybrid"; "L2-rho-optimal"]; 
methods_all= ["auto-RKHS-LC"; "auto-RKHS-hybrid"; "Gaussian-RKHS-LC"; "Gaussian-RKHS-hybrid"; "L2-rho-LC"; "L2-rho-hybrid"]; 
Iter_result = table(methods_all, mean_err, std_err); 
disp(Iter_result)


figure;
hold on;
h = gobjects(n_methods,1);
for i = 1:n_methods
    x = i*ones(1,n_trials);
    y = err_data(i,:);
    h(i) = boxchart(x, y, ...
        'BoxWidth', 0.5, ...
        'BoxFaceColor', colors(i,:), ...
        'BoxEdgeColor', colors(i,:), ...
        'WhiskerLineColor', colors(i,:), ...
        'JitterOutliers','on', ...
        'MarkerColor', colors(i,:), ...
        'MarkerStyle', '*', ...
        'MarkerSize', 5);
    plot(i, mean_err(i), 'p', ...
         'MarkerSize', 6, ...
         'MarkerFaceColor', colors(i,:), ...
         'MarkerEdgeColor', colors(i,:));
end
set(gca, 'YScale', 'log'); 
labels = {'$H_{\bar{G}}$, LC','$H_{\bar{G}}$, hyb','$H_{k}$, LC','$H_{k}$, hyb','$L^{2}(\rho)$, LC','$L^{2}(\rho)$, hyb'};
fade_ratio = 0.7;  
legendMarkers = gobjects(n_methods, 1);
for i = 1:n_methods
    legend_color = (1 - fade_ratio) * colors(i,:) + fade_ratio * [1 1 1];  
    legendMarkers(i) = plot(nan, nan, 's', ...
        'MarkerSize', 8, ...
        'MarkerFaceColor', legend_color, ...
        'MarkerEdgeColor', colors(i,:), ...
        'LineWidth', 1.2, ...
        'DisplayName', labels{i});
end
legend(legendMarkers, labels, 'Interpreter', 'latex', 'FontSize', 12, 'Location', 'northeast');
set(gca, 'XTick', 1:n_methods, 'XTickLabel', 1:n_methods,'FontSize', 14);
ylabel('Relative  error','Fontsize',14);
title('Iter. Est.','FontSize',16,'interpreter','latex');
grid on;
grid minor;
set(gca, 'GridAlpha', 0.1);
set(gca, 'MinorGridAlpha', 0.01);
box on;


