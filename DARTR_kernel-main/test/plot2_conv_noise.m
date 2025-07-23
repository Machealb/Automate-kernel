% Plot boxchart of the convergence of estimators as noise goes 
% to zero for multiple simulations
% 

clear; clc; close all; restoredefaultpath;  
addpath('../')
add_mypaths;
saveON = 0;


%%% 0 setttings: R_phi[u](x)  = \sum_r phi(r) * fun_g[u](x,r) dr, x in [0,1], r= [0,R0]
% % the first example----Itegral operator
% dx       = 0.005;                          % space mesh size in observation. 
% N        = 10;                               %  number of data pairs (u_i,f_i)
% u_Type   = 'stocFourier';        % types: 'Bspline', 'Fourier'; 'stocFourier';  'stocCosine' randomDensity    
% jump_disc = 0;               % jump discontinuity to increase rank of G 
% 
% R0       = 1;                % maximal interaction range [0,R0] for radial kernel
% supp_u   = [-R0 1+R0];       % data u support    >>> f(x) with x = [0,1] 
% example_type = 'LinearIntOpt';   % {'LinearIntOpt','nonlocal','Aggregation_StrForm'};
% kernel_type  = 'sinx_smooth';       % Gaussian, sinkx, FracLap, sinx_smooth

  
% the second example-----Nonlocal operator
dx       = 0.005;                          % space mesh size in observation. 
N        = 6;                               %  number of data pairs (u_i,f_i)
u_Type   = 'stocFourier';        % types: 'Bspline', 'Fourier', 'stocFourier';  'stocCosine' randomDensity   
jump_disc = 1;               % jump discontinuity to increase rank of G 

R0       = 1;                % maximal interaction range [0,R0] for radial kernel
supp_u   = [-R0 1+R0];       % data u support    >>> f(x) with x = [0,1] 
example_type = 'nonlocal';   % {'LinearIntOpt','nonlocal','Aggregation_StrForm'};
kernel_type  = 'sinkx';       % Gaussian, sinkx, FracLap


% % the third example-----Aggregation operator
% dx       = 0.005;                          % space mesh size in observation. 
% N        = 10;                               %  number of data pairs (u_i,f_i)
% u_Type   = 'randomDensity';        % types: 'Bspline', 'Fourier', 'stocFourier';  'stocCosine' randomDensity   
% jump_disc = 0;               % jump discontinuity to increase rank of G 
% 
% R0       = 1;                % maximal interaction range [0,R0] for radial kernel
% supp_u   = [-R0 1+R0];       % data u support    >>> f(x) with x = [0,1] 
% example_type = 'Aggregation_StrForm';   % {'LinearIntOpt','nonlocal','Aggregation_StrForm'};
% kernel_type  = 'sinx_cubic';       % Gaussian, sinkx, FracLap% sinx_smooth, powerFn, sinx_cubic


[kernelInfo, obsInfo]  = load_settings_v2(N, u_Type,jump_disc, supp_u, R0, dx,kernel_type,example_type);
integrator = 'quadgk'; %  'Riemann', 'quadgk'

obsInfo.plotON = 0; % no plots in the parallel simulations. 


%%%---------settings for multiple simulations of Tikhonov&Iterative estimators------------
numSimulations = 20;
rkhsType = {'auto-RKHS','Gaussian-RKHS','L2-rho'};
Tikh_para = {'LC','gcv'};
Iter_para = {'LC','hyb'};
n_type = length(rkhsType);
nsr = [1, 2^(-1), 2^(-2), 2^(-3), 2^(-4), 2^(-5)];   % noise-signal-ratio
nn = length(nsr);
maxIter = [25, 30, 30, 35, 40, 40];
showError = 0;
plotOn = 0; 

err_Tikh_LC  = zeros(nn,n_type,numSimulations);    % Tikh_err_LC
err_Tikh_GCV = zeros(nn,n_type,numSimulations);    % Tikh_err_gcv
err_Iter_LC  = zeros(nn,n_type,numSimulations);    % Iter_err_LC
err_Iter_hyb = zeros(nn,n_type,numSimulations);    % Iter_err_hyb
err_Iter_opt = zeros(nn,n_type,numSimulations);    % Iter_err_opt


% Compute regularized solutions, auto basis for kernel methods -------------------
for j = 1:numSimulations
    Tikh_err_LC = zeros(nn,n_type);    % errors for four norms
    Tikh_err_gcv = zeros(nn,n_type);
    Iter_err_LC = zeros(nn,n_type);
    Iter_err_hyb = zeros(nn,n_type);
    Iter_err_opt = zeros(nn,n_type);

    for i = 1:nn
        noise_ratio = nsr(i);  
        obsInfo.noise_ratio  = noise_ratio;  
        % Generate data on x_mesh (adaptive to u, f) and pre-process data
        rng(i*j);
        [obsInfo,ux_val,fx_val]   = generateData2(kernelInfo, obsInfo, SAVE_DIR,saveON,integrator);   % get observation data
        if strcmp(example_type, 'Aggregation_StrForm')
            noise_std_upperBd = max(round(obsInfo.noise_std_upperBd,2),0.1);
        else
            noise_std_upperBd = max(round(obsInfo.noise_std_upperBd,2),0.001);
        end
        nsr_i               = obsInfo.noise_ratio;
        noise_std         = nsr_i *noise_std_upperBd;  obsInfo.noise_std = noise_std;% noise added to fx_val
        fx_val            = fx_val + noise_std.*randn(size(fx_val));

        dx        = obsInfo.x_mesh_dx; 
        data_str  = [obsInfo.example_type,kernelInfo.kernel_type,obsInfo.u_str,obsInfo.x_mesh_str,sprintf('NSR%1.1f_dx%1.4f_',nsr_i,dx)];

        % get boundary width and x-mesh in use. --------                            
        bdry_width = obsInfo.bdry_width; 
        r_seq      = dx*(1:bdry_width);
        Index_xi_inLoss = (bdry_width+1): (length(obsInfo.u_xmesh) - bdry_width); 

        %%% 2. pre-process data, get all data elements for regression
        normalizeOn    = ~strcmp(obsInfo.example_type, 'classicalReg');
        % normalizeOn    = ~strcmp(obsInfo.example_type, 'classicalReg');
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
        r_seq   = regressionData.r_seq(ind_rho);   
        dr      = r_seq(2)-r_seq(1);   
        K_true   = kernelInfo.K_true; 
        K_true_val  = K_true(r_seq);

        xx  = K_true_val';
        [Tikhonov_sol, Tikhonov_lamb, Tikhonov_err] = comp_Tikh_regu(regressionData, rkhsType, Tikh_para, xx, showError,plotOn);

        Tikh_err_LC(i,1) = Tikhonov_err.autoRKHS_LC;
        Tikh_err_LC(i,2) = Tikhonov_err.GaussRKHS_LC;
        Tikh_err_LC(i,3) = Tikhonov_err.L2rho_LC;

        Tikh_err_gcv(i,1) = Tikhonov_err.autoRKHS_gcv;
        Tikh_err_gcv(i,2) = Tikhonov_err.GaussRKHS_gcv;
        Tikh_err_gcv(i,3) = Tikhonov_err.L2rho_gcv;

        [Iter_sol, Iter_stop, Iter_lamb, Iter_err, Iter_all_err] = comp_Iter_regu(regressionData, rkhsType, Iter_para, maxIter(i), xx, showError);

        Iter_err_LC(i,1) = Iter_err.autoRKHS_LC;
        Iter_err_LC(i,2) = Iter_err.GaussRKHS_LC;
        Iter_err_LC(i,3) = Iter_err.L2rho_LC;

        Iter_err_hyb(i,1) = Iter_err.autoRKHS_hyb;
        Iter_err_hyb(i,2) = Iter_err.GaussRKHS_hyb;
        Iter_err_hyb(i,3) = Iter_err.L2rho_hyb;

        Iter_err_opt(i,1) = Iter_err.autoRKHS_opt;
        Iter_err_opt(i,2) = Iter_err.GaussRKHS_opt;
        Iter_err_opt(i,3) = Iter_err.L2rho_opt;
    end

    err_Tikh_LC(:,:,j) = Tikh_err_LC;
    err_Tikh_GCV(:,:,j) = Tikh_err_gcv;
    err_Iter_LC(:,:,j) = Iter_err_LC;
    err_Iter_hyb(:,:,j) = Iter_err_hyb;
    err_Iter_opt(:,:,j) = Iter_err_opt;
end

test_settings.numSimulations    = numSimulations; 
test_settings.rho_type          = rho_type; 
test_settings.noise_std_upperBd = noise_std_upperBd; 
test_settings.nsr               = nsr; 
test_settings.integrator        = integrator; 
test_settings.example_type      = example_type; 
test_settings.kernel_type       = kernel_type;

datafilename = 'test_results.mat';
save(datafilename,'test_settings',"err_Iter_opt","err_Iter_hyb","err_Iter_LC","err_Tikh_GCV","err_Tikh_LC","obsInfo","kernelInfo"); 
fig_name = 'fig_conv_'; 

%-----------List and plot----------------------
y_scale = 'log';

figure;
hold on;
[~, p] = plot_mean_std(permute(err_Tikh_LC, [3, 2, 1]), y_scale);    % get [numSimulations, n_type, nn]
handle=legend(p, {'$H_{\bar{G}}$ norm','$H_k$ norm','$L^2(\rho)$ norm'});
set(handle,'Fontsize',14,'interpreter','latex','Location', 'best');
xlabel('Noise  level','fontsize',16);
ylabel('Relative  error','Fontsize',16);
xticklabels(string(nsr));
title('Errors of Tikh. Est. with L-curve','FontSize',16,'interpreter','latex'); 
savefig([fig_name,'Tikh_LC.fig']); 

figure;
hold on;
[~, p] = plot_mean_std(permute(err_Tikh_GCV, [3, 2, 1]), y_scale);   
handle=legend(p, {'$H_{\bar{G}}$ norm','$H_k$ norm','$L^2(\rho)$ norm'});
set(handle,'Fontsize',14,'interpreter','latex','Location', 'best');
xlabel('Noise  level','fontsize',16);
ylabel('Relative  error','Fontsize',16);
xticklabels(string(nsr));
title('Errors of Tikh. Est. with GCV','FontSize',16,'interpreter','latex'); 
savefig([fig_name,'Tikh_GCV.fig']); 

figure;
hold on;
[~, p] = plot_mean_std(permute(err_Iter_LC, [3, 2, 1]), y_scale);    
handle=legend(p, {'$H_{\bar{G}}$ norm','$H_k$ norm','$L^2(\rho)$ norm'});
set(handle,'Fontsize',14,'interpreter','latex','Location', 'best');
xlabel('Noise  level','fontsize',16);
ylabel('Relative  error','Fontsize',16);
xticklabels(string(nsr));
title('Errors of Iter. Est. with L-curve','FontSize',16,'interpreter','latex'); 
savefig([fig_name,'Iter_LC.fig']); 

figure;
hold on;
[~, p] = plot_mean_std(permute(err_Iter_hyb, [3, 2, 1]), y_scale);    
handle=legend(p, {'$H_{\bar{G}}$ norm','$H_k$ norm','$L^2(\rho)$ norm'});
set(handle,'Fontsize',14,'interpreter','latex','Location', 'best');
xlabel('Noise  level','fontsize',16);
ylabel('Relative  error','Fontsize',16);
xticklabels(string(nsr));
title('Errors of Hybrid Est.','FontSize',16,'interpreter','latex'); 
savefig([fig_name,'Iter_hyb.fig']); 


figure;
hold on;
[~, p] = plot_mean_std(permute(err_Iter_opt, [3, 2, 1]), y_scale);    
handle=legend(p, {'$H_{\bar{G}}$ norm','$H_k$ norm','$L^2(\rho)$ norm'});
set(handle,'Fontsize',14,'interpreter','latex','Location', 'best');
xlabel('Noise  level','fontsize',16);
ylabel('Relative  error','Fontsize',16);
xticklabels(string(nsr));
title('Errors of optimal Iter. Est.','FontSize',16,'interpreter','latex'); 
savefig([fig_name,'Iter_opt.fig']); 