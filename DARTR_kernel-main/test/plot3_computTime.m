% Plot boxchart of the computational time of 
% Tikhonov/Iterative methods as n_0*J increases
% 

clear; clc; close all; restoredefaultpath;  
addpath('../')
add_mypaths;
saveON = 0;


%%% 0 setttings: R_phi[u](x)  = \sum_r phi(r) * fun_g[u](x,r) dr, x in [0,1], r= [0,R0]
% the first example----Itegral operator
dx       = 0.005;                          % space mesh size in observation. 
u_Type   = 'stocFourier';        % types: 'Bspline', 'Fourier'; 'stocFourier';  'stocCosine' %  Fourier   
jump_disc = 0;               % jump discontinuity to increase rank of G 

R0       = 1;                % maximal interaction range [0,R0] for radial kernel
supp_u   = [-R0 1+R0];       % data u support    >>> f(x) with x = [0,1] 
example_type = 'LinearIntOpt';   % {'LinearIntOpt','nonlocal','Aggregation_StrForm'};
kernel_type  = 'sinx_smooth';       % Gaussian, sinkx, FracLap, sinx_smooth

  
% the second example-----Nonlocal operator
% dx       = 0.005;                          % space mesh size in observation.                              %  number of data pairs (u_i,f_i)
% u_Type   = 'stocFourier';        % types: 'Bspline', 'Fourier', 'stocFourier';  'stocCosine' %  Fourier   
% jump_disc = 1;               % jump discontinuity to increase rank of G 
% 
% R0       = 1;                % maximal interaction range [0,R0] for radial kernel
% supp_u   = [-R0 1+R0];       % data u support    >>> f(x) with x = [0,1] 
% example_type = 'nonlocal';   % {'LinearIntOpt','nonlocal','Aggregation_StrForm'};
% kernel_type  = 'sinkx';       % Gaussian, sinkx, FracLap


% the third example-----Aggregation operator
% dx       = 0.005;                          % space mesh size in observation. 
% u_Type   = 'stocFourier';        % types: 'Bspline', 'Fourier', 'stocFourier';  'stocCosine' %  Fourier   
% jump_disc = 0;               % jump discontinuity to increase rank of G 
% 
% R0       = 1;                % maximal interaction range [0,R0] for radial kernel
% supp_u   = [-R0 1+R0];       % data u support    >>> f(x) with x = [0,1] 
% example_type = 'Aggregation_StrForm';   % {'LinearIntOpt','nonlocal','Aggregation_StrForm'};
% kernel_type  = 'sinx_smooth';       % Gaussian, sinkx, FracLap



%%%---------settings for multiple simulations of Tikhonov&Iterative estimators------------

numSimulations = 50;
Tikh_para = {'LC','gcv'};
Iter_para = {'LC','hyb'};
methods = {'Tikh-LC','Tikh-GCV','Iter-LC','Hyb','Iter-opt'};
n_type  = length(methods);
N = [6,12,18,24,30,36];    %  number of data pairs (u_i,f_i)
nn = length(N);
maxIter = [30, 30, 40, 40, 50, 50];
showError = 0;
plotOn = 0; 

integrator = 'quadgk'; %  'Riemann', 'quadgk'
noise_ratio = 0.1;

err_data  = zeros(nn,n_type,numSimulations);    
time_data = zeros(nn,n_type-1,numSimulations);    % Iter-LC/opt share the same time


% Compute regularized solutions, auto kernel with auto basis-------------------
for j = 1:numSimulations
    err_auto   = zeros(nn,n_type);    
    time_auto  = zeros(nn,n_type-1);

    for i = 1:nn
        N_i = N(i);
        [kernelInfo, obsInfo]  = load_settings_v2(N_i, u_Type,jump_disc, supp_u, R0, dx,kernel_type,example_type);
        obsInfo.noise_ratio  = noise_ratio;
        obsInfo.plotON = 0; % no plots in the parallel simulations. 

        % Generate data on x_mesh (adaptive to u, f) and pre-process data
        rng(i*j);
        [obsInfo,ux_val,fx_val]   = generateData2(kernelInfo, obsInfo, SAVE_DIR,saveON,integrator);   
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

        % pre-process data, get all data elements for regression
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
        r_seq   = regressionData.r_seq(ind_rho);    
        dr      = r_seq(2)-r_seq(1);   
        K_true   = kernelInfo.K_true; 
        K_true_val  = K_true(r_seq);
        xx  = K_true_val';
        nx = norm(xx);

        % generate matrix and vectors
        fx_vec = regressionData.fx_vec';    % Jxn0
        f      = fx_vec(:);                 % n0J, vectorized from (1,{x_j}) to (n_0,{x_j})
        Ki     = maxIter(i);
        D      = ones(length(f),1);


        %-------auto-RKHS kernel-----------------
        [~, basis_D1, Sigma_D1] = auto_kernel_mat(regressionData, 'auto');

        tic;
        tol = 1e-14;
        [~,s,V] = csvd(Sigma_D1);
        ind = find(s>tol);   % numerical rank of Sigma_D
        r = length(ind);
        Vr  = V(:,ind);      % compute an x\in R(Vr)
        sr  = s(ind);
        lr  = sqrt(sr);
        [reg_para,~,~,~] = l_curve(Vr,lr,f,'Tikh',plotOn);
        [z_reg,~,~] = tikhonov(Vr,lr,eye(r),f,reg_para,zeros(r,1));
        c_reg = Vr * (z_reg./lr);
        x_reg1 = basis_D1' * c_reg;
        time1 = toc;
        time_auto(i,1) = time1;    % Tikh-LC
        err_auto(i,1) = norm(x_reg1-xx) / nx;
        clear s V ind Vr sr lr c_reg;

        tic;
        tol = 1e-14;
        [~,s,V] = csvd(Sigma_D1);
        ind = find(s>tol);   % numerical rank of Sigma_D
        r = length(ind);
        Vr  = V(:,ind);      % compute an x\in R(Vr)
        sr  = s(ind);
        lr  = sqrt(sr);
        [reg_para,~,~] = gcv(Vr,lr,f,'Tikh',plotOn);
        [z_reg,~,~] = tikhonov(Vr,lr,eye(r),f,reg_para,zeros(r,1));
        c_reg = Vr * (z_reg./lr);
        x_reg2 = basis_D1' * c_reg;
        time2 = toc;
        time_auto(i,2) = time2;    % Tikh-gcv
        err_auto(i,2) = norm(x_reg2-xx) / nx;
        clear s V ind Vr sr lr z_reg c_reg;

        tic;
        [C, ~, ~, iter_stop3] = iter_regu(Sigma_D1, D, f, Ki, 0, 'on');
        x_reg3 = basis_D1' * C(:,iter_stop3);
        time3 = toc;
        time_auto(i,3) = time3;    % Iter-LC
        err_auto(i,3) = norm(x_reg3-xx) / nx;
        err_all = vecnorm(basis_D1'*C-xx, 2, 1)' / nx;
        [err_opt, iter_opt] = min(err_all);
        err_auto(i,5) = err_opt;
        clear C;

        tic;
        [C, ~, ~, ~, iter_stop4] = hyb_regu(Sigma_D1, D, f, Ki, 1e-5, 1, 0);
        x_reg4 = basis_D1' * C(:,iter_stop4);
        time4 = toc;
        time_auto(i,4) = time4;    % Iter-hyb
        err_auto(i,4) = norm(x_reg4-xx) / nx;
        clear C;
    end

    err_data(:,:,j)  = err_auto;
    time_data(:,:,j) = time_auto;
end



%-----------List and plot----------------------
y_scale = 'log';    % or 'linear'

% figure;
% hold on;
% [~, p] = plot_mean_std(permute(err_data, [3, 2, 1]), y_scale);    % 得到 [numSimulations, n_type, nn]
% handle=legend(p, {'Tikh.-LC','Tikh.-GCV','Iter.-LC','Hyb.-WGCV','Iter.-opt.'});
% set(handle,'Fontsize',14,'interpreter','latex','Location', 'best');
% xlabel('$n_{0}$','fontsize',20,'interpreter','latex');
% ylabel('Relative  error','Fontsize',14);
% xticklabels(string(N));
% title('Errors of Est. for $H_{\bar{G}}$ regu.','FontSize',16,'interpreter','latex'); 

figure;
hold on;
[~, p] = plot_mean_std(permute(time_data, [3, 2, 1]), 'linear');   
% handle=legend(p, {'Tikh.-LC','Tikh.-GCV','Iter.-LC','Hyb.-WGCV'});
% set(handle,'Fontsize',14,'interpreter','latex','Location', 'best');
xlabel('$n_{0}$','fontsize',23,'interpreter','latex');
ylabel('Time (seconds)','fontsize',16);
xticklabels(string(N));
title('Running time for $H_{\bar{G}}$ regu.','FontSize',16,'interpreter','latex'); 
