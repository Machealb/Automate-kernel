clear; clc; close all; restoredefaultpath;  
addpath('../')
add_mypaths;
% rng(1);
plotOn = 1; saveON = 0; 

%{
Learn the convolution kernel phi in the operator: 
         R_phi[u](x)  = \sum_r phi(r) * fun_g[u](x,r) dr,   
At observation mesh points: 
               f_vec  =  g_array * phi_vec,        for each u 
     size:    J x 1      J x n_r    n_r x 1              nx x 1;    nx>J

  - Key: r ∈ [0,R0], x in [a,b] such that [a-R0,b+R0] \subset supp(u), 
         so that the above integral can be computed from data u for Nonlocal and Aggregation operators. 
  - The interaction range R0 is given, but the data only explore part of [0,R0].   
  - The function fun_g[u](x,r) encodes the fact that the kernel is radial.

Data f:
  - x\in [0,1] for simplicity. It is 1 period of u.  

Data u: 
  - random samples of the 1-periodic process 
        u(x) = \sum_{n=2} X_n \cos(2pi n x),    x\in [-R0,1+R0],  
    where $X_n\sim N(0,sigma_n^2)$ such that \sum_n \sigma_n<\infty. 

Note: 
  1. The x-mesh of observed output function f and x-mesh of the input
  function u are different in the above operator.  
    - In time-dependent PDEs, e.g., f = \partial_t u- \Delta u, f has the 
      same x domain as u. We use index_xi_inUse to show the difference. 
  2. The interaction range R0 may not be given in practice. We can first detect
  it from all the data, then re-arrange the meshes. This is done in DARTR.
  We skip this step in this study to simplify the tests. 
  3. The aggregation operator has input being probability densities in the
  mean-field equations. Here, our data are not. We can add such examples later.  
%}


%%% 0 setttings:  
%       R_phi[u](x)  = \sum_r phi(r) * fun_g[u](x,r) dr, x in [0,1], r= [0,R0]
% dx       = 0.005;                          % space mesh size in observation. 
% N        = 12;                               %  number of data pairs (u_i,f_i)
% u_Type   = 'stocFourier';        % types: 'Bspline', 'Fourier', 'stocFourier';  'stocCosine' %  Fourier   
% jump_disc = 0;                   % jump discontinuity to increase rank of G 
% 
% R0       = 1;                    % maximal interaction range [0,R0] for radial kernel
% supp_u   = [-R0 1+R0];           % data u support    >>> f(x) with x = [0,1] 
% example_type = 'LinearIntOpt';   % {'LinearIntOpt','nonlocal','Aggregation_StrForm'};
% kernel_type  = 'sinx_smooth';       % Gaussian, sinkx, FracLap, sinx_smooth, polyx, powerFn, sinx_cubic

% the second example-----Nonlocal operator
dx       = 0.005;                          % space mesh size in observation. 
N        = 6;                               %  number of data pairs (u_i,f_i)
u_Type   = 'stocFourier';        % types: 'Bspline', 'Fourier', 'stocFourier';  'stocCosine' randomDensity   
jump_disc = 1;               % jump discontinuity to increase rank of G 
R0       = 1;                % maximal interaction range [0,R0] for radial kernel
supp_u   = [-R0 1+R0];       % data u support    >>> f(x) with x = [0,1] 
example_type = 'nonlocal';   % {'LinearIntOpt','nonlocal','Aggregation_StrForm'};
kernel_type  = 'sinkx';       % Gaussian, sinkx, FracLap

%       R_phi[u](x)  = \sum_r phi(r) * fun_g[u](x,r) dr, x in [0,1], r= [0,R0]
% dx       = 0.005;                          % space mesh size in observation. 
% N        = 10;                             %  number of data pairs (u_i,f_i)
% u_Type   = 'randomDensity';        % types: 'Bspline', 'Fourier', 'stocFourier';  'stocCosine', randomDensity  
% jump_disc = 0;                     % jump discontinuity to increase rank of G 
% 
% R0       = 1;                    % maximal interaction range [0,R0] for radial kernel
% supp_u   = [-R0 1+R0];           % data u support    >>> f(x) with x = [0,1] 
% example_type = 'Aggregation_StrForm';   % {'LinearIntOpt','nonlocal','Aggregation_StrForm'};
% kernel_type  = 'sinx_cubic';       % Gaussian, sinkx, FracLap, sinx_smooth, polyx, powerFn, sinx_cubic



%{
Proper settings:  smoothness of the data and the kernel must match.  
                        (jump_disc, kernel_type)
- LinearIntOpt:        (0,sinx_smooth), (1,sinkx)   
- nonlocal operator:   (0,polyx),       (1,sinkx); 
- Aggregation_StrForm: (0,sinx_smooth), (1,sinkx)
  + LinearIntOpt: Gaussian-RKHS has an advantange when true kernel is similar to Gaussian. 
  + Nonlocal: sinx_smooth is not good because it is almost orthogonal to the DA-RKHS.      no good estimator. 
  + Aggregation_StrForm: polyx is not good because it is almost orthogonal to the DA-RKHS. no good estimator.  
  + The key is the projection the true kernel in the DA-RKHS and the regu-RKHS.  


Notes: ------ ( to remove after Haibo runs through the code.)   
 + FL250607 TODO: make Tikh methods tolerate difficient ranked G. The code returns error at cgsvd when setting  
   [ jump_disc = 0;  example_type = 'LinearIntOpt', kernel_type  =   'Gaussian'; ]
   >> DONE by Haibo: adding 1e-14*eye(n_s) to G and Gbar_D. However, after
   removing it, the code still work. Where are the other changes? 
   ~ The correction to L in auto_kernel_mat.m is a multiplicative factor, which would not remove the error. 
 + 0608: Auto-kernel or Gbar are better than Gaussian-kernel only when jump_disc= 1; kernel_type = sinkx; not in other cases.  
 + 0610: added stochastic u_k --- set u_Type = stocFourier. The results are similar to the Fourier case.   
        - TODO: investigate why 
            (1) Gaussian-RKHS outperforms auto-KRHS for smooth data; 
            (2) why Gaussian-RKHS can be as good as auto-RKHS when integrator = 'quadgk'; 
               >> ANS: Ax=b form, not numerical error in L_K^* * L_K phi = L_K^* * f
   >>> 0615: updated x_mesh is a subset set of u_xmesh, removing machine precision error. 
    - Gaussian-RKHS > auto-RKHS when: 
      + LinearIntOpt + Gaussian + usmooth: quadgk (slightly better); Riemann  (significantly)-- shift error TBD;
      + nonlocal + Gaussian + usmooth: quadgk (slightly better);
%}

[kernelInfo, obsInfo]  = load_settings_v2(N, u_Type,jump_disc, supp_u, R0, dx,kernel_type,example_type);
noise_ratio            = 0.03125;  obsInfo.noise_ratio  = noise_ratio;    


%%% 1. Generate data on x_mesh (adaptive to u, f) and pre-process data
integrator = 'quadgk'; %  'Riemann', 'quadgk'     % Riemann sum for checking. Quadgk for testing. 
fprintf('u smooth: %i,   integrator: %s \n ', 1-jump_disc,integrator); 
obsInfo.plotON = 0; % will be set to 0 in the parallel simulations. 

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
r_seq   = regressionData.r_seq(ind_rho);         % when r_seq is non-uniform, use dr = r_seq(2:end) - r_seq(1:end-1).       
dr      = r_seq(2)-r_seq(1);   

K_true   = kernelInfo.K_true; 
K_true_val  = K_true(r_seq);

% sanity check
sanity_checkOn = 1; 
if sanity_checkOn ==1
    g_ukxj  = regressionData.g_ukxj;
    fu_all  = regressionData.fx_vec;
    Rphiu   = 0*fu_all;
    for k=1:N
        guk   = squeeze(regressionData.g_ukxj(:,:,k));
        Rphiu(k,:) = K_true_val*guk*dx;
    end
    diff = Rphiu - fu_all;
    figure(150); clf;  subplot(311); plot(diff'); title('check: Rphi[u]-f =0 ');
end


%%%----- Tikhnov regularization with L-curve, auto-basis funcitons
plotOn = 1; 
[x_reg0,res0,eta0,reg0_corner] = Tikh_auto_basis(regressionData, 'auto-RKHS','gcv',plotOn);

rho_val = regressionData.rho_val; 
ind_rho = find(rho_val>0);  rho_val = rho_val(ind_rho);
r_seq   = regressionData.r_seq(ind_rho);    % when r_seq is non-uniform, use dr = r_seq(2:end) - r_seq(1:end-1).       
dr      = r_seq(2)-r_seq(1);   
K_true   = kernelInfo.K_true; 
K_true_val  = K_true(r_seq);

g = regressionData.g_ukxj;
[ns, J, n0] = size(g);
k  = n0*J;
g1 = zeros(ns, k);
for i = 1:n0
    g1(:,(i-1)*J+1:i*J) = g(:,:,i);
end
g1 = g1';  % n0Jxns
    
r_seq   = regressionData.r_seq;    % when r_seq is non-uniform, use dr = r_seq(2:end) - r_seq(1:end-1).       
dx      = r_seq(2)-r_seq(1); 
A = g1 * dx;
rho = regressionData.rho_val;
fx_vec = regressionData.fx_vec';    % Jxn0
f      = fx_vec(:);  



%%% 3. Iterative regularizations -----------------------------------------------------------------
% compute auto-regularized solution
K = 50;
[G_D1, basis_D1, Sigma_D1, LL] = auto_kernel_mat(regressionData, 'auto');
[X1, res1, eta1, iter_stop1] = idarr(regressionData, 'auto', K, 'LC', 0);
[k1,~] = Lcurve_corner(res1,eta1,111,'iDarr, auto-RKHS');
x_reg1 = X1(:,iter_stop1);

% compute Gaussian-RKHS solution
[X3, res3, eta3, iter_stop3] = idarr(regressionData, 'gauss', K, 'LC', 0, 0.1);
[k3,~] = Lcurve_corner(res3,real(eta3),3,'iDarr, Gaussian-RKHS');      
x_reg3 = X3(:,iter_stop3);

% L2(rho) regularization
[X5, res5, eta5, iter_stop5] = idarr(regressionData, 'L2-rho', K, 'LC', 0);
[k5,~] = Lcurve_corner(res5,eta5,5,'L2-rho regu');
x_reg5 = X5(:,iter_stop5);

% hybrid method
[X2, res2, lamb1, iter_stop2] = idarr(regressionData, 'auto', K, 'hyb', 0);
x_reg2 = X2(:,iter_stop2);

[X4, res4, lamb4, iter_stop4] = idarr(regressionData, 'gauss', K, 'hyb', 0, 0.01);
x_reg4 = X4(:,iter_stop4);

degflat = 1e-4;
[X6, res6, lamb6, GCV6, iter_stop6] = wlsqr_hyb(A, rho, f, K, degflat, 1, 0);
% [X6, res6, lamb6, iter_stop6] = idarr(regressionData, 'L2-rho', K, 'hyb', 0);
x_reg6 = X6(:,iter_stop6);




% relative error
er1 = zeros(length(eta1),1);
er2 = zeros(length(res2),1);
er3 = zeros(length(eta3),1);
er4 = zeros(length(res4),1);
er5 = zeros(length(res5),1);
er6 = zeros(length(res6),1);
xx  = K_true_val';
nx = sqrt( dr*rho_val'*(xx.^2)); % norm(phi_true);
for i=1:length(eta1)
    er1(i) = sqrt( dr*rho_val'*((X1(:,i)-xx).^2)) / nx; % norm(X1(:,i)-xx) / nx;
end
for i=1:length(er2)
    er2(i) = sqrt( dr*rho_val'*((X2(:,i)-xx).^2)) / nx; % norm(X1(:,i)-xx) / nx;
end
for i=1:length(eta3)
    er3(i) = sqrt( dr*rho_val'*((X3(:,i)-xx).^2)) / nx; % norm(X3(:,i)-xx) / nx;
end
for i=1:length(er4)
    er4(i) = sqrt( dr*rho_val'*((X4(:,i)-xx).^2)) / nx; % norm(X1(:,i)-xx) / nx;
end
for i=1:length(er5)
    er5(i) = sqrt( dr*rho_val'*((X5(:,i)-xx).^2)) / nx; % norm(X1(:,i)-xx) / nx;
end
for i=1:length(er6)
    er6(i) = sqrt( dr*rho_val'*((X6(:,i)-xx).^2)) / nx; % norm(X1(:,i)-xx) / nx;
end


er0 = sqrt(dr*rho_val'*((x_reg0-xx).^2)) / nx;


% select optimal k for iterative methods 
[~, k1_opt] = min(er1);
[~, k3_opt] = min(er3);
[~, k5_opt] = min(er5);



%%% Compare estimators  ------------------------------------------------------------

% plot estimators 
figure(11);  
plot(r_seq,K_true_val,'k:','Linewidth', 3); 
hold on;
plot(r_seq, x_reg1, '-.','Linewidth', 2);
hold on;
plot(r_seq, x_reg2, '-.','Linewidth', 2);
hold on;
plot(r_seq, x_reg3, '-.','Linewidth', 2);
hold on;
plot(r_seq, x_reg5, '-.','Linewidth', 2);
legend('True', 'auto-RKHS','auto-RKHS-hybrid','Gaussian-RKHS','L2-rho regu');
title('Iterative Estimators, L-curve'); 

figure(12);
plot(r_seq,K_true_val,'k:','Linewidth', 3); 
hold on;
plot(r_seq, X1(:,k1_opt), '-.','Linewidth', 2);
hold on;
plot(r_seq, x_reg2, '-.','Linewidth', 2);
hold on;
plot(r_seq, X3(:,k3_opt), '-.','Linewidth', 2);
hold on;
plot(r_seq, X5(:,k5_opt), '-.','Linewidth', 2);
legend('True', 'auto-RKHS','auto-RKHS-hybrid','Gaussian-RKHS','L2-rho regu');
title('Iterative estimators, opt'); 







