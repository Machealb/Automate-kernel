function [X, res, eta, iter_stop] = idarr(regressionData, rkhs_type, K, stop_rule, nsr, varargin) 
% Iterative data-adaptive RKHS regularization.
%
% Inputs:
%   regressionData: discrtized data from the inverse problem model
%   rhks_type: type of rkhs kernel
%       'auto': data-adaptive RKHS kernel for auto regularization
%       'auto-no-rho': data-adaptive RKHS kernel for auto regularization without using rho
%       'gauss': Gaussian kernel
%       'L2-rho': L2(rho) regularization
%   K: maximum iteration
%   stop_rule:
%       'DP': discrepancy principle
%       'LC': L-curve
%       'hyb': hybrid method with WGCV 
%   nsr: noise norm. Set to 0 if unkown
%   varargin:
%       rkhs_type='gauss', p = varargin(1)---the bandwidth for the Gaussian kernel) 
%

% Check for acceptable number of input arguments
if nargin < 5
    error('Not Enough Inputs')
end

if strcmp(rkhs_type, 'auto')
    [~, basis_D, Sigma_D] = auto_kernel_mat(regressionData, 'auto');
elseif strcmp(rkhs_type, 'auto-no-rho')
    [~, basis_D, Sigma_D] = auto_kernel_mat(regressionData, 'auto-no-rho');
elseif strcmp(rkhs_type, 'gauss')
    l = varargin{1};
    [~, basis_D, Sigma_D] = auto_kernel_mat(regressionData, 'gauss', l);
elseif strcmp(rkhs_type, 'L2-rho')    % this is not an RKHS, just for coding convenience
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
else
    error('Wrong RKHS kernel type')
end

fx_vec = regressionData.fx_vec';    % Jxn0
f      = fx_vec(:);                 % n0J, vectorized from (1,{x_j}) to (n_0,{x_j})

if strcmp(stop_rule, 'LC')
    nsr = 0;  
    NoStop = 'on';  % no noise level provided--the iteration should run to complete and then use L-curve 
elseif strcmp(stop_rule, 'DP')
    NoStop = 'off';
end

% using hybrid method
if strcmp(stop_rule, 'hyb')
    if strcmp(rkhs_type, 'auto')
        D = ones(length(f),1);
        degflat = 1e-5;
        [C, res, Lam, GCV, iter_stop] = hyb_regu(Sigma_D, D, f, K, degflat, 1, 0);
        X = basis_D' * C;
        eta = Lam;
        return
    elseif strcmp(rkhs_type, 'auto-no-rho')
        D = ones(length(f),1);
        degflat = 1e-5;
        [C, res, Lam, GCV, iter_stop] = hyb_regu(Sigma_D, D, f, K, degflat, 1, 0);
        X = basis_D' * C;
        eta = Lam;
        return
    elseif strcmp(rkhs_type, 'gauss')
        D = ones(length(f),1);
        degflat = 1e-4;
        [C, res, Lam, GCV, iter_stop] = hyb_regu(Sigma_D, D, f, K, degflat, 1, 0);
        X = basis_D' * C;
        eta = Lam;
        return    
    elseif strcmp(rkhs_type, 'L2-rho')
        degflat = 1e-4;
        [X, res, Lam, GCV, iter_stop] = wlsqr_hyb(A, rho, f, K, degflat, 1, 0);
        eta = Lam;
        return
    end
end


if strcmp(rkhs_type, 'auto') || strcmp(rkhs_type, 'auto-no-rho') || strcmp(rkhs_type, 'gauss')
    % [C, res, eta, iter_stop] = cg_rkhs(Sigma_D, f, K, stop_rule, nsr, NoStop);
    D = ones(length(f),1);
   % check if f is in range(Sigma_D); 
   %  cc = lsqminnorm(Sigma_D,f);  norm(Sigma_D*cc-f); 
    [C, res, eta, iter_stop] = iter_regu(Sigma_D, D, f, K, nsr, NoStop);
    X = basis_D' * C;  % (ns,n0J)x(n0J,k)-->(ns,k)
elseif strcmp(rkhs_type, 'L2-rho') 
    [X, res, eta, iter_stop] = wlsqr(A, rho, f, K, stop_rule);
end