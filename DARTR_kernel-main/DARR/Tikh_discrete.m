function [x_reg,res,eta,reg_para] = Tikh_discrete(regressionData,regu_type,method,plotOn,varargin)
% Tikhonov regularization for RKHS regularization, where we first 
% discretize the system by sampling points using the RKHS kernel, or
% by Riemann summation 
%
% Input:
%   regressionData: generated data
%   regu_type: 'auto-RKHS', 'auto-no-rho', 'Gaussian-RKHS', L2-rho
%   method: L-curve, GCV
%   plotOn: plot the L-curve or GCV curve
%   varargin: bandwighth of Gussian kernel
%
% Output:
%   x_reg: regularization 
%   res: redisual norm
%   eta: solution norm
%   reg_para: regularization parameter
%

% Check for acceptable number of input arguments
if nargin < 4
    error('Not Enough Inputs')
elseif nargin==3 && strcmp(regu_type, 'Gaussian-RKHS')
    error('Not Enough Inputs')
end

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
rho_val = regressionData.rho_val;
rho = rho_val(:);

fx_vec = regressionData.fx_vec';    % Jxn0
b      = fx_vec(:);                 % n0J, vectorized from (1,{x_j}) to (n_0,{x_j})


if strcmp(regu_type, 'auto-RKHS')
    G      = regressionData.G;             %   G      = \sum_{k,j} g_ukxj' * g_ukxj *dx /N  = g1'*g1/n0J   
    Gbar = G./(rho_val*rho_val');        %   Gbar_D = L'*L; 
elseif strcmp(regu_type, 'auto-no-rho')
    Gbar = regressionData.G;
elseif strcmp(regu_type, 'Gaussian-RKHS')
    l = varargin{1};
    G_fun = @(s1,s2) Gauss(s1, s2, l);
    r_seq = r_seq(:);    % nsx1
    rr1   = r_seq * ones(1,ns);
    rr2   = rr1';
    G_mat = arrayfun(G_fun, rr1(:), rr2(:));
    Gbar = reshape(G_mat, ns, ns);   % form G_matrix with evaluations on {s_l}    
end


%----------------------------------------------------------------
if strcmp(regu_type, 'auto-RKHS') || strcmp(regu_type, 'auto-no-rho')
    % solve min {||g'*L'*dr*Lx-b||^2+lambda*||Lx||^2.
    % Let z = Lx to avoid the GSVD of ill-conditioned matrix pair {A,L}.
    % L = B;    % for auto-kernel, the square root of Gram matrix is known
    % Gbar = Gbar + 1e-14*eye(size(Gbar,1));
    L = sqrtm1(Gbar);
    A1 = g1 * L' * dx;
    [U,s,V] = csvd(A1);
    n = size(A1,2);
    z0 = zeros(n,1);

    if strcmp(method, 'LC')
        [reg_para,~,~,~] = l_curve(U,s,b,'Tikh',plotOn);
    elseif strcmp(method, 'gcv')
        [reg_para,~,~] = gcv(U,s,b,'Tikh',plotOn);
    else
        error('Wrong parameter choice rule')
    end
    [z_reg,res,~] = tikhonov(U,s,V,b,reg_para,z0);
    c_reg = L \ z_reg;
    x_reg = Gbar * c_reg;
    eta = norm(x_reg);
elseif strcmp(regu_type, 'Gaussian-RKHS')
    % Gbar = Gbar + 1e-14*eye(size(Gbar,1));
    L = sqrtm1(Gbar);    % for other kernel, the square root of Gram matrix need to be computed
    A = g1 * Gbar * dx;
    [U,sm,X,~,~] = gsvd1(A,L);
    n = size(A,2);
    c0 = zeros(n,1);
    if strcmp(method, 'LC')
        [reg_para,~,~,~] = l_curve(U,sm,b,'Tikh',plotOn);
    elseif strcmp(method, 'gcv')
        [reg_para,~,~] = gcv(U,sm,b,'Tikh',plotOn);
    else
        error('Wrong parameter choice rule')
    end
    [c_reg,res,~] = tikhonov(U,sm,X,b,reg_para,c0);
    x_reg = Gbar * c_reg;
    eta = norm(x_reg);
elseif strcmp(regu_type, 'L2-rho')
    B_inv = diag(sqrt(1./rho));
    A = g1 * dx;
    A1 = A * B_inv;
    n = size(A1,2);
    [U,s,V] = csvd(A1);
    z0 = zeros(n,1);
    if strcmp(method, 'LC')
        [reg_para,~,~,~] = l_curve(U,s,b,'Tikh',plotOn);
    elseif strcmp(method, 'gcv')
        [reg_para,~,~] = gcv(U,s,b,'Tikh',plotOn); 
    else
        error('Wrong parameter choice rule')
    end
    [z_reg,res,~]    = tikhonov(U,s,V,b,reg_para,z0);
    x_reg = B_inv * z_reg;
    eta = norm(x_reg);
else
    error('wrong regulatization type')
end

end


%-----------------------
function val = Gauss(s1, s2, l)
    d = (s1-s2)^2;
    val = exp(-d/(2*l));
end


%------------------------
function L = sqrtm1(G)
    G = (G+G') / 2;
    tol = 1e-16;
    [~,s,V] = csvd(G);
    ind = find(s>tol);   
    Vr  = V(:,ind);      % compute an x\in R(Vr)
    sr  = sqrt(s(ind));
    L = Vr * diag(sr) * Vr';
end