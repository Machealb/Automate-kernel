function [x_reg,res,eta,reg_para] = Tikh_auto_basis(regressionData,rkhs_type,method,plotOn,varargin)
% Tikhonov regularization for RKHS regularization with finite observations,
% where the solution must lie in the finite subspace spanned by the auto-basis.
% For GCV or L-curve, First compute the SVD: Sigma_D = V*S*V^T, then determine the 
% numerical rank of Sigma_D as r = length(ind) with  ind = find(s>tol).
% let x = Vr*y, then min{||Sigma_D*x-f||^2+lambda x^T*Sigma_D*x} becomes
% min{||Vr*Sr*y-f||^2+lambda ||Lr*y||^2}, whiich is equivalent to
% min{||Vr*Lr*z-f||^2+lambda ||z||^2}, where z=Lr*y.
%
% Input:
%   regressionData: generated data
%   rkhs_type: 'auto-RKHS', 'auto-no-rho', 'Gaussian-RKHS'
%   method: L-curve, GCV, dGCV
%   plotOn: plot the L-curve or GCV curve
%   varargin: bandwighth of Gussian kernel / partition number of dGCV
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
elseif (nargin==4 && strcmp(method, 'dgcv')) || (nargin==4 && strcmp(rkhs_type, 'Gaussian-RKHS'))
    error('Not Enough Inputs')
elseif nargin==5 && strcmp(method, 'dgcv') && strcmp(rkhs_type, 'Gaussian-RKHS')
    error('Not Enough Inputs')
end

if (strcmp(rkhs_type, 'auto-RKHS')||strcmp(rkhs_type, 'auto-no-rho')) && strcmp(method, 'dgcv')
    m = varargin{1};
elseif strcmp(rkhs_type, 'Gaussian-RKHS') && strcmp(method, 'dgcv')
    m = varargin{1};
    l = varargin{2};
elseif strcmp(rkhs_type, 'Gaussian-RKHS') && ~strcmp(method, 'dgcv')
    l = varargin{1};
end

if strcmp(rkhs_type, 'auto-RKHS')
    [~, basis_D, Sigma_D] = auto_kernel_mat(regressionData, 'auto');
elseif strcmp(rkhs_type, 'auto-no-rho')
    [~, basis_D, Sigma_D] = auto_kernel_mat(regressionData, 'auto-no-rho');
elseif strcmp(rkhs_type, 'Gaussian-RKHS')
    l = varargin{1};
    [~, basis_D, Sigma_D] = auto_kernel_mat(regressionData, 'gauss', l);
else
    error('Wrong RKHS kernel type')
end

fx_vec = regressionData.fx_vec';    % Jxn0
f      = fx_vec(:);                 % n0J, vectorized from (1,{x_j}) to (n_0,{x_j})


%%-----------------------------------------------
if strcmp(method, 'gcv')
    tol = 1e-14;
    [~,s,V] = csvd(Sigma_D);
    % n = size(Sigma_D,1);

    ind = find(s>tol);   % numerical rank of Sigma_D
    r = length(ind);
    Vr  = V(:,ind);      % compute an x\in R(Vr)
    sr  = s(ind);
    lr  = sqrt(sr);
    % A1 = Vr * Lr;    % nxr, the SVD of A1 is Vr*Lr*I_r
    [reg_para,~,~] = gcv(Vr,lr,f,'Tikh',plotOn);
    [z_reg,res,~] = tikhonov(Vr,lr,eye(r),f,reg_para,zeros(r,1));
    c_reg = Vr * (z_reg./lr);
    x_reg = basis_D' * c_reg;
    eta = norm(x_reg);
elseif strcmp(method, 'LC')
    tol = 1e-14;
    [~,s,V] = csvd(Sigma_D);
    % n = size(Sigma_D,1);
    ind = find(s>tol);   % numerical rank of Sigma_D
    r = length(ind);
    Vr  = V(:,ind);      % compute an x\in R(Vr)
    sr  = s(ind);
    lr  = sqrt(sr);
    % A1 = Vr * Lr;    % nxr, the SVD of A1 is Vr*Lr*I_r
    [reg_para,~,~,~] = l_curve(Vr,lr,f,'Tikh',plotOn);
    [z_reg,res,~] = tikhonov(Vr,lr,eye(r),f,reg_para,zeros(r,1));
    c_reg = Vr * (z_reg./lr);
    x_reg = basis_D' * c_reg;
    eta = norm(x_reg);
elseif strcmp(method, 'dgcv')
    [N, ~] = size(Sigma_D);
    if mod(N,m) ~= 0
        error('N should be exactly devided by m')
    end

    l = N / m;
    S = zeros(l, m);
    U = zeros(l, l, m);

    for i = 1:m 
        ind = (i-1)*l+1:i*l;
        [~,s,V] = csvd(Sigma_D(ind,ind));
        S(:,i) = s;
        U(:,:,i) = V;
    end

    G_D = basis_D;

    [reg_para, G, param_list] = dgcv(f, Sigma_D, S, U);

    ns = size(G_D, 2);
    G1 = G_D';    %  nsxN
    x_reg = zeros(ns, 1);
    tol = 1e-14;
    for i = 1:m 
        fi   = f((i-1)*l+1:i*l);
        G_Di = G1(:, (i-1)*l+1:i*l);
        Ui = U(:,:,i);
        Si = S(:,i);
        ind = find(Si>tol);
        S1i = 1 ./ (Si(ind)+reg_para);
        x_reg = x_reg + G_Di * (Ui(:,ind) * (diag(S1i) * (Ui(:,ind)' * fi)));
    end
    x_reg = x_reg / m;
    eta = norm(x_reg);
    res = 0;  % we not compute the residual here; it needs the forward matrix
else
    error('Wrong parameter choice rule')
end

end


%------------------------------
function [reg_min,G,reg_param] = dgcv(f, Sigma_D, S, U)
% Plot the dGCV function and find its minimum.

npoints = 200;                       % Number of points on the curve.
smin_ratio = 100*eps;                % Smallest regularization parameter.

% Vector of regularization parameters
reg_param = zeros(npoints,1); 
G = reg_param; 
[l, ~] = size(S);
reg_param(npoints) = max([S(l,1),S(1,1)*smin_ratio]);
ratio = (S(1,1)/reg_param(npoints))^(1/(npoints-1));
for i=npoints-1:-1:1
    reg_param(i) = ratio*reg_param(i+1); 
end

% Vector of GCV-function values
for i=1:npoints
    G(i) = dgcvfun(reg_param(i), f, Sigma_D, S, U);
end 

% Plot dGCV function.
figure;
loglog(reg_param,G,'-'), xlabel('\lambda'), ylabel('dGCV(\lambda)');
title('dGCV function');

% Find minimum, if requested.
[~,minGi] = min(G);    % Initial guess.
gfun = @(x) dgcvfun(x, f, Sigma_D, S, U);
reg_min = fminbnd(gfun, reg_param(min(minGi+1,npoints)),reg_param(max(minGi-1,1)),optimset('Display','off')); 
minG = gfun(reg_min);  % Minimum of GCV function.

ax = axis;
HoldState = ishold; hold on;
loglog(reg_min,minG,'*r',[reg_min,reg_min],[minG/1000,minG],':r')
title(['dGCV function, minimum at \lambda = ',num2str(reg_min)])
axis(ax)
if (~HoldState)
    hold off;
end

end


%-------------------------------------------------
function G = dgcvfun(lambda, f, Sigma_D, S, U)
% Function of the divide and conquer GCV.
%
% Inputs:    
%   lambda: regularization parameter
%   f: observation, Nx1, N=n_0*J
%   Sigma_D: Gram matrix, NxN
%   S: store the eigenvalues of {Sigma_D_i}_{i=1}^{m};
%      S is lxm, where each Sigma_D_i are lxl, and l*m=N.
%   U: store the eigenvectors of {Sigma_D_i}_{i=1}^{m};
%      U is lxlxm, where lxl store the eigenvectors.
%
% Outputs:
%   G: function value of dgcvfun

% Check for input arguments
[N, ~] = size(f);
if size(Sigma_D,1) ~= N 
    error('Dimension not consistent')
end
[l, m] = size(S);
if l*m ~= N 
    error('Dimension not consistent')
end
if (size(U,1)~=size(U,2)) || (size(U,1)~=l) || (size(U,3)~=m)
    error('Dimension not consistent')
end  

% compute Sigma_lambda * f, and the trace term
val2 = 0;    % value of the trace term
tol = 1e-16;
f1 = zeros(N,1);
for i = 1:m 
    fi = f((i-1)*l+1:i*l);
    Ui = U(:,:,i);
    Si = S(:,i);
    ind = find(Si>tol);
    r = length(ind);
    S1i = zeros(l,1);
    S1i(1:r) = 1 ./ (Si(ind)+lambda);
    S1i(r+1:l) = zeros(l-r,1);

    f1((i-1)*l+1:i*l) = Ui * (diag(S1i)*(Ui'*fi));

    val2 = val2 + sum(Si(ind)./(Si(ind)+lambda));
end

% compute 1/m*Sigma_D*Sigma_lambda*f
f_hat = 1/m * Sigma_D * f1;

% residual norm
val1 = norm(f-f_hat)^2;

% compute the dGCV value
G = val1 / ((N-1/m*val2))^2;

end