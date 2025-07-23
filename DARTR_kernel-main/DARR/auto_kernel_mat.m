function [Gbar_D, basis_D, Sigma_D, varargout] = auto_kernel_mat(regressionData, rkhs_type, varargin) 
% Construct from data the evaluation of kernel, auto-basis functions, and the reproducing kernel matrix.
%{
% Inputs:
%   regressionData: discrtized data from the inverse problem model
     - .g_ukxj: size =  (n_s,J,N);  convl_gu(r,x) = g[u](x,r)  -- used in G, rho.  g_ukxj(s):=g[u_k](x_j,s) in the paper. 
     - .fx_vec:  size = J x N  ;    fx_vec ~ phi_vec * convl_gu * dr  
     - .G:      size = n_s x n_s;  G(r,s)    --- used in vector & function learning
                              G = \sum_{k,j} g_ukxj' * g_ukxj *dx /N    
%   rhks_type: type of rkhs kernel
        'auto': data-adaptive RKHS kernel for auto regularization
        'gauss': Gaussian kernel
%   varargin:
       rkhs_type='gauss', p = varargin(1)---the bandwidth for the Gaussian kernel)
% Outputs:
      Gbar_D:       matrix of evaluations of the kernel on {s_l}---ns x ns
      basis_D:      matrix of values of n0xJ basis functions on {s_l} points--- n0J x ns
      Sigma_D:      kernel matrix via n0J linear functionals---n0J x n0J
      L (optional): g_ukxj/rho_vec; size = (n0J, n_s). Gbar_D = L'*L*dx/(n0) (different by a constant)
%} 
% Check for acceptable number of input arguments

if nargin < 2
    error('Not Enough Inputs')
end

% if strcmp(rkhs_type, 'gauss')
%     l = varargin{1};
% elseif any(strcmp(rkhs_type, {'auto','auto-no-rho'}))
%     % pass
% else
%     error('Wrong RKHS kernel type')
% end
switch rkhs_type
    case 'gauss'
        l = varargin{1};
    case {'auto','auto-no-rho'}
        % pass
    otherwise 
        error('Wrong RKHS kernel type')
end 

r_seq   = regressionData.r_seq;         % when r_seq is non-uniform, use dr = r_seq(2:end) - r_seq(1:end-1).       
dx      = r_seq(2)-r_seq(1); 
ds      = dx;
rho_val = regressionData.rho_val;

g = regressionData.g_ukxj;
[ns, J, n0] = size(g);
k  = n0*J;
g1 = zeros(ns, k);

for i = 1:n0
    g1(:,(i-1)*J+1:i*J) = g(:,:,i);
end

g1 = g1';  % n0J x ns

L = g1 * diag(1./rho_val) / sqrt(n0) * sqrt(dx);   % Gbar_D = L'*L. Here: dx = 1/J. Use dx, not 1/J to be consistent.  

if nargout == 4
    varargout{1} = L;
end

if strcmp(rkhs_type, 'auto')
    G      = regressionData.G;             %   G      = \sum_{k,j} g_ukxj' * g_ukxj *dx /n0  = g1'*g1/n0J if J= 1/dx
    Gbar_D = G./(rho_val*rho_val');        %   Gbar_D = L'*L; 
    % figure; 
    % Gbar_D2 = L'*L;   % or:  L = g1 / sqrt(n0);    Gbar_D2 = (L'*L)./(rho_val*rho_val');  
    % imagesc(Gbar_D-Gbar_D2); colorbar; 
elseif strcmp(rkhs_type, 'auto-no-rho')
    Gbar_D = regressionData.G; 
elseif strcmp(rkhs_type, 'gauss')
    G_fun = @(s1,s2) Gauss(s1, s2, l);
    r_seq = r_seq(:);    % nsx1
    rr1   = r_seq * ones(1,ns);
    rr2   = rr1';
    G_mat = arrayfun(G_fun, rr1(:), rr2(:));
    Gbar_D = reshape(G_mat, ns, ns);   % form G_matrix with evaluations on {s_l}    
end

basis_D = g1*Gbar_D*ds;        % auto-basis from the Representer theorem: (n0J,ns)x(ns,ns)-->(n0J,ns)
Sigma1  = g1 * Gbar_D * g1';   % n0J x n0J 
Sigma_D = Sigma1 * ds^2;       % Sigma_D * c_xi  = fvec    

% 
% Gbar_D  = Gbar_D  + 1e-16*eye(ns);      % TO Haibo 250608: not adding it here. TO add it at the SVD part instead. 
% Sigma_D = Sigma_D + 1e-16*eye(n0J);

end


%--------------------
function val = Gauss(s1, s2, l)
    d = (s1-s2)^2;
    val = exp(-d/(2*l));
end