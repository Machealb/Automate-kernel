function [reg_corner,rho,eta,reg_param] = l_curve(U,sm,b,method,varargin)
%L_CURVE Plot the L-curve and find its "corner".
% Plots the L-shaped curve of eta, the solution norm || x || or
% semi-norm || L x ||, as a function of rho, the residual norm
% || A x - b ||, for the following methods:
%    method = 'Tikh'  : Tikhonov regularization   (solid line )
% The corresponding reg. parameters are returned in reg_param.  If no
% method is specified then 'Tikh' is default.  For other methods use plot_lc.
%
% Note that 'Tikh', require either U and s (standard-
% form regularization) computed by the function csvd, or U and sm (general-
% form regularization) computed by the function cgsvd, while 'mtvsd'
% requires U and s as well as L and V computed by the function csvd.
%
% If any output arguments are specified, then the corner of the L-curve
% is identified and the corresponding reg. parameter reg_corner is
% returned.  Use routine l_corner if an upper bound on eta is required.

% Reference: P. C. Hansen & D. P. O'Leary, "The use of the L-curve in
% the regularization of discrete ill-posed problems",  SIAM J. Sci.
% Comput. 14 (1993), pp. 1487-1503.

% Per Christian Hansen, DTU Compute, October 27, 2010.

% check inputs
if nargin==5 
  plotOn = varargin{1};
end

% Set defaults.
if (nargin==3), method='Tikh'; end  % Tikhonov reg. is default.
npoints = 200;  % Number of points on the L-curve for Tikh and dsvd.
smin_ratio = 16*eps;  % Smallest regularization parameter.

% Initialization.
[m,n] = size(U); [p,ps] = size(sm);
if (nargout > 0), locate = 1; else locate = 0; end
beta = U'*b; beta2 = norm(b)^2 - norm(beta)^2;
if (ps==1)
  s = sm; beta = beta(1:p);
else
  s = sm(p:-1:1,1)./sm(p:-1:1,2); beta = beta(p:-1:1);
end
xi = beta(1:p)./s;
xi( isinf(xi) ) = 0;

if (strncmp(method,'Tikh',4) | strncmp(method,'tikh',4))

  eta = zeros(npoints,1); rho = eta; reg_param = eta; s2 = s.^2;
  reg_param(npoints) = max([s(p),s(1)*smin_ratio]);
  ratio = (s(1)/reg_param(npoints))^(1/(npoints-1));
  for i=npoints-1:-1:1, reg_param(i) = ratio*reg_param(i+1); end
  for i=1:npoints
    f = s2./(s2 + reg_param(i)^2);
    eta(i) = norm(f.*xi);
    rho(i) = norm((1-f).*beta(1:p));
  end
  if (m > n & beta2 > 0), rho = sqrt(rho.^2 + beta2); end
  marker = '-'; txt = 'Tikh.';
else
  error('Illegal method')
end

% Locate the "corner" of the L-curve, if required.
if (locate)
  [reg_corner,rho_c,eta_c] = l_corner(rho,eta,reg_param,U,sm,b,method);
end

% Make plot.
if nargin==4 || plotOn==1
  plot_lc(rho,eta,marker,ps,reg_param);
  if locate
    ax = axis;
    HoldState = ishold; hold on;
    loglog([min(rho)/100,rho_c],[eta_c,eta_c],':r',...
         [rho_c,rho_c],[min(eta)/100,eta_c],':r')
    title(['L-curve, ',txt,' corner at ',num2str(reg_corner)]);
    axis(ax)
    if (~HoldState), hold off; end
  end
end