function [reg_min,G,reg_param] = gcv(U,s,b,method,varargin)
%GCV Plot the GCV function and find its minimum.
%
% Plots the GCV-function
%          || A*x - b ||^2
%    G = -------------------
%        (trace(I - A*A_I)^2
% as a function of the regularization parameter reg_param. Here, A_I is a
% matrix which produces the regularized solution.
%
% The following methods are allowed:
%    method = 'Tikh' : Tikhonov regularization   (solid line )
%
% If any output arguments are specified, then the minimum of G is
% identified and the corresponding reg. parameter reg_min is returned.

% Per Christian Hansen, DTU Compute, Dec. 16, 2003.

% Reference: G. Wahba, "Spline Models for Observational Data",
% SIAM, 1990.

% check inputs
if nargin==5 
  plotOn = varargin{1};
end

% Set defaults.
if (nargin==3), method='Tikh'; end  % Default method.
npoints = 200;                      % Number of points on the curve.
smin_ratio = 16*eps;                % Smallest regularization parameter.

% Initialization.
[m,n] = size(U); [p,ps] = size(s);
beta = U'*b; beta2 = norm(b)^2 - norm(beta)^2;
if (ps==2)
  s = s(p:-1:1,1)./s(p:-1:1,2); beta = beta(p:-1:1);
end
if (nargout > 0), find_min = 1; else find_min = 0; end

if (strncmp(method,'Tikh',4) || strncmp(method,'tikh',4))

  % Vector of regularization parameters.
  reg_param = zeros(npoints,1); G = reg_param; s2 = s.^2;
  reg_param(npoints) = max([s(p),s(1)*smin_ratio]);
  ratio = (s(1)/reg_param(npoints))^(1/(npoints-1));
  for i=npoints-1:-1:1, reg_param(i) = ratio*reg_param(i+1); end

  % Intrinsic residual.
  delta0 = 0;
  if (m > n & beta2 > 0), delta0 = beta2; end

  % Vector of GCV-function values.
  for i=1:npoints
    G(i) = gcvfun(reg_param(i),s2,beta(1:p),delta0,m-n);
  end 

  % Find minimum, if requested.
  if (find_min)
    [minG,minGi] = min(G); % Initial guess.
    reg_min = fminbnd('gcvfun',...
      reg_param(min(minGi+1,npoints)),reg_param(max(minGi-1,1)),...
      optimset('Display','off'),s2,beta(1:p),delta0,m-n); % Minimizer.
    minG = gcvfun(reg_min,s2,beta(1:p),delta0,m-n); % Minimum of GCV function.
  end

  % Plot GCV function.
  if nargin==4 || plotOn==1
    figure;
    loglog(reg_param,G,'-'), xlabel('\lambda'), ylabel('G(\lambda)')
    title('GCV function')
    ax = axis;
    HoldState = ishold; hold on;
    loglog(reg_min,minG,'*r',[reg_min,reg_min],[minG/1000,minG],':r')
    title(['GCV function, minimum at \lambda = ',num2str(reg_min)])
    axis(ax)
    if (~HoldState), hold off; end
  end
else
  error('Illegal method')
end

end