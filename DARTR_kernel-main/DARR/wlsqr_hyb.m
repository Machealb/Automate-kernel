function [X, res, Lam, GCV, iterstop] = wlsqr_hyb(A, rho, b, k, degflat, adaptW, plotOn)
% Hybrid iterative method for L^2(rho)-regularization, with form
%       min||x||_B  s.t. min||Ax-b||_2,
% where B is the L_{\rho}^2 norm (weights),
% and B is a diagonal matrix whose elements are the atom of
% exploration measure \rho, which can be computed as
%      B(i,i) = rho(i)
%
% Inputs:
%   A: either (a). a full or sparse mxn matrix;
%             (b). a matrix object that performs the matrix*vector operation
%   rho: atoms of the exploration measure
%   b: right-hand side vector
%   k: the maximum number of iterations 
%   adaptW:
%     0: not adapt weight parameters w, use GCV method
%     1: adapt weight parameters w, use WGCV method
%   degflat: tolerance for estimate WGCV stopping point
%
% Outputs: 
%   X: store the first k regularized solutions
%   res: strore residual norm of the first k regularized solution
%   Lam: stores the first k regularization parameters
%   GCV - store values of the GCV function, i.e. WGCV with w=1.
%   iterstop: the stopping iteration estimated by WGCV
%
% Initializationr
if nargin < 6
  error('Not Enough Inputs')
end

[m, n] = size(A);
reorth = 1;
rho = rho(:);
rho1 = sqrt(rho);

X = zeros(n, k); 
res = zeros(k, 1); 
Lam = zeros(k, 1);  
GCV = zeros(k, 1);    % store values of WGCV functions
omega = zeros(k, 1);  % store auxiliary weight parameters

iterstop = 0;         % initialize the early stopping iteration
terminate = 1;        % indicate wether we still need to estimate iterstopb
% adaptW = 1;  
% degflat = 1e-6;    % tolerance for estimate WGCV stopping point
warning = 1;       % avoid possible semi-convergence of GCV
step1 = 3;         % should not big than than step2
step2 = 4;

% fprintf('Start the iter-hybrid iteration =====================\n');
B = zeros(k+1, k+1);
U = zeros(m, k+1);
V = zeros(n, k+1);

% start iteration
bbeta = norm(b);  beta = bbeta;
u = b / beta;  U(:,1) = u;
r = (A' * u) ./ rho1;
alpha = norm(r);  B(1,1) = alpha;
v = r / alpha;    V(:,1) = v;


% GKB ieration and WCGV hybrid  regularization
if plotOn > 0
  h = waitbar(0, 'Beginning hybrid iterations: please wait ...');
end
for j = 1:k 
    % fprintf('Running WGCV hybrid regularizing iteration: the %d-th step -------\n', i);

    %%%----------- GKB to generate U, Z, B ------------------
    p = A * (v ./ rho1) - alpha * u;
    if reorth == 1  
      for i = 1:j
        p = p - U(:,i)*U(:,i)'*p;  % MGS
      end
    end

    beta = norm(p);  B(j+1,j) = beta;
    if beta < 1e-14
        fprintf('[Breakdown...], beta=%f, gGKB breakdown at %d\n', [beta,j]);
        break;
    end

    u = p / beta;    U(:,j+1) = u;

    % compute v
    r = (A' * u) ./ rho1 - beta * v;
    if reorth == 1  
      for i = 1:j
        r = r - V(:,i)*V(:,i)'*r;  % MGS
      end
    end

    alpha = norm(r);  B(j+1,j+1) = alpha;
    if alpha < 1e-14
        fprintf('[Breakdown...], alpha=%f, gGKB breakdown at %d--\n', [alpha,j]);
        break;
    end

    v = r / alpha;    V(:,j+1) = v;


  %%% --------------- GCV hybrid regularization  --------------------------
  Zk = V(:,1:j);
  Bk = B(1:j+1, 1:j); 
  Ck = eye(j);
  vector = [bbeta; zeros(j,1)];

  [Ub, ss, Xb] = gsvd1(Bk, Ck);

  if adaptW  % Use the adaptive weighted GCV method
    omega(j)= min(1.0, findomega(Ub, vector, ss));
    om = mean(omega(1:j));
    lambda = wgcv(Ub, ss, vector, om, plotOn);  % figure(1024) in wgcv.m
  else
    lambda = wgcv(Ub, ss, vector, 1, plotOn);  % w=1.0, i.e. the standard GCV method
  end
  % Solve the projected problem with Tikhonov regularization
  f = tikhonov(Ub, ss, Xb, vector, lambda);
  x = Zk * f;
  X(:,j) = x ./ rho1;
  Lam(j) = lambda^2;  % regularization parameter of the projected problem
  res(j) = norm(Bk*f-vector);
  GCV(j) = GCVstopfun(lambda, Ub, ss, vector);

  % use the GCV value to find the stopping point
  if terminate && warning && j > step1
    l = j - step1;
    if GCV(l) < min(GCV(l+1:j)) 
      iterstop = l-1;  % stop at the minimum GCV to avoid possible semi-convergence   
      fprintf('********* WGCV jump 1! k=%d********** \n', j);
      terminate = 0;
    end
  end
  % If GCV curve is flat, stop and avoid bumps in the GCV curve by using a window of step2+1 iterations 
  if terminate && j > (step2+1)
    if abs((GCV(j)-GCV(j-1)))/GCV(1) <= degflat
      flag = 0;
      for i = 0:step2
        if abs((GCV(j-i)-GCV(j-i-1)))/GCV(1) > degflat 
          flag = flag + 1;
        end
      end
      if flag == 0
        %iterstop = i;
        [~, iterstop1] = min(GCV(j-step2:j));
        iterstop = iterstop1 + j -step2 -1;
        if iterstop == j
            fprintf('********* WGCV decreases flat, k=%d********** \n', j);
        else
            fprintf('********* WGCV jump 2! k=%d********** \n', j);
        end
        terminate = 0;
        iterstop = iterstop - 1;
      end
    end
  end
  if plotOn > 0
    waitbar(j/k, h)
  end
end
if plotOn > 0
  close(h);
end

% check if the stopping iteration is satisfied
if terminate == 1
  iterstop = k;
  fprintf('The WGCV method has not been stabalized. \n');
end

end


% ----------------------- SUBFUNCTION -----------------------
function omega = findomega(U, b, s)
%  This function computes a value for the omega parameter.
%  The method assumes the 'optimal' regularization parameter to be the
%  smallest (generalized) singular value.  
%  Then we take the derivative of the GCV function with respect to alpha, 
%  evaluate it at alpha_opt, set the derivative equal to zero and then solve for omega.
%
%  First assume the 'optimal' regularization parameter to be the 
%  smallest singular value.
%
%  Inputs:  bhat - vector U'*b, where U = left (generalized) singular vectors
%           s - vector containing the (generalized) singular values
%
%  Outputs: omega - computed value for the omega parameter
%

[m, n] = size(U);
bhat = U' * b;
if m > n
  delta0 = norm(b)^2 - norm(bhat)^2;  % ||(I-U*U')b||^2
else
  delta0 = 0;
end

[p, ps] = size(s);  % for the SVD case: p=n, ps=1
if ps==2
  s = s(p:-1:1, 1) ./ s(p:-1:1, 2);  % generalized singular values in decreasing order
  bhat = bhat(p:-1:1);
end
alpha1 = s(p);  % suppose it is the optimal regularization parameter

% compute needed elements for derivative of the WGCV function
s2 = abs(s) .^ 2;
alpha2 = alpha1^2;

tt = 1.0 ./ (s2 + alpha2);

t1 = sum(s2 .* tt) + n - p;
t2 = abs(bhat(1:p).*alpha1.*s) .^2;
t3 = sum(t2 .* abs((tt.^3)));

t4 = sum((s.*tt) .^2);
t5 = sum((abs(alpha2.*bhat(1:p).*tt)).^2);

v1 = abs(bhat(1:p).*s).^2;
v2 = sum(v1.* abs((tt.^3)));

% compute omega by letting derivative of WGCV to be zero
omega = (m*alpha2*v2) / (t1*t3 + t4*(t5 + delta0));
end


% --------------- SUBFUNCTION ---------------------------------------
function G = GCVstopfun(alpha1, U, s, b)
%  This function evaluates the GCV function G(1, alpha1), that will be used
%  to determine the early stopping iteration
%
[m,n] = size(U); 
[p,ps] = size(s);
beta = U'*b; 
beta2 = norm(b)^2 - norm(beta)^2;  % ||(I-U*U')b||^2
if ps==2
  s = s(p:-1:1, 1) ./ s(p:-1:1, 2); 
  beta = beta(p:-1:1);
end

s2 = s.^2;

% Intrinsic residual, ||(I-U*U')b||^2
delta0 = 0;
if m > n && beta2 > 0 
  delta0 = beta2; 
end

G = wgcvfun(alpha1, s2, beta(1:p), delta0,m, n, 1);
end