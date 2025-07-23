function [X, res, eta, iter_stop] = wlsqr(A, rho, b, k, stop_rule)
    % Weighted LSQR for linear discretized I-kind Fredholm integral equation, with form
    %       min||x||_B  s.t. min||Ax-b||_2,
    % where B is the L_{\rho}^2 norm (weights),
    % and B is a diagonal matrix whose elements are the atom of
    % exploration measure \rho, which can be computed as
    %      B(i,i) = rho(i).
    % 
    % Let B = L^2. wlsqr uses the LSQR based on the bidiagonal reduction of the preconditioned
    %    ||b - A*L^{-1}(Lx)||_2
    % by Lanczos bidiagonaliation of AL^{-1}.
    %
    % Inputs:
    %   A: either (a) a full or sparse mxn matrix;
    %             (b) a matrix object that performs the matrix*vector operation
    %   rho: atoms of the exploration measure
    %   b: right-hand side vector
    %   A and b construct the ill-posed linear system: Ax + e = b, where e is the noise
    %   k: the maximum number of iterations  
    %   reorth: 
    %       0: no reorthogonalization
    %       1: full reorthogonaliation, MGS
    %   method: early stopping rule, 'LC' by default
    %
    % Outputs:
    %   X: store the first k regularized solution 
    %   res: store the first k residual norm
    %   eta: store the first k solution L_{\rho}^2 norm
    %   iter_stop：early stopping iteration
    %
    
    % Check for input arguments    
    [m, n] = size(A); 
    if m~= size(b,1)
        error('The dimensions are not consistent')
    end

    reorth = 1;
    
    rho = rho(:);
    rho1 = sqrt(rho);

    tol_alpha_beta = 1e-15; %% set the tolerance of small alpha, beta. Avoid using small eigenvalues's eig-spaces.  
    k_lowbound = 5;         % the lower bound of k dimensions to be computed, so that the L-curve will work.  
    tol_res    = 1e-15; 
    tol_eta    = 1e-10; 


    % declares the matrix size
    Bk = zeros(k+1, k);
    U = zeros(m, k+1);
    V = zeros(n, k);

    X = zeros(n, k);
    res = zeros(k,1);
    eta = zeros(k,1);
    
    % start iteration
    bbeta = norm(b);  beta = bbeta;
    u = b / beta;  U(:,1) = u;
    r = (A' * u) ./ rho1;
    alpha = norm(r);  Bk(1,1) = alpha;
    v = r / alpha;    V(:,1) = v;

    w = V(:,1);
    phi_bar = bbeta;
    rho_bar = Bk(1,1);
    x = zeros(n, 1);

    beta = 1; j=0; 
    % The j-th step gen-GKB iteration and update procedure
    while  j<=k-1 && alpha*beta >0  
        j= j+1; 

        % compute u
        p = A * (v ./ rho1) - alpha * u;
        if reorth == 1  % full reorthogonalization of u
            for i = 1:j
                p = p - U(:,i)*U(:,i)'*p;  % MGS
            end
        end
        beta = norm(p);  Bk(j+1,j) = beta;
        u = p / beta;    U(:,j+1) = u;
        if j>k_lowbound && (isnan(beta) || beta <= tol_alpha_beta); j= j-1;  break; end

        % compute v
        r = (A' * u) ./ rho1 - beta * v;
        if reorth == 1  % full reorthogonalization of v
            for i = 1:j
                r = r - V(:,i)*V(:,i)'*r;  % MGS
            end
        end
        alpha = norm(r);  Bk(j+1,j+1) = alpha;
        v = r / alpha;    V(:,j+1) = v;
        % if j>k_lowbound && (isnan(alpha)|| alpha<= tol_alpha_beta); j= j-1; break;  end

        % Construct and apply orthogonal transformation.
        rrho = sqrt(rho_bar^2 + Bk(j+1,j)^2);
        c = rho_bar / rrho;
        s =  Bk(j+1,j) / rrho;
        theta = s * Bk(j+1,j+1);
        rho_bar = - c* Bk(j+1,j+1);
        phi = c * phi_bar;
        phi_bar = s * phi_bar;

        % Update the solution.
        x = x + (phi/rrho)*w;
        w = V(:,j+1) - (theta/rrho)*w;
        X(:,j) = x ./ rho1;
        res(j) = abs(phi_bar);
        eta(j) = sqrt(x' * X(:,j));
        % if (j>k_lowbound) && (res(j)< tol_res || eta(j)-eta(j-1)<tol_eta); break;  end
    end
    
n_terminate = j; 
X   = X(:,1:n_terminate); 
res = res(1:n_terminate); 
eta = eta(1:n_terminate);   

if strcmp(stop_rule, 'LC')
    if length(res) < 3
        iter_stop = length(res);
    else    
        [iter_stop, ~] = Lcurve_corner(res, eta, 0);
    end
    if iter_stop < 1
        iter_stop = 1;
    end
    fprintf('[LC is satisfied], k_LC=%d\n', iter_stop);
else 
    error('No such early stopping rule')
end


end
    
