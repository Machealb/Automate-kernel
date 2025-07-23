
function [U, Z, Z1, B, bbeta] = gGKB(A, D, b, k, reorth)
% generalized Golub-Kahan bidiagonalizition (gGKB) of the linear operator:
% D^{1/2}A: (R^n, <.,.>_A) ---> (R^2, <.,.>_2),  x |--> Ax ,
% where A is symmetric positive definite, and D is diagonal with positive elements.
% D is the covariance of the (uncorelated) noise

    D = D(:);
    rho = sqrt(D);
    b = b(:);
    [m, n] = size(A);

    % declares the matrix size
    % fprintf('[Start gGKB...], max_Iter=%d, reorth=%d\n', [k,reorth]);
    B  = zeros(k+1, k+1);
    U  = zeros(m, k+1);
    Z  = zeros(n, k+1);
    Z1 = zeros(n, k+1);  % Z1=AZ

    % initial step of gGKB
    b1 = rho .* b;
    bbeta = norm(b1);
    u = b1 / bbeta;  
    U(:,1) = u;
    r = rho .* u;
    alpha = sqrt(r'*A*r);
    z = r / alpha;
    Z(:,1)  = z;
    Z1(:,1) = A*z;
    B(1,1)  = alpha;

    % k step iteration of gGKB
    for j = 1:k
        % fprintf('[gGKB iterating...], step=%d--------\n', j);
        % compute u in 2-inner product
        s = A * z - alpha * u;
        if reorth == 1
            for i = 1:j
                s = s - U(:,i)*(U(:,i)'*s);  % MGS
            end
        elseif reorth == 2
            for i = 1:j
                s = s - U(:,i)*(U(:,i)'*s);  % MGS
            end
            for i = 1:j
                s = s - U(:,i)*(U(:,i)'*s);  % MGS
            end
        end

        beta = norm(s);
        if beta < 1e-14
            fprintf('[Breakdown...], beta=%f, gGKB breakdown at %d--\n', [beta,j]);
            U = U(:,1:j);
            Z = Z(:,1:j);
            Z1 = Z1(:,1:j);
            B = B(1:j,1:j);
            break;
        end

        u = s / beta;
        U(:,j+1) = u;
        B(j+1,j) = beta;

        % compute z in A-inner product
        r = rho .* u - beta * z;
        if reorth == 1
            for i = 1:j
                r = r - Z(:,i)*(Z1(:,i)'*r);
            end
        elseif reorth == 2
            for i = 1:j
                r = r - Z(:,i)*(Z1(:,i)'*r);
            end
            for i = 1:j
                r = r - Z(:,i)*(Z1(:,i)'*r);
            end
        end
    
        alpha = sqrt(r'*A*r);
        if alpha < 1e-14
            fprintf('[Breakdown...], alpha=%f, gGKB breakdown at %d--\n', [alpha,j]);
            U = U(:,1:j+1);
            Z = Z(:,1:j);
            Z1 = Z1(:,1:j);
            B = B(1:j+1,1:j);
            break;
        end

        z = r / alpha;
        Z(:,j+1) = z;
        Z1(:,j+1) = A*z;
        B(j+1,j+1) = alpha;
    end

end
