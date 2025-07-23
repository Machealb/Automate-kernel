function [X, res, eta, iter_stop] = iter_regu(A, D, b, k, nsr, NoStop)
% iterative data-adaptive RKHS regularization with data-adaptive 
% basis functions based on gGKB.
% Use discrepancy principle (DP) or L-curve as the early stopping rule.

    iter_stop = 0;  % initial stopping iteration
    flag = false;   % indicate wether we have find an early stopping iteration 
    % whether or not providing noise level
    if nsr == 0   
        NoStop = 'on';  % no noise level provided--the iteration should run to complete and then use L-curve 
    elseif nsr > 0
        DP_tol = 1.001 * nsr;    % 1.01*noise norm
    end

    D = D(:);
    rho = sqrt(D);
    b = b(:);
    
    [m, n] = size(A);
    A = A + 1e-13*eye(n);

    % fprintf('[Start gGKB...], max_Iter=%d\n', k);
    if strcmp(NoStop, 'on')
        [~, Z, Z1, B, bbeta] = gGKB(A, D, b, k, 1);
        k1 = size(Z, 2)-1;
        X = zeros(n, k1);
        res = zeros(k1,1);
        eta = zeros(k1,1);

        phi_bar = bbeta;
        rho_bar = B(1,1);
        x = zeros(n,1);
        w = Z(:,1);
        x_bar  = zeros(n,1);  % x_bar = Ax, assisting to compute ||x||_A
        w_bar  = Z1(:,1);     % w_bar = Aw, used to iteratively update x_bar

        % update solution procedure
        for l = 1:k1
            % fprintf('[iDARR iterating...], step=%d\n', l);
            %  Construct and apply orthogonal transformation.
            rrho = sqrt(rho_bar^2 + B(l+1,l)^2);
            c = rho_bar / rrho;
            s =  B(l+1,l) / rrho;
            theta = s * B(l+1,l+1);
            rho_bar = -c * B(l+1,l+1);
            phi = c * phi_bar;
            phi_bar = s * phi_bar;

            %  Update the solution, solution RKHS-norm and residual 2-norm
            x = x + (phi/rrho) * w;
            w = Z(:,l+1) - (theta/rrho) * w;
            X(:,l) = x;
            res(l,1) = abs(phi_bar);                      % residual 2-norm
            x_bar = x_bar + (phi/rrho) * w_bar;           % A*x
            w_bar = Z1(:,l+1) - (theta/rrho) * w_bar;     % A*w
            eta(l,1) = sqrt(x'*x_bar);                    % solution RKHS-norm (under DAR basis)
            
            %  determine early stopping iteration by DP
            if nsr>0 && flag==false && abs(phi_bar)<=DP_tol
                iter_stop = l;
                flag = true;
            end
        end

    elseif strcmp(NoStop, 'off') 
        reorth = 1;
        % declares the matrix size
        B = zeros(k+1, k+1);
        U = zeros(m, k+1);
        Z = zeros(n, k+1);
        Z1 = zeros(n, k+1);  % Z1=AZ
        X = zeros(n, k);
        res = zeros(k,1);
        eta = zeros(k,1);

        %  initial step of gGKB
        b1 = rho .* b;
        bbeta = norm(b1);
        u = b1 / bbeta;  
        U(:,1) = u;
        r = rho .* u;
        alpha = sqrt(r'*A*r);
        z = r / alpha;
        Z(:,1)  = z;
        z1 = A*z;
        Z1(:,1) = z1;
        B(1,1)  = alpha;

        %  Prepare for update procedure
        phi_bar = bbeta;
        rho_bar = alpha;
        x = zeros(n,1);
        w = z;
        x_bar  = zeros(n,1);
        w_bar = z1 ;

        % update solution procedure
        for j = 1:k
            % fprintf('[iDARR iterating...], step=%d\n', j);
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
                fprintf('[Breakdown...], beta=%f, gGKB breakdown at %d\n', [beta,j]);
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
                break;
            end

            z = r / alpha;
            Z(:,j+1) = z;
            Z1(:,j+1) = A*z;
            z1 = A * z;
            B(j+1,j+1) = alpha;
           
            % Construct and apply orthogonal transformation
            rrho = sqrt(rho_bar^2 + beta^2); 
            c1 = rho_bar / rrho;
            s1 = beta / rrho; 
            theta = s1 * alpha; 
            rho_bar = -c1 * alpha;
            phi = c1 * phi_bar;
            phi_bar = s1 * phi_bar;

            % Update the solution, solution RKHS-norm and residual 2-norm
            x = x + (phi/rrho) * w;  
            w = z - (theta/rrho) * w;
            X(:,j) = x;
            res(j,1) = abs(phi_bar); 
            x_bar = x_bar + (phi/rrho) * w_bar;
            w_bar = z1 - (theta/rrho) * w_bar;
            eta(j,1) = sqrt(x'*x_bar);

            % determine early stopping iteration by DP
            if nsr>0 && flag==false && abs(phi_bar)<=DP_tol
                iter_stop = j;
                flag = true;
            end
            
            % end the iteration if DP is satisfied
            if flag == true
                X = X(:,1:iter_stop);
                res = res(1:iter_stop);
                eta = eta(1:iter_stop);
                fprintf('[DP is satisfied], k_DP=%d\n', iter_stop);
                break
            end
        end
    end

    % Estimate early stopping iteration by L-curve
    if nsr == 0 
        if length(res) < 3
            iter_stop = length(res);
        else    
            [iter_stop, ~] = Lcurve_corner(res, eta, 0);
        end
        if iter_stop < 1
            iter_stop = 1;
        end
        fprintf('[LC is satisfied], k_LC=%d\n', iter_stop);
    end

end
