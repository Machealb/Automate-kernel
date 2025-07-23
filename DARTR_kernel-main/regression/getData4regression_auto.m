function regressionData = getData4regression_auto(ux_val,fx_val,dx,obsInfo,bdry_width,Ind_x_Loss,r_seq,data_str, normalizeOn)
%  % pre-processData for regression: 
%  %           min |L_phi[u]- f|^2 = c'*Abar*c - 2c'*b + bnorm_sq  
%{ 
 L_phi[u](x) = \int phi(|y|)*g0[u](x,y) dy 
             = \sum_r phi(r)* convl_gu(x,r) dr   
    fn_val   = phi_vec * convl_gu                   % Riemann sum.       
 1. g[u](x,r) = u(x+r)                                           linearIntOpt: integral kernel: 1D, Laplace Transform 
 2. g[u](x,y) = u(x+r) + u(x-r) - 2*u(x);                        nonlocal kernel
 3. g[u](x,y) = [Du(x-r)-Du(x+r)]·u(x) + [u(x-r)-u(x+r)]·Du(x)   Aggregation_StrForm
---- convl_gu(r,x) = g[u](x,r) 

====== In the inverse-integral-operator
  <L_phi[u], L_psi[u]> = \int_x L_phi[u](x) L_psi[u](x) dx
                       = \sum_{r,s} phi(r)psi(s) \int_x convl_gu(r,x)convl_gu(s,x)dx drds
         convl_gu(r,x) = g[u](x,r)          
                G(r,s) = \int_x convl_gu(r,x)convl_gu(s,x)dx    *****
 - for function learning: we need the
                A(i,j) = <L_{phi_i}[u], L_{phi_j}[u]> = sum_{r,s} phi_i(r)phi_j(s) G(r,s) drds
                b(i)   = <L_{phi_i}[u], f> = int_{r} phi_i(r) sum_x convl_gu(r,x)f(x)dx  dr
                       >>>>>                                    *** convl_gu_f(r)   *****
 
 - for vector learning: we only need [with phi_i = phi(r_i) ]
                A(i,j) = G(r_i,r_j) *dr*dr
                b(i)   = G*phi  = convl_gu_f(r_i) *dr
== Output: ===>>>  ***** in either case, we only need those *****
regressionData.
- .g_ukxj: size =  (n_r,J,N);  convl_gu(r,x) = g[u](x,r)  -- used in G, rho.  g_ukxj(s):=g[u_k](x_j,s) in the paper. 
- .fx_vec:  size = J x N  ;    fx_vec ~ phi_vec * convl_gu * dr  

- .G:      size = n_r x n_r;  G(r,s)    --- used in vector & function learning
                              G = \sum_{k,j} g_ukxj' * g_ukxj *dx /N    
- .rho:    size = n_r x 1 ;   rho(r)    --- used in all:  
      exploration measure rho(r) = \int |convl_gu(r,x)|^p dx,  p = 0,1,2 
- .gu_f:    size = n_r x 1;      gu_f ~ \sum_{k} convl_gu_k*fx_vec_k * dx /N 
%}


%% %% 1 load and normalize process data (u,f)
if numel(size(ux_val ) ) == 3               % u(n,t,x) 
    [case_num, T, u_x_num] = size(ux_val);   
    N           = case_num*T;
    ux_val      = reshape(ux_val, [case_num*T, u_x_num]);    
    fx_val      = reshape(fx_val, case_num*T,[]);
elseif numel(size(ux_val)) == 2           % u(n,x) 
    [case_num, u_x_num] = size(ux_val);     N = case_num; 
end

% Approximate Du by central difference  
du = zeros(size(ux_val)); 
switch obsInfo.example_type 
    case 'Aggregation_StrForm'  
        method   = 'filterDiff'; % 'centralDiff';  filterDiff
        dx       = obsInfo.x_mesh_dx;
        [du,~]   = compute_derivatives(ux_val,dx,method,1);  
        % % substract ddu from fx_val ---- the mean-field equation
        % fx_val  = fx_val-ddu;      
        use_u_derivative = 1;
        if use_u_derivative==1
            n0 = length(obsInfo.div_u);
            du = zeros(n0,length(obsInfo.u_xmesh));
            for k = 1:n0
                du(k,:) = obsInfo.div_u{k}(obsInfo.u_xmesh);
            end
        end
end


% normalize u by L2 norm if normalizeOn 
if normalizeOn 
    [ux_val, fx_val] = normalize_byL2norm(ux_val, fx_val, dx); 
    data_str         = [data_str,'NormalizeL2']; 
    fprintf('\n L2-Normalized u and f: for linear operators only \n');
end
N_xi_loss  = u_x_num-2*bdry_width;
N_r_seq    = length(r_seq);     % r_seq       = dx*(1:bdry_width);
if (length(Ind_x_Loss) ~= N_xi_loss || (N_r_seq ~= bdry_width && bdry_width >0)) 
    error('Index_xi_inUse does not match with data and bdry_width.  \n'); % error and terminate 
end
%% 2. get data for  regression:   
%  get data for regression: convl_gu(r,x) = g[u](x,x+r) + g[u](x,x-r); 
    fun_g_vec = obsInfo.fun_g_vec;
%    ind_p     = 1:bdry_width;   ind_m = -(1:bdry_width);       % Index plus r  
    rInd       = 1:bdry_width; 
    g_ukxj   = zeros(N_r_seq,N_xi_loss,N);     % the array: g_kjl = g[u_k](x_j,r_l)
    rhoN     = zeros(N_r_seq,N);                % N copies: exploration measure rho: rho(r) = \int   |convl_gu(r,x)| dx
    GN       = zeros(N_r_seq,N_r_seq,N);        % N copies: G(r,s) = \int_x convl_gu(r,x)convl_gu(s,x)dx 
    gu_fN    = zeros(N_r_seq,N);                % N copies: sum_x convl_gu(r,x)f(x)dx
    gu_fN2   = gu_fN;                           % a downsampled estimator of gu_fN 
    bnorm_sq = 0; 
%     fx_vec   = zeros(N,N_xi_loss);             % right-hand side of f_i with values on Index_xi_inUse
    fx_vec   = fx_val; 
    for nn=1:N                                  % compute these terms for each u(x) 
         u1           = ux_val(nn,:); 
         f1           = fx_val(nn,:)';          % fx_val' ~ phi*g_ukxj,   size = N x J

         du1 = du(nn,:);
         convl_gu = zeros(N_r_seq,N_xi_loss);  val_abs = convl_gu;  
         % convl_gu2 = convl_gu;   % old version that does not taking into radial in g[u]
         for k  = 1:N_xi_loss
             temp  = fun_g_vec(u1,du1,k+bdry_width,rInd); 
             convl_gu(:,k) = temp'; 
             val_abs(:,k)  = abs(convl_gu(:,k));  
             % temp_p        = fun_g_vec(u1,du1, k+bdry_width, rInd);
             % temp_m        = fun_g_vec(u1,du1, k+bdry_width, -rInd);
             % convl_gu2(:,k) = (temp_p + temp_m)';
         end
         % [rank(convl_gu), rank(convl_gu2)] 
         if nn<4 && obsInfo.plotON ==1
             figure(121); clf;  % plot one of u and g[u](x,r)
             subplot(121); plot(obsInfo.u_xmesh,u1);
             subplot(122); plot(convl_gu); title(['[g[u_i](x,r)], Rank = ',num2str(rank(convl_gu)) ] );
             pause(1)
         end

         g_ukxj(:,:,nn) = convl_gu;                % n_s x J 
         rhoN(:,nn)     = mean(val_abs,2); 
         GN(:,:,nn)     = convl_gu* convl_gu';     % later: G= mean(GN,3)*dx;      
         gu_fN(:,nn)    = convl_gu*f1;   bnorm_sq = bnorm_sq+ sum(f1.^2)*dx; 
         gu_fN2(:,nn)   = convl_gu(:,1:2:end)*f1(1:2:end); 
    end 

    rho_val  = mean(rhoN,2);   
    rho_tol= 1e-8; 
    if any(rho_val<rho_tol) 
         rho_val = rho_val + rho_tol;                        % should remove small weighted points. Keep all to compare rho1,rho2's.  
         fprintf('\n rho value < 1e-8. Added 1e-8. \n '); 
    end 
    indr = find(rho_val>rho_tol);  rho_val = rho_val(indr);  % remove those zero weight points 
    r_seq   = r_seq(indr);    
    G       = mean(GN,3)*dx;       regressionData.G    = G(indr,indr);   % r.k. G. Note: Gbar= G./(rho_val*rho_val')    *dr*ds,
    gu_f    = mean(gu_fN,2)*dx;    regressionData.gu_f = gu_f(indr);  % b                                      *dr
    gu_f2   = mean(gu_fN2,2)*dx;   regressionData.gu_f2= gu_f2(indr); % b2 

    regressionData.rho_val0  = (1+ 0*rho_val)/sum(1+0*rho_val)/dx; % exploration measure: uniform 
    regressionData.rho_val1  = rho_val/sum(rho_val)/dx;            % exploration measure: L1 in x 
   
    rho_val2  = diag(G); 
    if any(rho_val2<rho_tol)                                % should remove small weighted points. 
        rho_val2  = rho_val2+ rho_tol+1e-14; 
        fprintf('\n rho2 value < 1e-8. Added 1e-8. \n '); 
    end 
    indr = find(rho_val2>rho_tol);  
    rho_val2  = rho_val2(indr);                                    % remove those zero weight points 
    regressionData.rho_val2  = rho_val2/sum( rho_val2)/dx;         % exploration measure: L2 in x
    
    % regressionData.g_ukxj     = g_ukxj; 
    regressionData.g_ukxj     = g_ukxj(indr,:,:);    % remove those values of g with zero weight points in all rows. 
	regressionData.bnorm_sq   = bnorm_sq/N;
	regressionData.bdry_width = bdry_width; 
	regressionData.r_seq      = r_seq;  
	regressionData.data_str   = data_str;

    regressionData.fx_vec   = fx_vec;                % noisy right-hand side ---- normalizeOn makes u with normal in L2

    if ~isfield(obsInfo, 'plotON') 
        obsInfo.plotON = 1;
    end
    if obsInfo.plotON ==1
        figure(122); clf; % plot
        subplot(121);
        plot(r_seq,regressionData.rho_val0); hold on;
        plot(r_seq,regressionData.rho_val1,':','Linewidth', 2);
        plot(r_seq,regressionData.rho_val2,'--','Linewidth', 2);  title('Exploration measures \rho');
        xlabel('s'); ylabel('\rho'); legend('Uniform','rhoL1','rhoL2')

        subplot(122); imagesc(G); title(['matrix G, Rank = ',num2str(rank(G)) ] ); pause(2);
    end 
end
