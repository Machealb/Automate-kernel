function  [kernelInfo, obsInfo] = load_settings_v2(N, u_Type,jump_disc, u_supp, R0, dx, kernel_type, example_type)
%{
Detailed settings for all computations: outputs are function handles.
- 1. Set example types.
- 2. Set kernel types.
- 3. Set u types.
- 4. x-mesh and effective x-mesh from data.
%}

%{
Comparison with previous version: 
0. Change operator setting the Fredholm format: 
        R_phi[u](x) = \int phi(r) g[u](x,r)dr
   + The previous version use \int phi(|y|) fn_gu(x,y)dy and put the
   transformation later. The new setting is clearer and more unified. 
1. Removing Adaptive interaction range adjustment. 
   + The previous version adaptively adjust the interaction range through the
   difference between supp(f) and u_supp. (1) This tech does not work for
   nonlocal operators. (2) The exploration measure will also reflect interaction range in the data. 
2. set x-mesh to be x-domain of the output f that can be used for loss
function, and use u_xmesh for the mesh of u. 
   + The previous version x-mesh is on the domain of u, and use
   zero-padding for f to make the two on the same x-mesh. 
 3. 
%}

jump_discontinuous = jump_disc; % jump-discontinuous u-data in Fourier. 
 
%% 0: operator and mesh setting 
xDim = 1;       % Dimension
%{
         R_phi[u](x)  = \sum_r phi(r) * fun_g[u](x,r) dr,   
At observation mesh points: 
               f_vec  =  g_array * phi_vec,        for each u 
     size:    n_o x 1     n_o x n_r    n_r x 1              nx x 1 

  - Key: r ∈ [0,R0], x in [a,b] such that [a-R0,b+R0] \subset supp(u). 
         so that the above integral can be computed from data u for Nonlocal and Aggregation operators. 
  - u_xmesh: on [a-R0,b+R0] = support(u). 
  - The interaction range R0 is given, but the data may explore only  part of [0,R0].   
  - The function fun_g[u](x,r) encodes the fact that the kernel is radial.
%}

% Ensure that R0 (the preset interaction range, corresponding to δ in the note)
% is provided; otherwise, set a default.
if ~exist('R0','var')
    R0 = 3;  % Default value if not provided.
end
% observation x-mesh in [xmin,xmax] such that [a-R0,b+R0] \subset supp(u). Set xmin = u_supp1 +R0; xmax= u_supp2-R0;  
if u_supp(2) - u_supp(1) < 2*R0+ dx     % reset u_supp if it is too small 
    fprintf('\n Error: Support(u)= [lb,ub] should have ub-lb > 2* interaction range! \n')
    fprintf('Set Support(u) to be [lb-R0,ub+R0]. \n')
    u_supp = [u_supp(1)-R0, u_supp(2)+R0];  % reset u_supp 
end 
xmin = u_supp(1)+R0;    % the [xmin,xmax] is for f(x), NOT u, to save computational cost when generating data. 
xmax = u_supp(2)-R0;  

u_xmesh    = u_supp(1):dx:u_supp(2); 
bdry_width = floor((R0+1e-10)/dx); 
x_mesh     = u_xmesh((1+bdry_width): (end-bdry_width)); 

%% 1. Set example types; define the function in convolution with RADIAL phi

switch example_type     % Return the first part of the string using the '-' character as a delimiter.   
    case 'LinearIntOpt'  
        % 1. g[u](x,r) =  u(x+r)          integral kernel: 1D, 
        %       R_\phi[u](x) = \int_0^{R0} \phi(r) u(x-r) dr.   x \in Dom(u)_R0 
        fun_g     = @(u,Du,x,y) u(x-y); 
        fun_g_vec = @(u,Du,xInd,rInd) u(xInd - rInd);

    case 'nonlocal'
        % 2. fun_g[u](x,r) = u(x+r) + u(x-r) - 2*u(x); nonlocal kernel: 1D.
        fun_g     = @(u,Du,x,y) u(x+y) + u(x-y) - 2*u(x);
        fun_g_vec = @(u,Du,xInd,rInd) u(xInd + rInd) + u(xInd - rInd) - 2*u(xInd);

    case 'Aggregation_StrForm'
        % 3. Aggregation operator in strong form:
        %    fun_g[u](x,r) = [Du(x-r)-Du(x+r)]·u(x) + [u(x-r)-u(x+r)]·Du(x)
        fun_g     = @(u,Du,x,y) (Du(x-y) - Du(x+y)).*u(x) + (u(x-y) - u(x+y)).*Du(x);
        fun_g_vec = @(u,Du,xInd,rInd) (Du(xInd - rInd) - Du(xInd + rInd)).*u(xInd) + (u(xInd - rInd) - u(xInd + rInd)).*Du(xInd);

    case 'Aggregation-weak'  % TBD 
        % 4. Aggregation operator in weak form:
        %    fun_g[u](x,r) = sign(r)·(Du(x-r).*u(x) + u(x-r).*Du(x))
        fun_g     = @(u,Du,x,y) sign(y).*( Du(x-y).*u(x) + u(x-y).*Du(x) );
        fun_g_vec = @(u,Du,xInd,rInd) sign(rInd).*( Du(xInd - rInd).*u(xInd) + u(xInd - rInd).*Du(xInd) );
end
obsInfo.example_type = example_type;
obsInfo.fun_g        = fun_g;
obsInfo.fun_g_vec    = fun_g_vec;

%% 2. True kernel K: [0,R0] --> ℝ (we now consider only radial kernels)
%{
    The interaction range of the kernel is set to be [0, R0]. The true kernel is
    now defined with an explicit indicator so that it vanishes outside [0, R0].
%}
kernelInfo.d = xDim; 
if ~exist('kernel_type','var')
    kernel_type = 'Gaussian';
end

% Save the interaction range in kernelInfo:
kernelInfo.R0 = R0;  

switch kernel_type   % the true kernel can have a support larger or smaller than the support of data u. 
    case 'sinkx'   
        k = 2; rEnd = 0.8;  % rEnd< R0. 
        K_true     = @(r) sin(k*2*pi*r).*(r>=0).*(r<=rEnd);  % r<=3)
        threshold  = 1e-8;   % threshold for support 
        kernel_str = 'sinkx'; 
    case 'coskx'   
        k = 2; rEnd = 0.8;  % rEnd< R0. 
        K_true     = @(r) 1+ cos(k*2*pi*r).*(r>=0.3).*(r<=rEnd); 
        % K_true0     = @(r) 1+ cos(k*2*pi*r).*(r>=0.3).*(r<=rEnd);  
        % temp  = K_true0(0.3); 
        % K_true    = @(r)  (0.3-r).*(r<0.3)+K_true0(r)- temp; 
        threshold  = 1e-8;   % threshold for support 
        kernel_str = 'coskx'; 
    case 'sinx_smooth'     % sin(x) 1-period
        k = 1; rEnd = R0;  % rEnd= R0. 
        K_true     = @(r) sin(k*r*2*pi).*(r>=0).*(r<=rEnd);   
        threshold  = 1e-8;   % threshold for support 
        kernel_str = 'sinx_smooth'; 
    case 'sinx_cubic'     % sin(x) 1-period
        k = 3; rEnd = R0;  % rEnd= R0. 
        K_true     = @(r) - 2*sin(k*r*2*pi).^3.*(r>=0).*(r<=rEnd);   
        threshold  = 1e-8;   % threshold for support 
        kernel_str = 'sinx_cubic'; 
    case 'polyx'     % sin(x) 1-period
        rEnd = R0;  % rEnd= R0. 
        K_true     = @(r) (1-r).*r.*(r>=0).*(r<=rEnd);   
        threshold  = 1e-8;   % threshold for support 
        kernel_str = 'polyx'; 
    case 'powerFn'     % sin(x) 1-period
        k = 2; rEnd = R0;  % rEnd= R0. 
        % K_true     = @(r) (2*r.^3+r.^2-1) .*(r>=0).*(r<=rEnd);   
        K_true     =@(r) - ( (r-0.5).^4-2*(r-0.5).^3- 2*(r-0.8).*(r-0.5).^2 ) .*(r>=0).*(r<=rEnd); 
        threshold  = 1e-8;   % threshold for support 
        kernel_str = 'powerFn'; 
    case 'Gaussian' 
        s  = 0.75;         mu = 0;
        % Multiply by the indicator so that K_true is defined only for r in [0,R0]
        K_true     = @(r) exp(-0.5/s/s*(r-mu).^2) / (sqrt(2*pi)*s) .* (r>=0 & r<=R0);
        kernel_str = [kernel_type, '_mean_', num2str(mu), '_std_', num2str(s)];
        threshold  = 1e-10;
    case 'GaussianPN'
        s1 = 1; mu1 = 5; s2 = 2; mu2 = 0;
        K1_true = @(r) exp(-0.5/s1/s1*(r-mu1).^2)/(sqrt(2*pi)*s1);
        K2_true = @(r) exp(-0.5/s2/s2*(r-mu2).^2)/(sqrt(2*pi)*s2);
        K_true  = @(r) (K1_true(r) - K2_true(r)) .* (r>=0 & r<=R0);
        kernel_str = [kernel_type, '_mean1_',num2str(mu1), '_std1_',num2str(s1),'_mean2_',num2str(mu2), '_std2_',num2str(s2)];
        threshold = 1e-8;
    case 'FracLap'
        s = 0.5;    d = xDim; 
        c_ds = (4^s * gamma(d/2+s)) / (pi^(d/2) * abs(gamma(-s)));
        % Here we use the indicator on the first term to confine to [0,R0]. Adjust the small-r treatment as needed.
         K_true = @(r) c_ds * ( (1./(r.^(d+2*s)) .* (r>0.1 & r<=R0)) + (1./(0.1^(d+2*s)) .* (r<=0.1) ) );
        % K_true = @(r) c_ds * ((1./r.^(d+2*s)) .* (r>0.1).*(r<=3) + (1./0.1.^(d+2*s)) .* (r<=0.1))+0*0.00842*(r>3);
        kernel_str = [kernel_type, '_d_', num2str(d), '_s_', num2str(s)];
        kernel_str = strrep(kernel_str, '.', '');
        kernelInfo.s = s;  
        kernelInfo.c_ds = c_ds;
        threshold = 1e-4;
    case 'Compoundlevy'
        % g_\nu(r)=exp(-r^2)
        rEnd = R0;
        K_true = @(r) exp(-r.^2).*(r>=0).*(r<=rEnd);
        kernel_str = 'Compoundlevy';
        threshold = 1e-8;
end
kernelInfo.K_true      = K_true;
kernelInfo.kernel_type = kernel_type;
kernelInfo.kernel_str  = kernel_str;

obsInfo.threshold = threshold;
obsInfo.u_supp    = u_supp;

%% 3 function handle of u and f, in terms of basis functions
% Here (u_i, f_i) are functions to generate data. They will be evaluated at points x_j later
switch u_Type
    % case 'Bspline'
    %     degree   = 1; % 1, 2, 3...
    %     knotsAug = 0;
    %     knots    = linspace(u_supp(1), u_supp(end), N+degree)';
    %     [v, vknots] = Basis_bspline(degree, knots, knotsAug);
    %     obsInfo.u = v;
    %     obsInfo.u_knots  = vknots;
    %     obsInfo.u_degree = degree;
    %     obsInfo.div_u    = @(x) 0;   % Not providing
    %     obsInfo.laplacian_u = @(x) 0;
    %     obsInfo.u_str   = [u_Type, '_', num2str(knots(1)), '_', num2str(knots(end)),'_N_',...
    %         num2str(numel(v)),'_deg_',num2str(degree), '_knotsAug_',num2str(knotsAug)];
    case 'Fourier'
        odd_even = 'cosine';    % 'sine', 'cosine',sincos  % use sine or cosine series, or use both;
        obsInfo.u_str = [u_Type,'_N_',num2str(N), odd_even];
        [v,basisInfo] = Basis_fourier(N, odd_even, u_supp,jump_discontinuous);
        obsInfo.u     = v;
        obsInfo.div_u = basisInfo.dbasis_funs;
        obsInfo.laplacian_u = basisInfo.ddbasis_funs;
    case 'GaussianMixture'
        temp = cell(N, 1);
        lb = u_supp(1);
        ub = u_supp(2);
        md = (lb + ub)/2;
        wd = md-lb;
        lb_mu   = md - wd/2;
        ub_mu   = md + wd/2;
        std_max = wd/6;
        mixture_num = 3;
        for i = 1:N
            mu  = unifrnd(lb_mu, ub_mu, [1, mixture_num]);
            std = unifrnd(0, std_max, [1, mixture_num]);
            density = @(x) 0*x;
            for j = 1:mixture_num
                density = @(x) density(x) + normpdf(x, mu(j), std(j));
            end
            temp{i} = density;
        end
        obsInfo.u = temp;
        obsInfo.u_str = 'GaussianMixture';
    case 'stocCosine'   %    periodic u: finite sum of cosine with random coefficient
         odd_even = 'cosine';    % 'sine', 'cosine'  % use sine or cosine series, or use both;
         n_u = 30;     %  
         obsInfo.u_str = [u_Type,'_N_',num2str(N), odd_even,'_nmodes_',num2str(n_u)];

         sqrt_covX_n = diag((1:n_u).^(-2)); 
         coef_Xn     = randn(N,n_u)*sqrt_covX_n; 
         u_fns   = cell(N, 1);
         du_fns  = cell(N, 1); % derivative of u
         ddu_fns = cell(N, 1); % second-order derivative of u 
         for k=1:N
             v1 =@(x) 0;  dv1 =@(x) 0; ddv1 =@(x) 0; 
             for n=1:n_u   
               %    v1 = @(x) v1+ basis_fn{n,1}*coef_Xn(k,n); 
                 v1   = @(x) v1(x)+cos(n*x*2*pi)*coef_Xn(k,n);
                 dv   = @(x) dv1(x)-n*2*pi*sin(n*x*2*pi)*coef_Xn(k,n);
                 ddv  = @(x) ddv1(x)-(n*2*pi)^2*cos(n*x*2*pi)*coef_Xn(k,n);
             end
             u_fns{k,1}   = v1;   
             du_fns{k,1}  = dv;
             ddu_fns{k,1} = ddv; 
         end 
         obsInfo.u           = u_fns;  % { u_k }
         obsInfo.div_u       = du_fns;
         obsInfo.laplacian_u = ddu_fns;
         obsInfo.u_str = 'stocCosine';
    case 'stocFourier'   %    periodic u: finite sum of cosine with random coefficient
        odd_even = 'cosine';    % 'sine', 'cosine' sincos % use sine or cosine series, or use both;
        n_u = 30;     %
        obsInfo.u_str = [u_Type,'_N_',num2str(N), odd_even,'_nmodes_',num2str(n_u)];
        [v,basisInfo] = Basis_fourier(n_u, odd_even, u_supp,jump_discontinuous);
        dv   = basisInfo.dbasis_funs;   % first order derivative
        ddv  = basisInfo.ddbasis_funs;  % second order derivative 
       
        sqrt_covX_n = diag((1:n_u).^(-2));
        coef_Xn     = randn(N,n_u)*sqrt_covX_n;
        u_fns   = cell(N, 1);
        du_fns  = cell(N, 1); % derivative of u
        ddu_fns = cell(N, 1); % second-order derivative of u
        for k=1:N
            v1 =@(x) 0;  dv1 =@(x) 0; ddv1 =@(x) 0;
            for n=1:n_u
                %    v1 = @(x) v1+ basis_fn{n,1}*coef_Xn(k,n);
                v1   = @(x) v1(x)   + v{n,1}(x)*coef_Xn(k,n);
                dv1  = @(x) dv1(x)  + dv{n,1}(x)*coef_Xn(k,n);
                ddv1 = @(x) ddv1(x) + ddv{n,1}(x)*coef_Xn(k,n);
            end
            u_fns{k,1}   = v1;
            du_fns{k,1}  = dv1;
            ddu_fns{k,1} = ddv1;
        end
        obsInfo.u           = u_fns;  % { u_k }
        obsInfo.div_u       = du_fns;
        obsInfo.laplacian_u = ddu_fns;
        obsInfo.u_str = 'stocFourier';
    case 'randomDensity'
        odd_even = 'cosine';    % 'cosine' sincos 
        n_u = 30;     %
        obsInfo.u_str = [u_Type,'_N_',num2str(N), odd_even,'_nmodes_',num2str(n_u)];
        [v,basisInfo] = Basis_fourier(n_u, odd_even, u_supp,jump_discontinuous);
        dv   = basisInfo.dbasis_funs;   % first order derivative
        ddv  = basisInfo.ddbasis_funs;  % second order derivative 

         sqrt_covX_n = diag((1:n_u).^(-2)); 
         temp        = -1 + 2*(rand(N,n_u) >= 0.5); 
         coef_Xn     = temp*sqrt_covX_n; 
          
        u_fns   = cell(N, 1);
        du_fns  = cell(N, 1); % derivative of u
        ddu_fns = cell(N, 1); % second-order derivative of u
        for k=1:N
            v1 =@(x) 1;  dv1 =@(x) 0; ddv1 =@(x) 0;
            for n=1:n_u
                %    v1 = @(x) v1+ basis_fn{n,1}*coef_Xn(k,n);
                v1   = @(x) v1(x)   + v{n,1}(x)*coef_Xn(k,n);
                dv1  = @(x) dv1(x)  + dv{n,1}(x)*coef_Xn(k,n);
                ddv1 = @(x) ddv1(x) + ddv{n,1}(x)*coef_Xn(k,n);
            end
            u_fns{k,1}   = v1;
            du_fns{k,1}  = dv1;
            ddu_fns{k,1} = ddv1;
        end
        obsInfo.u           = u_fns;  % { u_k }
        obsInfo.div_u       = du_fns;
        obsInfo.laplacian_u = ddu_fns;
        obsInfo.u_str = 'randomDensity'; 
end

% add an extra data  u(x) = x for case nonlinear opt
if strcmp(u_Type, 'randomDensity')
    if strcmp(example_type, 'Aggregation_StrForm') == 2 
        obsInfo.u{N+1}     = @(x) x.*(x>=0).*(x<=u_supp(2));
        obsInfo.div_u{N+1} = @(x) ones(size(x)).*(x>=0).*(x<=u_supp(2));
        obsInfo.laplacian_u{N+1} = @(x) zeros(size(x));
        obsInfo.u_str = strrep(obsInfo.u_str, ['N_',num2str(N)], ['N_',num2str(N+1)]);
    end
end

%% 4. x-mesh for data f(x) = R_phi[u](x)    
% We now choose the observation x-mesh based on the u support and the fixed kernel range R0.
% x_mesh = xmin:dx:xmax;   % f observation  x-mesh, not the u-mesh 
x_mesh_str = [num2str(xmin), '_', num2str(dx), '_', num2str(xmax)];
x_mesh_str = strrep(x_mesh_str, '.', 'p');
obsInfo.x_mesh = x_mesh;
obsInfo.x_mesh_dx = dx;
obsInfo.x_mesh_str = x_mesh_str;

obsInfo.u_xmesh    = u_xmesh; % u_supp(1):dx:u_supp(2); 
obsInfo.bdry_width = bdry_width;

% Compute the function handle for f using Quadrature.
f = cell(size(obsInfo.u)); 
for i = 1:numel(f)
    if strcmp(example_type, 'classicalReg') == 1
        f{1} = @(x) K_true(x);
    elseif strcmp(example_type, 'mfOpt') == 1 || strcmp(example_type, 'nonlinearOpt') == 1
        intgrand_i = @(x,y) K_true(y) .* fun_g(obsInfo.u{i}, obsInfo.div_u{i}, x, y);
        % Now integrate only over [0,R0]
        f{i} = @(x) obsInfo.laplacian_u{i}(x) + quadgk(@(y) intgrand_i(x,y), 0, R0, ...
                     'Waypoints', obsInfo.x_mesh, 'MaxIntervalCount', 10^10);
    else  % e.g., 'LinearIntOpt', 'nonlocal_radial1D', 'Aggregation_StrForm', etc.
        intgrand_i = @(x,y) K_true(y) .* fun_g(obsInfo.u{i}, obsInfo.div_u{i}, x, y);
        % Integrate over [0,R0]
        f{i} = @(x) quadgk(@(y) intgrand_i(x,y), 0, R0, ...
                     'Waypoints', obsInfo.x_mesh, 'MaxIntervalCount', 10^10);
    end
end

obsInfo.f = f;

fprintf('Example type: %s \nKernel type: %s\n', example_type, kernel_type);

end

