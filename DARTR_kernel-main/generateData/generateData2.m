function  [obsInfo,ux_val,fx_val] = generateData2(kernelInfo, obsInfo, SAVE_DIR,saveON,integrator)
% generate data without noise

if ~exist('saveON','var'); saveON = 0; end
filename = [SAVE_DIR, 'Data_', kernelInfo.kernel_str, '_ui_', obsInfo.u_str,'_xj_', obsInfo.x_mesh_str, '.mat'];

if ~exist('integrator','var'); integrator = 'Riemann'; end

if ~exist(filename, 'file')
            u      = obsInfo.u;  Du = obsInfo.div_u; 
            f      = obsInfo.f;

            x_mesh = obsInfo.x_mesh;
            dx     = obsInfo.x_mesh_dx;
            J      = numel(x_mesh); N = numel(u);

            if strcmp(integrator,'Riemann')
                fun_g_vec  = obsInfo.fun_g_vec;
                N_x_f      = length(x_mesh);
                bdry_width = obsInfo.bdry_width;
                rInd       = 1:bdry_width; 
                r_mesh     = rInd*obsInfo.x_mesh_dx; 
                Kernel_val = kernelInfo.K_true(r_mesh);
                Kernel_val = reshape(Kernel_val,[1,length(r_mesh)]); 
            end


            u_xmesh =  obsInfo.u_xmesh; nx_u = length(u_xmesh);
            ux_val  = zeros(N, nx_u); % ux_val(n,j) = u_n(x_j)
            % dux_val = zeros(N, J, J); % dux_val(n,j1,j2) = u_n(x_j1) - u_n(x_j2)
            fx_val = zeros(N, J); % fx_val(n,j) = f_n(x_j) --> b_mat

            switch obsInfo.example_type
                case {'LinearIntOpt','nonlocal'}
                    for n = 1:N
                        u1  = arrayfun(u{n}, u_xmesh);
                        Du1 = 0*u1;
                        ux_val(n,:) = u1;
                        if strcmp(integrator,'Riemann')
                            covlgu = zeros(length(rInd),N_x_f);
                            for k  = 1:N_x_f
                                covlgu(:,k)  = fun_g_vec(u1,Du1,k+bdry_width,rInd);
                            end
                            fx_val(n,:) = Kernel_val*covlgu*dx;    % Riemann sum
                        else
                            fval_temp   = arrayfun(f{n}, x_mesh);
                            fx_val(n,:) = fval_temp';
                        end
                    end
                case {'Aggregation_StrForm'}   % 
                    for n = 1:N
                        u1  = arrayfun(u{n}, u_xmesh);
                        if isequal(Du, 0)
                            method   = 'filterDiff'; % 'centralDiff';  filterDiff
                            [Du1,~]  = compute_derivatives(u1,dx,method,1);
                        else
                            Du1 = arrayfun(Du{n}, u_xmesh);
                        end
                        ux_val(n,:) = u1;
                        if strcmp(integrator,'Riemann')
                            covlgu = zeros(length(rInd),N_x_f);
                            for k  = 1:N_x_f
                                covlgu(:,k)  = fun_g_vec(u1,Du1,k+bdry_width,rInd);
                            end
                            fx_val(n,:) = Kernel_val*covlgu*dx;    % Riemann sum
                        else
                            fval_temp   = arrayfun(f{n}, x_mesh);
                            fx_val(n,:) = fval_temp';
                        end
                    end
            end
    
       
    %% noise range to be reasonable: depend on mesh, depend on f
    % noiseL2^2 = noise_std^2 *sum(length(x_mesh))*dx
    % fL2^2     = sum(f(:,1).^2)*dx
    
    dx          = obsInfo.x_mesh_dx;
    f_L2norm2   = sum(fx_val.^2,2)*dx;
    f_mean      = mean(fx_val,2);
    f_std       = std(fx_val,0,2);
    noise_std_upperBd = min( [ min(f_mean+f_std), sqrt(min(f_L2norm2)/sum(length(x_mesh))*dx)]);
    obsInfo.noise_std_upperBd = noise_std_upperBd;
    obsInfo.f_L2norm  = f_L2norm2;
    obsInfo.f_mean    = f_mean;
    obsInfo.f_std     = f_std;
    
    
    if saveON==1;     save(filename,  'obsInfo','ux_val','fx_val'); end
else
    load(filename,  'obsInfo','ux_val','fx_val'); %
end

if obsInfo.plotON==1     % plot the first 3 input functions u_k
    figure(81);clf; 
    for i=1:3
        plot(u_xmesh,ux_val(i,:)); hold on;
    end
end

end