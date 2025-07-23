
function [Iter_sol, Iter_stop, Iter_lamb, Iter_err, Iter_all_err] = comp_Iter_regu(regressionData, rkhsType, method, K, xx, showError) 
% compare the Iterative regularization estimators.
% rkhsType = {'auto-RKHS','auto-no-rho','Gaussian-RKHS','L2-rho'};
% method = {'LC','hyb'};
% 

nx = norm(xx);

Iter_sol = struct();
Iter_stop = struct();
Iter_lamb = struct();
Iter_err = struct();
Iter_all_err = struct();


% Loop for all normTypes 
for nn = 1:length(rkhsType)
    rkhs_type = rkhsType{nn};
    % figure(nn);
    switch rkhs_type
        case 'auto-RKHS'
            for k = 1:length(method)
                method_reg = method{k};

                if strcmp(method_reg, 'LC')
                    [X_reg,res,eta,iter_stop] = idarr(regressionData, 'auto', K, 'LC', 0);
                    Iter_sol.autoRKHS_LC   = X_reg(:,iter_stop);
                    Iter_stop.autoRKHS_LC  = iter_stop;
                    err = zeros(length(res),1);
                    for i = 1:length(res)
                        err(i) = norm(X_reg(:,i)-xx) / nx;
                    end
                    Iter_all_err.autoRKHS_LC = err;
                    Iter_err.autoRKHS_LC = err(iter_stop);
                    [err_opt, iter_opt] = min(err);
                    
                    Iter_err.autoRKHS_opt  = err_opt;
                    Iter_stop.autoRKHS_opt = iter_opt;
                    Iter_sol.autoRKHS_opt  = X_reg(:,iter_opt);
                elseif strcmp(method_reg, 'hyb')
                    [X_reg,res,lamb,iter_stop] = idarr(regressionData, 'auto', K, 'hyb', 2048);
                    Iter_sol.autoRKHS_hyb   = X_reg(:,iter_stop);
                    Iter_stop.autoRKHS_hyb  = iter_stop;
                    Iter_lamb.autoRKHS_hyb  = lamb(iter_stop);
                    err = zeros(length(res),1);
                    for i = 1:length(res)
                        err(i) = norm(X_reg(:,i)-xx) / nx;
                    end
                    Iter_all_err.autoRKHS_hyb = err;
                    Iter_err.autoRKHS_hyb = err(iter_stop);
                end
            end 
        case 'auto-no-rho'
            for k = 1:length(method)
                method_reg = method{k};

                if strcmp(method_reg, 'LC')
                    [X_reg,res,eta,iter_stop] = idarr(regressionData, 'auto-no-rho', K, 'LC', 0);
                    Iter_sol.autonorho_LC   = X_reg(:,iter_stop);
                    Iter_stop.autonorho_LC  = iter_stop;
                    err = zeros(length(res),1);
                    for i = 1:length(res)
                        err(i) = norm(X_reg(:,i)-xx) / nx;
                    end
                    Iter_all_err.autonorho_LC = err;
                    Iter_err.autonorho_LC = err(iter_stop);
                    [err_opt, iter_opt] = min(err);

                    Iter_err.autonorho_opt  = err_opt;
                    Iter_stop.autonorho_opt = iter_opt;
                    Iter_sol.autonorho_opt  = X_reg(:,iter_opt);
                elseif strcmp(method_reg, 'hyb')
                    [X_reg,res,lamb,iter_stop] = idarr(regressionData, 'auto-no-rho', K, 'hyb', 0);
                    Iter_sol.autonorho_hyb   = X_reg(:,iter_stop);
                    Iter_lamb.autonorho_hyb  = lamb(iter_stop);
                    err = zeros(length(res),1);
                    for i = 1:length(res)
                        err(i) = norm(X_reg(:,i)-xx) / nx;
                    end
                    Iter_all_err.autonorho_hyb = err;
                    Iter_err.autonorho_hyb = err(iter_stop);
                end
            end
        case 'Gaussian-RKHS'
            for k = 1:length(method)
                method_reg = method{k};

                if strcmp(method_reg, 'LC')
                    [X_reg,res,eta,iter_stop] = idarr(regressionData, 'gauss', K, 'LC', 0, 0.1);
                    Iter_sol.GaussRKHS_LC   = X_reg(:,iter_stop);
                    Iter_stop.GaussRKHS_LC  = iter_stop;
                    err = zeros(length(res),1);
                    for i = 1:length(res)
                        err(i) = norm(X_reg(:,i)-xx) / nx;
                    end
                    Iter_all_err.GaussRKHS_LC = err;
                    Iter_err.GaussRKHS_LC = err(iter_stop);
                    [err_opt, iter_opt] = min(err);

                    Iter_err.GaussRKHS_opt  = err_opt;
                    Iter_stop.GaussRKHS_opt = iter_opt;
                    Iter_sol.GaussRKHS_opt  = X_reg(:,iter_opt);
                elseif strcmp(method_reg, 'hyb')
                    [X_reg,res,lamb,iter_stop] = idarr(regressionData, 'gauss', K, 'hyb', 2049, 0.1);
                    Iter_sol.GaussRKHS_hyb   = X_reg(:,iter_stop);
                    Iter_stop.GaussRKHS_hyb  = iter_stop;
                    Iter_lamb.GaussRKHS_hyb  = lamb(iter_stop);
                    err = zeros(length(res),1);
                    for i = 1:length(res)
                        err(i) = norm(X_reg(:,i)-xx) / nx;
                    end
                    Iter_all_err.GaussRKHS_hyb = err;
                    Iter_err.GaussRKHS_hyb = err(iter_stop);
                end
            end  
        case 'L2-rho'
            for k = 1:length(method)
                method_reg = method{k};

                if strcmp(method_reg, 'LC')
                    [X_reg,res,eta,iter_stop] = idarr(regressionData, 'L2-rho', K, 'LC', 0);
                    Iter_sol.L2rho_LC   = X_reg(:,iter_stop);
                    Iter_stop.L2rho_LC  = iter_stop;
                    err = zeros(length(res),1);
                    for i = 1:length(res)
                        err(i) = norm(X_reg(:,i)-xx) / nx;
                    end
                    Iter_all_err.L2rho_LC = err;
                    % fprintf('Iter_Stop = %d\n', iter_stop);
                    Iter_err.L2rho_LC = err(iter_stop);
                    [err_opt, iter_opt] = min(err);

                    Iter_err.L2rho_opt  = err_opt;
                    Iter_stop.L2rho_opt = iter_opt;
                    Iter_sol.L2rho_opt  = X_reg(:,iter_opt);
                elseif strcmp(method_reg, 'hyb')
                    [X_reg,res,lamb,iter_stop] = idarr(regressionData, 'L2-rho', K, 'hyb', 2050);
                    Iter_sol.L2rho_hyb   = X_reg(:,iter_stop);
                    Iter_stop.L2rho_hyb  = iter_stop;
                    Iter_lamb.L2rho_hyb  = lamb(iter_stop);
                    err = zeros(length(res),1);
                    for i = 1:length(res)
                        err(i) = norm(X_reg(:,i)-xx) / nx;
                    end
                    Iter_all_err.L2rho_hyb = err;
                    Iter_err.L2rho_hyb = err(iter_stop);
                end
            end 
    end
end


% display relative errors 
if showError == 1
    fprintf('Iterative regularization parameter and relative L2 Errors: \n'); 
    methods_all= ["auto-RKHS-LC"; "auto-RKHS-hybrid"; "Gaussian-RKHS-LC"; "Gaussian-RKHS-hybrid"; "L2-rho-LC"; "L2-rho-hybrid"]; 
    Iter_err_all  = [Iter_err.autoRKHS_LC; Iter_err.autoRKHS_hyb; Iter_err.GaussRKHS_LC; Iter_err.GaussRKHS_hyb; Iter_err.L2rho_LC; Iter_err.L2rho_hyb];
    Iter_opt_err_all  = [Iter_err.autoRKHS_opt; 0; Iter_err.GaussRKHS_opt; 0; Iter_err.L2rho_opt; 0];
    Iter_lamb_all   = [Iter_stop.autoRKHS_LC; Iter_lamb.autoRKHS_hyb; Iter_stop.GaussRKHS_LC; Iter_lamb.GaussRKHS_hyb; Iter_stop.L2rho_LC; Iter_lamb.L2rho_hyb];
    Iter_opt_stop_all  = [Iter_stop.autoRKHS_opt; 0; Iter_stop.GaussRKHS_opt; 0; Iter_stop.L2rho_opt; 0];
    % Format the errors to 4 digits using sprintf
    % Iter_err_fmt  = compose('%.4f', Iter_err_all);
    % Iter_lamb_fmt = compose('%.4f', Iter_lamb_all);
    % Iter_opt_err_fmt  = compose('%.4f', Iter_opt_err_all);
    % Iter_opt_stop_fmt  = compose('%.4f', Iter_opt_stop_all);

    Iter_result = table(methods_all,Iter_err_all,Iter_lamb_all,Iter_opt_err_all,Iter_opt_stop_all); 
    disp(Iter_result)

end

end
