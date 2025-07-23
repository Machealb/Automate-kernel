
function [Tikhonov_sol, Tikhonov_lamb, Tikhonov_err] = comp_Tikh_regu(regressionData, rkhsType, method, xx, showError, plotOn) 
% compare the Tikhonov regularization estimators.
% rkhsType = {'auto-RKHS','auto-no-rho','Gaussian-RKHS','L2-rho'};
% method = {'LC','gcv'};
% 

nx = norm(xx);

Tikhonov_sol = struct();
Tikhonov_lamb = struct();
Tikhonov_err = struct();


% Loop for all normTypes 
for nn = 1:length(rkhsType)
    rkhs_type = rkhsType{nn};
    % figure(nn);
    switch rkhs_type
        case 'auto-RKHS'
            for k = 1:length(method)
                method_reg = method{k};

                if strcmp(method_reg, 'LC')
                    [x_reg,res,eta,reg_corner] = Tikh_auto_basis(regressionData, 'auto-RKHS','LC',plotOn); 
                    Tikhonov_sol.autoRKHS_LC   = x_reg;
                    Tikhonov_lamb.autoRKHS_LC  = reg_corner;
                    Tikhonov_err.autoRKHS_LC   = norm(x_reg-xx) / nx;
                elseif strcmp(method_reg, 'gcv')
                    [x_reg,res,eta,reg_corner] = Tikh_auto_basis(regressionData, 'auto-RKHS','gcv',plotOn);
                    Tikhonov_sol.autoRKHS_gcv  = x_reg;
                    Tikhonov_lamb.autoRKHS_gcv = reg_corner;
                    Tikhonov_err.autoRKHS_gcv  = norm(x_reg-xx) / nx;
                end
            end 
        case 'auto-no-rho'
            for k = 1:length(method)
                method_reg = method{k};

                if strcmp(method_reg, 'LC')
                    [x_reg,res,eta,reg_corner] = Tikh_auto_basis(regressionData, 'auto-no-rho','LC',plotOn); 
                    Tikhonov_sol.autonorho_LC  = x_reg;
                    Tikhonov_lamb.autonorho_LC = reg_corner;
                    Tikhonov_err.autonorho_LC  = norm(x_reg-xx) / nx;
                elseif strcmp(method_reg, 'gcv')
                    [x_reg,res,eta,reg_corner] = Tikh_auto_basis(regressionData, 'auto-no-rho','gcv',plotOn);
                    Tikhonov_sol.autonorho_gcv = x_reg;
                    Tikhonov_lamb.autonorho_gcv = reg_corner;
                    Tikhonov_err.autonorho_gcv  = norm(x_reg-xx) / nx;
                end
            end
        case 'Gaussian-RKHS'
            for k = 1:length(method)
                method_reg = method{k};

                if strcmp(method_reg, 'LC')
                    [x_reg,res,eta,reg_corner] = Tikh_auto_basis(regressionData, 'Gaussian-RKHS','LC',plotOn,0.1); 
                    Tikhonov_sol.GaussRKHS_LC  = x_reg;
                    Tikhonov_lamb.GaussRKHS_LC = reg_corner;
                    Tikhonov_err.GaussRKHS_LC  = norm(x_reg-xx) / nx;
                elseif strcmp(method_reg, 'gcv')
                    [x_reg,res,eta,reg_corner] = Tikh_auto_basis(regressionData, 'Gaussian-RKHS','gcv',plotOn,0.1);
                    Tikhonov_sol.GaussRKHS_gcv = x_reg;
                    Tikhonov_lamb.GaussRKHS_gcv = reg_corner;
                    Tikhonov_err.GaussRKHS_gcv  = norm(x_reg-xx) / nx;
                end
            end  
        case 'L2-rho'
            for k = 1:length(method)
                method_reg = method{k};

                if strcmp(method_reg, 'LC')
                    [x_reg,res,eta,reg_corner] = Tikh_discrete(regressionData,'L2-rho','LC',plotOn);
                    Tikhonov_sol.L2rho_LC  = x_reg;
                    Tikhonov_lamb.L2rho_LC = reg_corner;
                    Tikhonov_err.L2rho_LC  = norm(x_reg-xx) / nx;
                elseif strcmp(method_reg, 'gcv')
                    [x_reg,res,eta,reg_corner] = Tikh_discrete(regressionData,'L2-rho','gcv',plotOn);
                    Tikhonov_sol.L2rho_gcv = x_reg;
                    Tikhonov_lamb.L2rho_gcv = reg_corner;
                    Tikhonov_err.L2rho_gcv  = norm(x_reg-xx) / nx;
                end
            end 
    end
end


% display relative errors 
if showError == 1
    fprintf('Tikhonov regularization parameter and relative L2 Errors: \n'); 
    methods_all= ["auto-RKHS-LC"; "auto-RKHS-GCV"; "Gaussian-RKHS-LC"; "Gaussian-RKHS-GCV"; "L2-rho-LC"; "L2-rho-GCV"]; 
    Tikh_err_all  = [Tikhonov_err.autoRKHS_LC; Tikhonov_err.autoRKHS_gcv; Tikhonov_err.GaussRKHS_LC; Tikhonov_err.GaussRKHS_gcv; Tikhonov_err.L2rho_LC; Tikhonov_err.L2rho_gcv];
    Tikh_lamb_all   = [Tikhonov_lamb.autoRKHS_LC; Tikhonov_lamb.autoRKHS_gcv; Tikhonov_lamb.GaussRKHS_LC; Tikhonov_lamb.GaussRKHS_gcv; Tikhonov_lamb.L2rho_LC; Tikhonov_lamb.L2rho_gcv];

    % Format the errors to 4 digits using sprintf
    % Tikh_err_fmt  = compose('%.4f', Tikh_err_all);
    % Tikh_lamb_fmt = compose('%.4f', Tikh_lamb_all);
    Tikh_err_lamb = table(methods_all,Tikh_err_all,Tikh_lamb_all); 
    disp(Tikh_err_lamb)

end

end
