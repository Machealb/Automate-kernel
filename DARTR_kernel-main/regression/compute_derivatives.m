function [du,ddu] = compute_derivatives(udata,dx,method,order)        
% compute the first- and 2nd-order derivatives of uk from udata(k,:) on mesh with dx;
[nk,nx] = size(udata); 
du  = 0*udata; 
ddu = du; 

% method = 'centralDiff'; % filterDiff

% If method and order are not provided, default to method = 'filterDiff' and order = 1
% Set defaults for optional inputs
if nargin < 3 || isempty(method)
    method = 'filterDiff';
end
if nargin < 4 || isempty(order)
    order = 1;
end

switch method
    case 'centralDiff'
    du(:, 2:end-1)  = (udata(:,3:1:end)-udata(:,1:1:end-2))/2/dx; 
    du(:,1)        = (udata(:,2)-udata(:,1))/dx; 
    du(:,end)      = (udata(:,end)-udata(:,end-1))/dx; 
    if order==2 
        ddu(:, 2:end-1) = (udata(:,3:1:end) - 2*udata(:,2:1:end-1) + udata(:,1:1:end-2))/(dx*dx);
    end
    case 'filterDiff'
        % Use Savitzky-Golay filtering to denoise the data before differentiation
        for k=1:nk
            uk        = udata(k,:);
            % Compute first derivative du_k/dx using central differences
            % Define Savitzky-Golay filter parameters
            windowSize = min(11,nx-1);  % Choose an odd window size, adjust based on N
            if mod(windowSize, 2) == 0
                windowSize = windowSize + 1;  % Ensure window size is odd
            end
            polyOrder = 3;               % Polynomial order (typically 2 or 3)
            % Apply Savitzky-Golay filter to denoise u_l
            uk_denoised = sgolayfilt(uk, polyOrder, windowSize);
            % Compute derivative on denoised u_l using central differences
            duk_dx      = gradient(uk_denoised, dx);
            du(k,:)     = duk_dx;
            if order==2
                ddu(k,:)    = gradient(duk_dx, dx);
            end
        end
end
end 