function [h, p] = plot_mean_std(data_array,option) 
% plot the ensemble mean with std bar of samples in data_array:  X(samples,types,x_ind)
%{
    x_ind      - the index variable for the data 
    data_array -  n_samples x n_types x n_ind
%}

Color  = [zeros(3,1) eye(3)];
Color(:,1) = [85;170;170]/255;
Color(:,2) = [60;60;230]/255;
Color(:,3) = [170;0;170]/255;
Color(:,4) = [200;0;0]/255;
Color(:,5) = [225;125;0]/255;
Color(:,6) = [120;160;200]/255;


% matlab RGB 
matcolor = zeros(6,3);
matcolor(1,:) = [0 0.4470 0.7410];        % 'darkblue'
matcolor(2,:) = [0.8500 0.3250 0.0980];   % 'darkred'
matcolor(3,:) = [0.9290 0.6940 0.1250];   % 'darkyellow'
matcolor(4,:) = [0.4940 0.1840 0.5560];   % 'darkpurple'
matcolor(5,:) = [0.4660 0.6740 0.1880];   % 'darkgreen'
matcolor(6,:) = [0.3010 0.7450 0.9330];   % 'lightblue' 
code_str = {'#0072BD', '#D95319','#EDB120',	'#7E2F8E', '#77AC30','#4DBEEE','#D95319'};
name_str = {'darkblue','darkorange','darkyellow','darkpurple','darkgreen','lightblue','darkred'};

dark_color.matcolor = matcolor; 
dark_color.code_str = code_str; 
dark_color.name_str = name_str;
dred_do_db = matcolor([2,5,1,4,3],:); 

% linestyle = {'-',':','-.','--'};  %  A,B,C=the best, true
linestyles = {'--v','--o','--d','--p','--x'};
% markerstyle = {'x','o','diamond'};


%------------------------------ 
[n_samples,n_types,n_ind] = size(data_array); 
std_data    = zeros(n_types,n_ind);
mean_data   = zeros(n_types,n_ind);
for nn= 1:n_ind
    temp     = squeeze(data_array(:,:,nn));
    std_data(:,nn) = std(temp,0,1);
    mean_data(:,nn)= mean(temp,1);
end
fprintf('\n Mean of %i simulations, (n_type, x_ind): \n ',n_samples);
disp(mean_data);


h = gobjects(n_types,1);
p = gobjects(n_types,1);
for i= 1:n_types
     h(i) = boxchart(squeeze(data_array(:,i,:)), ...
        'BoxWidth', 0.3, ...
        'BoxFaceColor', dred_do_db(i,:), ...
        'BoxEdgeColor', dred_do_db(i,:), ...
        'WhiskerLineColor', dred_do_db(i,:), ...
        'JitterOutliers','on', ...
        'MarkerStyle', 'x', ...
        'MarkerColor', dred_do_db(i,:), ...
        'MarkerSize', 5); 
    hold on;
    p(i) = plot(mean_data(i,:), ...
        linestyles{i}, ...
        'linewidth',2, ...
        'Color',dred_do_db(i,:), ...
        'MarkerSize',8); 
    hold on;
end

if strcmp(option, 'log')
    set(gca, 'YScale', 'log');
elseif strcmp(option, 'linear')
    % Do nothing, keep default linear scale
else 
    error('wrong scale type')
end

grid on;
grid minor;
set(gca, 'GridAlpha', 0.1);
set(gca, 'MinorGridAlpha', 0.01);
box on;

end

