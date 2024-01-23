function [X_lasso_all, v_lasso_all, lambda_init_all, lambda_thres_all] = Lasso_tune(X, theta_s, T_all, v_star, c0)
%This function tunes the Lasso parameters
d = size(X, 1);
K = size(X, 2);
s = sum(theta_s ~= 0);
if nargin < 4
    c0 = 0.25;
end

for t_index = 1:length(T_all)
    T = T_all(t_index)
    T2 = floor(T / (1 + c0));
    T1 = T - T2;
    
    v_lasso = roundT(v_star, T1);
    X_lasso = zeros(T1, d);
    numTimes = round(v_lasso * T1);
    temp_t = 0;
    for i = 1:K 
        if numTimes(i) > 0
            X_lasso(temp_t + 1:temp_t + numTimes(i), :) = repmat(X(:, i)', [numTimes(i), 1]);
            temp_t = temp_t + numTimes(i);
        end
    end
    
    fprintf('Lasso parameter tuning: ')
    tic
    lambda_init_m = zeros(1, 5);
    lambda_thres_m = zeros(1, 5);
    for m = 1:5
        Y_lasso = X_lasso * theta_s + randn(T1, 1);
        [~, lambda_init_m(m), lambda_thres_m(m)] = t_lasso_cv_admm_rep(X_lasso, Y_lasso, 10, 3, s);
    end
    toc

    % The following averaging helps obtain smoother changes over parameters across T.
    lambda_init_hist(t_index) = mean(lambda_init_m);
    lambda_thres_hist(t_index) = mean(lambda_thres_m);

    lambda_init_all(t_index) = mean(lambda_init_hist);
    lambda_thres_all(t_index) = mean(lambda_thres_hist);

    v_lasso_all{t_index} = v_lasso;
    X_lasso_all{t_index} = X_lasso;

end


end