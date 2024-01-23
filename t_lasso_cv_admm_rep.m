function [theta_est, lambda_init, lambda_thres, current_init, current_thres, log_lambda_init_set, log_lambda_thres_set] = t_lasso_cv_admm_rep(X, Y, K, num_rep, s)

if nargin < 5
    s = 2;
end
n = size(X, 1);
d = size(X, 2);
numLambda = 25;
numLambda_thres = 25;
repetition = 5;
maxLambda_init = norm(X' * Y, inf) / n;
maxLambda_thres = 1 / sqrt(2);

log_lambda_init_set = linspace(-3, 0, numLambda);
log_lambda_thres_set = linspace(-3, 0, numLambda_thres);
log_size = 3;
lambda_init_set =  maxLambda_init .* 10.^(log_lambda_init_set);
lambda_thres_set = maxLambda_thres .* 10.^(log_lambda_thres_set);
[~, lambda_init0] = lasso_admm_cv(X, Y, numLambda, K, 1);
current_init = zeros(1, num_rep);
current_thres = zeros(1, num_rep + 1);
current_thres(1) = min(10 * lambda_init0, 1/sqrt(2));

for r = 1:num_rep
[~, current_init(r)] = lasso_admm_cv_init(X, Y, current_thres(r), K, lambda_init_set, repetition, s);
init_idx = find(lambda_init_set == current_init(r), 1);
log_size = log_size / 2;
log_lambda_init_set = linspace(log_lambda_init_set(init_idx) - log_size / 2, log_lambda_init_set(init_idx) + log_size / 2, numLambda);
lambda_init_set =  maxLambda_init .* 10.^(log_lambda_init_set);

[~, current_thres(r + 1)] = lasso_admm_cv_thres(X, Y, current_init(r), K, lambda_thres_set, repetition, s);
thres_idx = find(lambda_thres_set == current_thres(r + 1), 1);
log_lambda_thres_set = linspace(log_lambda_thres_set(thres_idx) - log_size / 2, min(0, log_lambda_thres_set(thres_idx) + log_size / 2), numLambda_thres);
lambda_thres_set =  maxLambda_thres .* 10.^(log_lambda_thres_set);

end

lambda_init = current_init(num_rep);
lambda_thres = current_thres(num_rep + 1);

[theta_est, ~] = lasso_admm(X, Y, lambda_init, 1, 1, 1000);
theta_est(abs(theta_est) < lambda_thres) = 0;

end


