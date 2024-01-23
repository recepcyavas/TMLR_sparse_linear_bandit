function [theta_est, lambda_init, lambda_thres_best] = t_lasso_cv_admm(X, Y, K)

n = size(X, 1);
d = size(X, 2);
numLambda = 100;
numLambda_thres = 100;
maxLambda_thres = 1/sqrt(2);
lambda_thres_set = maxLambda_thres .* linspace(10^-2, 1, numLambda_thres); 
[theta_opt_admm, lambda_cv] = lasso_admm_cv(X, Y, numLambda, K, 3);
lambda_init = lambda_cv;
theta_est = theta_opt_admm;

%K-fold partition
kpartition = cvpartition(n, 'KFold', 10);
for k = 1:10
indexTrain = training(kpartition, k);
indexTest = test(kpartition, k);
X_train = X(indexTrain, :);
Y_train = Y(indexTrain, :);
X_test = X(indexTest, :);
Y_test = Y(indexTest, :);
[theta_opt_k, lambda_cv_k] = lasso_admm_cv(X_train, Y_train, numLambda, K, 1);
    
    for m = 1:length(lambda_thres_set)
        lambda_thres_m = lambda_thres_set(m);
        theta_thres_km = theta_opt_k .* (abs(theta_opt_k) >= lambda_thres_m);
        error_km(k, m) = 1/(2 * length(Y_test)) * sum((Y_test - X_test * theta_thres_km).^2);
        error_km(k, m) = error_km(k, m) + 10 * (sum(theta_thres_km ~= 0) < 2) + 1 * (sum(theta_thres_km ~= 0) > 2) * sum(theta_thres_km ~= 0);
    end
    
end
error_m = mean(error_km);
best_m = find(error_m == min(error_m), 1);

lambda_thres_best = lambda_thres_set(best_m);  

theta_est(abs(theta_opt_admm) < lambda_thres_best) = 0;

end

