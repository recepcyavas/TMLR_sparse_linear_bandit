function [beta, lambda_cv] = lasso_admm_cv(X, y, numLambda, K, MC_rep)
% ADMM algorithm for LASSO with l1-norm constraint and k-fold cross-validation for tuning lambda
%
% Solves the problem:
%   minimize 0.5 * ||y - X*beta||_2^2 + lambda*||beta||_1
%
% Inputs:
% X: n x d matrix of input features
% y: n x 1 vector of output values
% lambda_range: vector of lambda values to test in cross-validation
% rho: scalar augmented Lagrangian parameter
% alpha: scalar relaxation parameter (typically 1.0)
% max_iter: maximum number of iterations to run the algorithm
% K: number of folds for k-fold cross-validation
%
% Outputs:
% beta: d x 1 vector of coefficients
% lambda_cv: selected value of lambda based on cross-validation
%

rho = 1;
alpha = 1;
max_iter = 10^3;
min_log_lambda = -3;
n = size(X, 1);

% Set maximum lambda value
lambdaMax = norm(X'*y, inf) / n;
log_lambda_range = linspace(min_log_lambda, 0, numLambda);
lambda_range = lambdaMax .* 10.^(log_lambda_range);

cv_error = zeros(MC_rep, length(lambda_range));
for j = 1:MC_rep
% Split data into K folds
kpartition = cvpartition(n, 'KFold', K);
    % Run cross-validation
    for i = 1:length(lambda_range)
        lambda = lambda_range(i);
        cv_error_i = 0;
        for k = 1:K
            indexTrain = training(kpartition, k);
            indexTest = test(kpartition, k);
            X_train = X(indexTrain, :);
            y_train = y(indexTrain, :);
            X_val = X(indexTest, :);
            y_val = y(indexTest, :);

            % Fit LASSO model on training set and compute validation error
            [beta_k, ~] = lasso_admm(X_train, y_train, lambda, rho, alpha, max_iter);
            cv_error_i = cv_error_i + sum((y_val - X_val*beta_k).^2)/(2*size(y_val,1));
        end
        cv_error(j, i) = cv_error_i/K;
    end
end
average_cv_error = mean(cv_error, 1);
    
% Select lambda based on minimum cross-validation error
[~, min_idx] = min(average_cv_error);
lambda_cv = lambda_range(min_idx);

% Solve LASSO problem using selected lambda
[beta, ~] = lasso_admm(X, y, lambda_cv, rho, alpha, max_iter);

end

