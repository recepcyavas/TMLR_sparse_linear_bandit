function [beta, lambda_thres_cv] = lasso_admm_cv_thres(X, y, lambda_init, K, set_lambda, num_rep, s)
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
c1 = 200;
c2 = 5;

n = size(X, 1);
if nargin < 6
    num_rep = 1;
end

if nargin < 7
    s = 2;
end

% Set maximum lambda value
% lambdaMax = norm(X'*y, inf) / n;
% log_lambda_range = linspace(min_log_lambda, 0, numLambda);
% lambda_range = lambdaMax .* 10.^(log_lambda_range);

L = length(set_lambda);

% Initialize cross-validation error array
cv_error = zeros(L, 1);

% Run cross-validation
for j = 1:num_rep
    % Split data into K folds
    kpartition = cvpartition(n, 'KFold', K);
    for i = 1:L
        lambda_thres = set_lambda(i);
        cv_error_i = 0;
        for k = 1:K
            indexTrain = training(kpartition, k);
            indexTest = test(kpartition, k);
            X_train = X(indexTrain, :);
            y_train = y(indexTrain, :);
            X_val = X(indexTest, :);
            y_val = y(indexTest, :);

            % Fit LASSO model on training set and compute validation error
            [beta_k, ~] = lasso_admm(X_train, y_train, lambda_init, rho, alpha, max_iter);

            % Thresholding
            beta_k(abs(beta_k) < lambda_thres) = 0;
            cv_error_i = cv_error_i + sum((y_val - X_val*beta_k).^2)/(2*size(y_val,1));
            cv_error_i = cv_error_i +  c1 * (sum(beta_k ~= 0) < s) + c2 * (sum(beta_k ~= 0) > s) * (sum(beta_k ~= 0));
        end
        cv_error(j, i) = cv_error_i/K;
    end
end
average_error = mean(cv_error, 1);

% Select lambda based on minimum cross-validation error
[~, min_idx] = min(average_error);
lambda_thres_cv = set_lambda(min_idx);

% Solve LASSO problem using selected lambda
[beta, ~] = lasso_admm(X, y, lambda_init, rho, alpha, max_iter);
beta(abs(beta) < lambda_thres_cv) = 0;

end

