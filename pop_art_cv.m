function [beta, eps_cv] = pop_art_cv(X, y, Q_pop, set_eps, num_rep, s)
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
[n, d] = size(X);
c1 = 200;
c2 = 5;
K = 10;

if nargin < 5
    num_rep = 1;
end

if nargin < 6
    s = 2;
end

% Set maximum lambda value
% lambdaMax = norm(X'*y, inf) / n;
% log_lambda_range = linspace(min_log_lambda, 0, numLambda);
% lambda_range = lambdaMax .* 10.^(log_lambda_range);

L = length(set_eps);

% Initialize cross-validation error array
cv_error = zeros(num_rep, L);


% Run cross-validation
for j = 1:num_rep
    % Split data into K folds
    kpartition = cvpartition(n, 'KFold', K);
    for i = 1:L
        eps_param = set_eps(i);
        cv_error_i = 0;
        for k = 1:K
            indexTrain = training(kpartition, k);
            indexTest = test(kpartition, k);
            X_train = X(indexTrain, :);
            y_train = y(indexTrain, :);
            X_val = X(indexTest, :);
            y_val = y(indexTest, :);

            % Fit LASSO model on training set and compute validation error
            [beta_k, ~] = popart(X_train, y_train, Q_pop, eps_param);

            % Calculate test error
            cv_error_i = cv_error_i + sum((y_val - X_val*beta_k).^2)/(2*size(y_val,1));
            cv_error_i = cv_error_i + c1 * (sum(beta_k ~= 0) < s) + c2 * (sum(beta_k ~= 0) > s) * (sum(beta_k ~= 0));
        end
        cv_error(j, i) = cv_error_i/K;
    end
end
average_error = mean(cv_error, 1);
% Select lambda based on minimum cross-validation error
[~, min_idx] = min(average_error);
eps_cv = set_eps(min_idx);

[beta, ~] = popart(X, y, Q_pop, eps_cv);

end

