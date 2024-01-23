function [eps_cv] = pop_art_cv_rand(arms, theta_s, T, v_pop, Q_pop, set_eps, num_rep, s)

c1 = 200;
c2 = 5;

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
    cv_error_i = 0;
    for i = 1:L
        eps_param = set_eps(i);
        [X_train, y_train] = generate_pop_pulls(arms, theta_s, T, v_pop);
        [beta_k, ~] = popart(X_train, y_train, Q_pop, eps_param);
        
        % Calculate test error
        cv_error_i = cv_error_i + sum((y_train - X_train*beta_k).^2)/(2*size(y_train,1));
        cv_error_i = cv_error_i + c1 * (sum(beta_k ~= 0) < s) + c2 * (sum(beta_k ~= 0) > s) * (sum(beta_k ~= 0));
    
        cv_error(j, i) = cv_error_i;
    end
end
average_error = mean(cv_error, 1);
% Select lambda based on minimum cross-validation error
[~, min_idx] = min(average_error);
eps_cv = set_eps(min_idx);

end

