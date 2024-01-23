function [beta, history] = lasso_admm(X, y, lambda, rho, alpha, max_iter)
% ADMM algorithm for LASSO with l1-norm constraint
%
% Solves the problem:
%   minimize 0.5 * ||y - X*beta||_2^2 + lambda*||beta||_1
%
% Inputs:
% X: n x d matrix of input features
% y: n x 1 vector of output values
% lambda: scalar regularization parameter
% rho: scalar augmented Lagrangian parameter
% alpha: scalar relaxation parameter (typically 1.0)
% max_iter: maximum number of iterations to run the algorithm
%
% Outputs:
% beta: d x 1 vector of coefficients
% history: structure containing the convergence history
%
% Written by: Salman Asif, Georgia Tech
% Modified by: ChatGPT
% Date: 2023-04-06

% Normalize with n

[n, d] = size(X);
X = X ./ sqrt(n);
y = y ./ sqrt(n);

XtX = X'*X;
Xty = X'*y;

% Initialize variables
beta = zeros(d, 1);
z = zeros(d, 1);
u = zeros(d, 1);

% Cache factorization of X'X
L = chol(XtX + rho*eye(d));

% Initialize convergence history
history.objval = zeros(max_iter, 1);
history.r_norm = zeros(max_iter, 1);
history.s_norm = zeros(max_iter, 1);
history.eps_pri = zeros(max_iter, 1);
history.eps_dual = zeros(max_iter, 1);

% Run ADMM algorithm
for k = 1:max_iter
    
    % Update beta
    q = Xty + rho*(z - u); 
    beta = L \ (L' \ q);
    
    % Update z
    zold = z;
    x_hat = alpha*beta + (1-alpha)*zold;
    z = soft_thresh(x_hat + u, lambda/rho);
    
    % Update u
    u = u + x_hat - z;
    
    % Compute convergence metrics
    history.objval(k) = (0.5) * norm(y - X*beta)^2 + lambda*norm(z, 1);
    history.r_norm(k) = norm(beta - z);
    history.s_norm(k) = norm(-rho*(z - zold));
    history.eps_pri(k) = sqrt(d)*1e-6 + 1e-3*max(norm(beta), norm(z));
    history.eps_dual(k) = sqrt(d)*1e-6 + 1e-3*norm(rho*u);
    
    % Check convergence
    if (history.r_norm(k) < history.eps_pri(k) && ...
            history.s_norm(k) < history.eps_dual(k))
        break;
    end
end

% Trim convergence history
history.objval = history.objval(1:k);
history.r_norm = history.r_norm(1:k);
history.s_norm = history.s_norm(1:k);
history.eps_pri = history.eps_pri(1:k);
history.eps_dual = history.eps_dual(1:k);

end

function z = soft_thresh(x, lambda)
    z = sign(x) .* max(abs(x) - lambda, 0);
end

