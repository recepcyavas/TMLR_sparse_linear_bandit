function [theta_pop, theta_prime] = popart(X, Y_pull, Q_pop, eps_param)

T = size(X, 1);
d = size(X, 2);

lambda = sqrt(8 * Q_pop(1,1) * eps_param / T);
alpha = sqrt(eps_param / Q_pop(1,1) * (1  + 2 * eps_param / (T - 2 * eps_param)));

X_pull = X';

for i = 1:T
    theta_tilde(:, i) = Q_pop * X_pull(:, i) * Y_pull(i);
end

for i = 1:d
    theta_prime(i, 1) = catoni(theta_tilde(i, :), alpha);
    
end

theta_pop = theta_prime .* (abs(theta_prime) >= lambda);

end