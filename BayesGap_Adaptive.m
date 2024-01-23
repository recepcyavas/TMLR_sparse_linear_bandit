function best_arm = BayesGap_Adaptive(theta, A, T, eta)
%BAYESGAP_ADAPTIVE Summary of this function goes here
%   Detailed explanation goes here
if size(theta, 1) ~= 1
    theta = theta';
end
K = size(A, 2);
d = size(A, 1);
if nargin < 4
    eta = 1000000;
end
kappa = 0;
for ii = 1 : K
    kappa = kappa + norm(A(:, ii), 2) ^ (-2);
end
mu = theta * A;
Delta = zeros(d, 1);
Sigma = eta ^ (-2) * eye(d);
theta_tmp = zeros(d, 1);
B_T = zeros(1, T);
J_T = zeros(1, T);
for t = 1 : T
    Sigma_inv = inv(Sigma);
    theta_hat = Sigma_inv * theta_tmp;
    mu_t = theta_hat' * A;
    sigma_t = zeros(1, K);
    for ii = 1 : K
        sigma_t(ii) = sqrt(A(:, ii)' * Sigma_inv * A(:, ii));
    end
    U_t_3sigma = mu_t + 3 * sigma_t;
    L_t_3sigma = mu_t - 3 * sigma_t;
    [U_t_3sigma_max_2, I] = maxk(U_t_3sigma, 2);
    Delta = U_t_3sigma_max_2(1) - L_t_3sigma;
    Delta(I(1)) = U_t_3sigma_max_2(2) - L_t_3sigma(I(1));
    Delta = abs(Delta);
    [Delta_min_2, I] = mink(Delta, 2);
    Delta(I(1)) = Delta_min_2(2);
    H_keps = (Delta + Delta_min_2(2)) / 2;
    H_keps = (Delta) / 2;
    H_eps = sum(H_keps .^ (-2));
    beta = sqrt((T - K + kappa / eta ^ 2) / 4 / H_eps);
    s_t = 2 * beta * sigma_t;
    U_t = mu_t + beta * sigma_t;
    L_t = mu_t - beta * sigma_t;
    [U_t_max_2, I] = maxk(U_t, 2);
    B_t = U_t_max_2(1) - L_t;
    B_t(I(1)) = U_t_max_2(2) - L_t(I(1));
    [B_T(t), J_T(t)] = min(B_t);
    index_tmp = [1 : (J_T(t) - 1), (J_T(t) + 1) : K];
    [~, I] = max(U_t(index_tmp));
    j_t = index_tmp(I);
    [~, I] = max([s_t(j_t), s_t(J_T(t))]);
    if I == 1
        a_t = j_t;
    else
        a_t = J_T(t);
    end
    Sigma = Sigma + A(:, a_t) * A(:, a_t)';
    theta_tmp = theta_tmp + A(:, a_t) * (theta * A(:, a_t) + randn);
end
[~, I] = min(B_T);
best_arm = J_T(I);
end

