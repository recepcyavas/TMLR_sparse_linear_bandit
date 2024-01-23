function best_arm = ODLinBAI(theta, A, T)
% Our version of OD-LinBAI
if size(theta, 1) ~= 1
    theta = theta';
end
K = size(A, 2);
d_tilde = rank(A);
index_r = 1 : K;
theta_r = theta;
A_r = A;

nRounds = ceil(log2(d_tilde));
if nRounds == 0
    nRounds = 1;
end
Ttilde = floor(T / nRounds);
Ttilde_all = ones(1, nRounds) * Ttilde;
Ttilde_all(nRounds) = T - (nRounds - 1) * Ttilde;

%%%%%%%%%%%%%%%%%%%
for r = 1 : nRounds
    rank_A = rank(A_r);
    if rank_A ~= size(A_r, 1)
        [U, S, ~] = svd(A_r,'econ');
        A_r = U(:, 1 : rank_A)' * A_r;
        theta_r = theta_r * U(:, 1 : rank_A);
    end
    pi_r = minvol(A_r);
    % pi_r = optimal_allocation(A_r);
    T_r = roundT(pi_r, Ttilde_all(r)) * Ttilde_all(r);
    V_r = 0;
    tmp = 0;
    for ii = 1 : size(A_r, 2)
        V_r = V_r + T_r(ii) * A_r(:, ii) * A_r(:, ii)';
        tmp = tmp + A_r(:, ii) * (theta_r * A_r(:, ii) * T_r(ii) + sqrt(T_r(ii)) * randn);
    end
    if rank(V_r) == size(V_r, 1)
        theta_r_hat = V_r \ tmp;
    else
        theta_r_hat = pinv(V_r) * tmp;
    end
    p_r_hat = theta_r_hat' * A_r;
    [~, I] = maxk(p_r_hat, ceil(d_tilde / 2 ^ r));
    A_r = A_r(:, I);
    index_r = index_r(I);
end
best_arm =  index_r;
end


