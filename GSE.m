function best_arm = GSE(theta, A, T)
%MYALGO Summary of this function goes here
%   Detailed explanation goes here
if size(theta, 1) ~= 1
    theta = theta';
end
K = size(A, 2);
d_tilde = rank(A);
index_r = 1 : K;
theta_r = theta;
A_r = A;
m = (T / ceil(log2(K)));
%%%%%%%%%%%%%%%%%%%
for r = 1 : ceil(log2(K))
    %rank_Ar = max(2, rank(A_r) - length(find(eig(A_r * A_r') < max(eig(A_r * A_r')) / 10000)));
    rank_Ar = rank(A_r);
    if rank_Ar ~= size(A_r, 1)
        [U, S, ~] = svd(A_r,'econ');
        A_r = U(:, 1 : rank_Ar)' * A_r;
        theta_r = theta_r * U(:, 1 : rank_Ar);
    end
    pi_r = minvol(A_r);
    T_r = (m * pi_r);
    V_r = 0;
    tmp = 0;
    for ii = 1 : size(A_r, 2)
        V_r = V_r + T_r(ii) * A_r(:, ii) * A_r(:, ii)';
        tmp = tmp + A_r(:, ii) * (theta_r * A_r(:, ii) * T_r(ii) + sqrt(T_r(ii)) * randn);
    end
    theta_r_hat = V_r \ tmp;
    p_r_hat = theta_r_hat' * A_r;
    [~, I] = maxk(p_r_hat, ceil(length(index_r) / 2 ));
    A_r = A_r(:, I);
    index_r = index_r(I);
end
best_arm =  index_r;
end

