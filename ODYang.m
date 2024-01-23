function best_arm = ODYang(theta, A, T)
%Yang and Tan (2022) version of OD-LinBAI
if size(theta, 1) ~= 1
    theta = theta';
end
K = size(A, 2);
d_tilde = rank(A);
index_r = 1 : K;
theta_r = theta;
A_r = A;
m = (T - min(K, d_tilde * (d_tilde + 1) / 2) - ...
    sum(ceil(d_tilde ./ (2 .^ (1 : (ceil(log2(d_tilde)) - 1)))))) ...
    / ceil(log2(d_tilde));
%%%%%%%%%%%%%%%%%%%
for r = 1 : ceil(log2(d_tilde))
    rank_A = rank(A_r);
    if rank_A ~= size(A_r, 1)
        [U, S, ~] = svd(A_r,'econ');
        A_r = U(:, 1 : rank_A)' * A_r;
        theta_r = theta_r * U(:, 1 : rank_A);
    end
    pi_r = minvol(A_r);
    T_r = ceil(m * pi_r);
    V_r = 0;
    tmp = 0;
    for ii = 1 : size(A_r, 2)
        V_r = V_r + T_r(ii) * A_r(:, ii) * A_r(:, ii)';
        tmp = tmp + A_r(:, ii) * (theta_r * A_r(:, ii) * T_r(ii) + sqrt(T_r(ii)) * randn);
    end
    theta_r_hat = V_r \ tmp;
    p_r_hat = theta_r_hat' * A_r;
    [~, I] = maxk(p_r_hat, ceil(d_tilde / 2 ^ r));
    A_r = A_r(:, I);
    index_r = index_r(I);
end
best_arm =  index_r;
end

