function best_arm = LinearExploration(theta, A, T)
%MYALGO Summary of this function goes here
%   Detailed explanation goes here
if size(theta, 1) ~= 1
    theta = theta';
end
K = size(A, 2);
N = T /  ceil(log2(K));
S = 1 : K;
%%%%%%%%%%%%%%%%%%%
for r = 1 : ceil(log2(K))
    if length(S) <= 1
        break;
    end
    w = XY_allocation(A, [], A(:, S));
    V_r = 0;
    tmp = 0;
    for ii = 1 : size(A, 2)
        V_r = V_r + (w(ii) * N) * A(:, ii) * A(:, ii)';
        tmp = tmp + A(:, ii) * (theta * A(:, ii) * (w(ii) * N) + sqrt((w(ii) * N)) * randn);
    end
    theta_hat = V_r \ tmp;
    p_hat = theta_hat' * A(:, S);
    [~, I] = maxk(p_hat, ceil(length(S) / 2));
    S = S(:, I);
end
best_arm =  S;
end

