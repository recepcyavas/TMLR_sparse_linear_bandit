function best_arm = Peace(theta, A, T)
%MYALGO Summary of this function goes here
%   Detailed explanation goes here
if size(theta, 1) ~= 1
    theta = theta';
end
K = size(A, 2);
S = 1 : K;
[~, value] = XY_allocation(A, [], A(:, S));
R = ceil(log2(value));
N = floor(T / R);
%%%%%%%%%%%%%%%%%%%
for r = 1 : R
    r
    if length(S) <= 2
        break;
    end
    [w, value] = XY_allocation(A, [], A(:, S));
    V_r = 0;
    tmp = 0;
    for ii = 1 : size(A, 2)
        V_r = V_r + (w(ii) * N) * A(:, ii) * A(:, ii)';
        tmp = tmp + A(:, ii) * (theta * A(:, ii) * (w(ii) * N) + sqrt((w(ii) * N)) * randn);
    end
    theta_hat = V_r \ tmp;
    p_hat = theta_hat' * A(:, S);
    [~, tmptmp] = max(p_hat);
    tmptmp = S(:, tmptmp);
    for ii = (length(S) - 1) : -1 : 2
        [~, I] = maxk(p_hat, ii);
        [~, value_tmp] = XY_allocation(A, [], A(:, S(:, I)));
        if value_tmp <= value/2
            break;
        end
        [~, I] = maxk(p_hat, 1);
    end
    S = S(:, I);
end
best_arm =  tmptmp;
end

