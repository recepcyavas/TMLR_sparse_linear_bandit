function [w, value] = XY_allocation(A, theta, S)
% Frank-Wolfe heuristic
% Rémy Degenne, Pierre Ménard, Xuedong Shang, and Michal Valko. Gamiﬁcation
% of pure exploration for linear bandits. In International Conference on
% Machine Learning, pages 24322442. PMLR, 2020.

K = size(A, 2);
d = size(A, 1);

if (nargin == 1)
    B = kron(ones(1, K), A) - kron(A, ones(1, K));
end

if (nargin == 2)
    B = [];
    if size(theta, 1) ~= 1
        theta = theta';
    end
    reward = theta * A;
    [max_reward, best_arm] = max(reward);
    reward_delta = max_reward - reward;
    for ii = 1 : K
        if ii ~= best_arm
            B = [B, (A(:, best_arm) - A(:, ii)) / reward_delta(ii)];
        end
    end
end

if (nargin == 3)
    K_tmp = size(S, 2);
    B = kron(ones(1, K_tmp), S) - kron(S, ones(1, K_tmp));
end

max_it = 250;
K_B = size(B, 2);
w = ones(K, 1);
V = eye(d);
for t = 1 : max_it
    w_old = w;
    max_tmp = 0;
    V_inverse = V ^ (-1);
    for ii = 1 : K
        for jj = 1 : K_B
            tmp = (A(:, ii)' * V_inverse * B(:, jj))^2;
            if tmp > max_tmp
                max_tmp = tmp;
                a_tilde = ii;
            end
        end
    end
    V = V + A(:, a_tilde) * A(:, a_tilde)';
    w = t / (t+1) * w;
    w(a_tilde, 1) = w(a_tilde, 1) + 1 / (t+1);
    if norm(w - w_old) <= 0.01
        break;
    end
end
w = w / sum(w);
V = 0;
for ii = 1 : K
    V = V + w(ii) * A(:, ii) * A(:, ii)';
end
V_inverse = V ^ (-1);
max_tmp = 0;
for jj = 1 : K_B
    tmp = B(:, jj)' * V_inverse * B(:, jj);
    if tmp > max_tmp
        max_tmp = tmp;
    end
end
value = max_tmp;
end

