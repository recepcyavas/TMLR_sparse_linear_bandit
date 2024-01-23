Hardness = 1;
Hardness_LB = 0.25;
min_H = 100;
max_H = 1000;
d = 10;
s = 2;
K = 50;

while Hardness < min_H || Hardness > max_H

X = 1/sqrt(s) * randn(d, K);
for i = 1:K
    if mode == 1
    elseif mode == 2
        X(:, i) = sqrt(d/s) * X(:, i) / norm(X(:, i));
    elseif mode == 3
        if i == 1
            X(1:s, i) = [1/sqrt(2); 1/sqrt(2)];
        elseif i == K
            X(1:s, i) = [cos(5 * pi/4); sin(5 * pi/4)];
        else
            X(1:s, i) = [cos(pi/2 + 0.09 * randn); sin(pi/2 + 0.09 * randn)];
        end
        X(s+1:end, i) = sqrt((d-s)/s)* X(s+1:end, i) ./ norm(X(s+1:end, i));
    elseif mode == 4
        X = zeros(d, K);
        for i = 1:d
            for j = 1:K
                if X(i, j) == 0
                    X(i, j) = (2 * (rand < 0.5) - 1) * cos(pi/4 + 0.1 * randn);
                end
            end
        end
    end
end

theta_s = zeros(d, 1);
theta_s(1:s) = 1/sqrt(s) * ones(s, 1);

mu = X' * theta_s;
best = find(mu == max(mu));

sorted_mu = flip(sort(mu));
for i = 2:K
    Delta(i) = sorted_mu(1) - sorted_mu(i);
end
Delta(1) = Delta(2);
Delta2 = Delta.^2;
Hi = [2:d] ./ Delta2(2:d);
Hardness = max(Hi);

end
% 
% subsets = nchoosek([1:d], s);
% for j = 1:nchoosek(d, s)
%     index = subsets(j, :);
%     theta_temp = zeros(d, 1);
%     theta_temp(index) = 1/sqrt(s) * ones(s, 1);
%     mu_temp = X' * theta_temp;
%     best_temp = find(mu_temp == max(mu_temp));
%     sorted_mu_temp = flip(sort(mu_temp));
%     for i = 2:K
%         Delta_temp(i) = sorted_mu_temp(1) - sorted_mu_temp(i);
%     end
%     Delta_temp(1) = Delta_temp(2);
%     Delta2_temp = Delta_temp.^2;
%     Hi_temp = [2:d] ./ Delta2_temp(2:d);
%     Hardness_temp(j) = max(Hi_temp);
% end
% Hardness_LB = min(Hardness_temp);

    for k = 1:K
        A{k} = X(:, k) * X(:, k)';
    end

    Imat = eye(d);
    
    cvx_begin quiet
        variable v(K)
        expression B(d, d)
        B = zeros(d, d);
        for i = 1:K
            B = B + v(i) * A{i};
        end
        maximize( lambda_min( B ))
            subject to
            v >= 0
            sum(v) == 1
    cvx_end

    cvx_begin quiet
        variable vpop(K)
        expression Bpop(d, d)
        expression Qdiag(d)
        Bpop = zeros(d, d);
        for i = 1:K
            Bpop = Bpop + vpop(i) * A{i};
        end 
        for i = 1:d
            Qdiag(i) = matrix_frac(Imat(:, i), Bpop);
        end
        minimize (max(Qdiag)) 
            subject to
            sum(vpop) == 1
            vpop >= 0
    cvx_end

    v_star = v;

    x2 = zeros(1,d);
    for j = 1:d
    for i = 1:K
        x2(j) = x2(j) + v_star(i) * X(j, i)^2;
    end
    end
    xmax2 = max(x2);