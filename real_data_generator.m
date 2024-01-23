function [theta_s, X, y, Hardness, v_star, v_pop, Q_pop, xmax2, B, best] = real_data_generator()
newstable = [readmatrix('OnlineNewsPopularity.csv')];
%newstable = [readmatrix('abalone.csv')];

d = size(newstable, 2) - 1;
K_all = size(newstable, 1);

for i = 1:d + 1
    newstable(:, i) = (newstable(:, i) - mean(newstable(:, i))) / std(newstable(:, i));
end
Sigma = corrcoef(newstable(:, 1:d));
eliminate = [];
for i = 1:d
    for j = 1:d
        if abs(Sigma(i, j)) > 0.9 && i < j
            eliminate = [eliminate; [i, j]];
        end
    end
end
if ~isempty(eliminate)
elim = eliminate(:, 2);
newstable(:, elim') = [];
end
X = newstable(:, 1:end-1);
y = newstable(:, end);
d = size(X, 2);

theta_s = lasso_admm_cv(X, y, 30, 10, 5);
%theta = newstable(:, 1:end-1) \ newstable(:, end);
additional = 15;
K_s = 500-additional;
K = K_s + additional;
% XX = [];
% all_ind = [];
% while length(all_ind) < K
%     ind = randi(size(newstable, 1));
%     if ~ismember(ind, all_ind)
%     all_ind = [all_ind, ind];
%     XX = [XX; X(ind, :)];
%     end
% end
% X = XX;
[~ , I] = maxk(X(:, end), K_s);
X = X(I, :);
X = X';
[U S V] = eig(1/K_s * X * X');
X = [X sqrt(d) * U(:, end-additional+1:end)];

thres = max(abs(theta_s)) * 0.2;
index = abs(theta_s) < thres;
theta_s(index) = 0;



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

    for k = 1:K
        A{k} = X(:, k) * X(:, k)';
    end

    Imat = eye(d);
    
    cvx_begin
        cvx_precision best
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

    % cvx_begin 
    %     variable vpop(K)
    %     expression Bpop(d, d)
    %     expression Qdiag(d)
    %     Bpop = zeros(d, d);
    %     for i = 1:K
    %         Bpop = Bpop + vpop(i) * A{i};
    %     end 
    %     for i = 1:d
    %         Qdiag(i) = matrix_frac(Imat(:, i), Bpop);
    %     end
    %     minimize (max(Qdiag)) 
    %         subject to
    %         sum(vpop) == 1
    %         vpop >= 0
    % cvx_end

    vpop = v;
    Bpop = B;

    v_star = v;
    v_pop = vpop .* (vpop > 0);
    Q_pop = pinv(Bpop);

    x2 = zeros(1,d);
    for j = 1:d
    for i = 1:K
        x2(j) = x2(j) + v_star(i) * X(j, i)^2;
    end
    end
    xmax2 = max(x2);

end

