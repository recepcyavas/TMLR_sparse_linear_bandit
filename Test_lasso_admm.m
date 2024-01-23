clear variables
%rng default
 n = 200;
 d = 20;
 s = 4;
 n_all = [25, 50, 100, 200, 400, 800, 1600, 3200]
 for j = 1:length(n_all)
     j
     n = n_all(j);
%  X = 1/sqrt(s) * randn(n, d);
%  for nn = 1:n
%      X(nn, :) = sqrt(d/s) * X(nn, :) / norm(X(nn, :));
%  end
 X = 1/sqrt(s) * randn(n, d);
% for i = 1:d
%      X(:, i) = sqrt(d/s)* X(:, i) / norm(X(:, i));
% end
 
 theta_s = zeros(d, 1);
 theta_s(1:s) = 1/sqrt(s);
 y = X * theta_s + randn(n, 1);
 numLambda = 100;
 K = 10;
 num_rep = 3;
 
 lambda = 0.2;
 rho = 1;
 alpha = 1;
 max_iter = 10^4;
%  tic
% [theta_admm] = lasso_admm(X, y, lambda, rho, alpha, max_iter);
% toc
% 
% tic
% theta_matlab = lasso(X, y, 'Lambda', lambda);
% toc
% 
% tic
% [theta_opt_admm, lambda_cv] = lasso_admm_cv(X, y, numLambda, K);
% toc

% for i = 1:d
%    X(:, i) =  sqrt(n / s) * X(:, i) / norm(X(:, i));
% end

% tic
% [theta_est, lambda_init(j), lambda_thres(j)] = t_lasso_cv_admm(X, y, K);
% toc
% for i = 1:1000
%     yy = X * theta_s + randn(n, 1);
%     [theta_est_yy, ~] = lasso_admm(X, yy, lambda_init(j), 1, 1, 10^3);
%     theta_est_yy(abs(theta_est_yy) < lambda_thres(j)) = 0;
%     score_1(i, j) = 1 - prod(ismember([1 2], find(theta_est_yy ~= 0)));
%     size_1(i, j) = sum(theta_est_yy ~= 0);
% end
    
tic
[theta_est_rep, lambda_init_rep, lambda_thres_rep,  current_init, current_thres, log_lambda_init_set, log_lambda_thres_set] = t_lasso_cv_admm_rep(X, y, K, num_rep, s);
toc

% lambda_init_rep = 0.0387;
% lambda_thres_rep = 0.1273;

for i = 1:1000
    yy = X * theta_s + randn(n, 1);
    [theta_est_rep_yy, ~] = lasso_admm(X, yy, lambda_init_rep, 1, 1, 10^3);
    theta_est_rep_yy(abs(theta_est_rep_yy) < lambda_thres_rep) = 0;
    score_2(i, j) = 1 - prod(ismember([1 2], find(theta_est_rep_yy ~= 0)));
    size_2(i, j) = sum(theta_est_rep_yy ~= 0);
end

average_size(j) = mean(size_2(:, j));
std_size(j) = std(size_2(:, j));
average_score(j) = mean(score_2(:, j));
std_score(j) = sqrt(average_score(j) * (1 - average_score(j)) / 1000);

 end

 figure
 semilogx(n_all, average_size, '-b');
 hold on
 semilogx(n_all, average_score, '--r');
 errorbar(n_all, average_size, std_size, 'b', "LineStyle", "none", "CapSize", 4)
 errorbar(n_all, average_score, std_score, 'r', "LineStyle", "none", "CapSize", 4)
 legend("Average estimated support size", "Probability of not detecting the support")
 grid on
 xlabel('$$T$$', 'interpreter', 'latex')
 title('$$s = 2, d = 20$$', 'interpreter', 'latex')
