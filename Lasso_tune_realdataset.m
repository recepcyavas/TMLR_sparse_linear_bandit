function [lambda_init, lambda_thres, score2] = Lasso_tune_realdataset(X_lasso, theta_s)
        T1 = size(X_lasso, 1);
        support = find(theta_s ~= 0);
        s = length(support);
        theta_min = min(abs(theta_s(support)));
        
        i2 = 15;
        lambda_thres_all = linspace(theta_min * 0.1, theta_min * 1, i2);
        lambda_init_all = linspace(0.001, 0.1, i2);
        lambda_init_size = 0.1 - 0.001;
        lambda_thres_size = theta_min * 0.9;
        score = zeros(1, i2);
        lambda_thres = theta_min / 2;

        for round = 1:4
            round
        for i = 1:length(lambda_init_all)
            lambda_init = lambda_init_all(i);
                i
                score(i) = 0;
                for k = 1:400
                    Y_lasso = X_lasso * theta_s + randn(size(X_lasso, 1), 1);
                    [theta_est, ~] = lasso_admm(X_lasso, Y_lasso, lambda_init, 1, 1, 10^3); 
                    theta_est_thresholded = theta_est;
                    theta_est_thresholded(abs(theta_est_thresholded) < lambda_thres) = 0;
                    theta_est_supp = find(theta_est_thresholded ~= 0);
                    % support = support(1:2);
                    score(i) = score(i)  + 10 * norm(theta_est_thresholded - theta_s, 1) + (length(theta_est_supp) < length(support)) * 50 + (1 - prod(ismember(support, theta_est_supp))) * 20 + abs(length(theta_est_supp) - s);
                    % score(i) = score(i) + (1 - prod(ismember(support, theta_est_supp)) * (length(theta_est_supp) <= 8));
                end
        end
        ind = find(score == min(score));
        lambda_init = lambda_init_all(ind);
        lambda_init_size = lambda_init_size / 2;
        lambda_init_all = linspace(max(0.001, lambda_init - lambda_init_size/2), lambda_init + lambda_init_size/2, i2);

            for j = 1:length(lambda_thres_all)
                lambda_thres = lambda_thres_all(j);
                score(j) = 0;
                j
                for k = 1:400
                    Y_lasso = X_lasso * theta_s + randn(size(X_lasso, 1), 1);
                    [theta_est, ~] = lasso_admm(X_lasso, Y_lasso, lambda_init, 1, 1, 10^3); 
                    theta_est_thresholded = theta_est;
                    theta_est_thresholded(abs(theta_est_thresholded) < lambda_thres) = 0;
                    theta_est_supp = find(theta_est_thresholded ~= 0);
                    score(j) = score(j)  + 10 * norm(theta_est_thresholded - theta_s, 1) + (length(theta_est_supp) < length(support)) * 50 + (1 - prod(ismember(support, theta_est_supp))) * 20 + abs(length(theta_est_supp) - s);
                    % score(j) = score(j) + (1 - prod(ismember(support, theta_est_supp)) * (length(theta_est_supp) <= 8));
                end
            end
        ind = find(score == min(score));
        lambda_thres = lambda_thres_all(ind);
        lambda_thres_size = lambda_thres_size / 2;
        lambda_thres_all = linspace(max(0.001, lambda_thres - lambda_thres_size/2), lambda_thres + lambda_thres_size/2, i2); 
        end
        
        score2 = 0;
        for k = 1:400
                    Y_lasso = X_lasso * theta_s + randn(size(X_lasso, 1), 1);
                    [theta_est, ~] = lasso_admm(X_lasso, Y_lasso, lambda_init, 1, 1, 10^3); 
                    theta_est_thresholded = theta_est;
                    theta_est_thresholded(abs(theta_est_thresholded) < lambda_thres) = 0;
                    theta_est_supp = find(theta_est_thresholded ~= 0);
                    score2 = score2 + (1 - prod(ismember(support, theta_est_supp)) * (length(theta_est_supp) <= 8));
        end
        score2 = score2 / 3000;

end