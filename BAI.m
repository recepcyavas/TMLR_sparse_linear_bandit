clearvars 
%cvx_solver SDPT3  % If the default solver is different, uncomment this.
rng default
d = 10
K = 50
T_all = floor(10.^linspace(2.33, 4, 20));
T_all = floor(T_all / 10) * 10;
s = 2

mode = 2;
% mode 1 = Gaussian entries
% mode 2 = Spherically distributed entries
% mode 3 = Yang and Tan example
% mode 4 = theta_s belongs to a finite set
[X, theta_s, best, Hardness, v_star, xmax2, B, Hardness_LB, v_pop, Q_pop] = generate_instance_pop(d, K, s, mode);
Hardness

%% This section calculates phi2(M, s) for Lasso-OD-Analytical. It requires YALMIP to be set up on MATLAB. It also requires Gurobi solver, which is available for free for academic use.
%  You can comment this section out if you do not want to simulate
%  Lasso-OD-Analytical
    fprintf('Phi2 calculation: ')
    tic
    M = B;
    subsets = nchoosek([1:d], s);
    ops = sdpsettings('solver','gurobi', 'verbose', 0);
    for i = 1:min(200, nchoosek(d, s))
    theta_var = sdpvar(d, 1);
    A = subsets(i, :);
    A_comp = setdiff([1:d], A);
    const = [norm(theta_var(A), 1) == 1, norm(theta_var(A_comp), 1) <= 3];
    objective = s * theta_var' * M * theta_var;
    sol = optimize(const, objective, ops);
    
    phi2S(i) = value(objective);
    end
    
    phi2 = min(phi2S);
    
    b = 4 / phi2;
    toc

    %% This section tunes the Lasso parameters for all T intended.  
    theta_min = 1/sqrt(s);
    tic
    [X_lasso_all, v_lasso_all, lambda_init_all, lambda_thres_all] = Lasso_tune(X, theta_s, T_all, v_star, 0.25);
    toc

    %% PopArt Tune
    tic
    [X_pop_all, eps_cv] = Popart_tune(X, theta_s, T_all, v_pop, Q_pop);
    eps_fit = fitlm(T_all/5,eps_cv);
    eps_cv = eps_fit.Fitted;
    toc
    
    
    %% This section simulates BayesGap for all T intended.
    runtimes = 10;
    fprintf('BayesGap_Adaptive time: ')
    tic
    for run = 1:runtimes
        est_best_bayesadaptive(run, :) = BayesGap_Adaptive_allT(theta_s, X, T_all);
    end
    toc
    prob_error_BayesGapAdaptive = mean(est_best_bayesadaptive ~= best, 1);
    std_prob_error_BayesGapAdaptive = sqrt(prob_error_BayesGapAdaptive .* (1 - prob_error_BayesGapAdaptive) ./ (runtimes ./ 4));

for t_index = 1:length(T_all)
    
    % Take the tune parameters
    T = T_all(t_index)
    lambda_init = lambda_init_all(t_index);
    lambda_thres = lambda_thres_all(t_index);
    X_lasso = X_lasso_all{t_index};
    % X_pop = X_pop_all{t_index};

    c0 = 0.25; % T1/T2 ratio for Lasso-OD-CV
    T2 = floor(T / (1 + c0));
    T1 = T - T2;
    runtimes = 100;
 
    c00 = 2 * (b/s + b*s)^2 * xmax2 / (ceil(log2(s + s^2)) * Hardness * theta_min^2); %T1/T2 ratio for Lasso-OD-Analytical
    if c00 < 0.1 % Avoids very small or very large fractions
        c00 = 0.1;
    elseif c00 > 2
        c00 = 2;
    end
    c1 = b / s; %Lasso-OD-Analytical parameters
    lambda_init00 = theta_min / (c1 + b * s);
    lambda_thres00 = lambda_init00 * c1;

    T2_analytical = floor(T / (1 + c00));
    T1_analytical = T - T2_analytical;

    v_lasso_A = roundT(v_star, T1_analytical);
    X_lasso_A = zeros(T1_analytical, d);
    numTimes_A = round(v_lasso_A * T1_analytical);
    temp_t = 0;
    for i = 1:K 
        if numTimes_A(i) > 0
            X_lasso_A(temp_t + 1:temp_t + numTimes_A(i), :) = repmat(X(:, i)', [numTimes_A(i), 1]);
            temp_t = temp_t + numTimes_A(i);
        end
    end

    % %% PopArt-OD simulation
    % fprintf('PopartOD time: ');
    % tic
    % for run = 1:runtimes/4
    %     [X_pop, Y_pop] = generate_pop_pulls(X, theta_s, T1, v_pop);
    %     % Y_pop = X_pop * theta_s + randn(T1, 1);
    %     [theta_pop, theta_prime] = popart(X_pop, Y_pop, Q_pop, eps_cv(t_index));
    % 
    %     S_pop = find(theta_pop ~= 0);
    % 
    %     if isempty(S_pop)
    %         S_pop = [randi(d)]; % If empty, add a random coordinate
    %     end
    % 
    %     est_best_PopartOD(run) = ODLinBAI(theta_s(S_pop), X(S_pop, :), T2); % Apply OD-LinBAI on S_thres
    %     est_prob_pop(run) = 1 - prod(ismember([1:s], S_pop));
    %     size_pop(run) = length(S_pop);
    % 
    % end
    % toc

    %% Lasso-OD simulation
    fprintf('LassoOD time: ')
    tic
    for run = 1:runtimes

        Y_lasso = X_lasso * theta_s + randn(T1, 1);
       
        [theta_est, ~] = lasso_admm(X_lasso, Y_lasso, lambda_init, 1, 1, 10^3); 
        theta_est_thresholded = theta_est;
        theta_est_thresholded(abs(theta_est_thresholded) < lambda_thres) = 0;

        S_thres = find(theta_est_thresholded ~= 0);  % Estimate the support
        if isempty(S_thres)
            S_thres = [randi(d)]; % If empty, add a random coordinate
        end
        est_best_LassoOD(run) = ODLinBAI(theta_s(S_thres), X(S_thres, :), T2); % Apply OD-LinBAI on S_thres
        est_prob_set(run) = 1 - prod(ismember([1:s], S_thres));
        size_S(run) = length(S_thres);
    end
    toc

    % fprintf('LassoOD-XY time: ')
    % tic
    % for run = 1:runtimes
    % 
    %     Y_lasso = X_lasso * theta_s + randn(T1, 1);
    %     [theta_est, ~] = lasso_admm(X_lasso, Y_lasso, lambda_init, 1, 1, 10^3);
    %     theta_est_thresholded = theta_est;
    %     theta_est_thresholded(abs(theta_est_thresholded) < lambda_thres) = 0;
    % 
    %     S_thres = find(theta_est_thresholded ~= 0); 
    %     if isempty(S_thres)
    %         S_thres = [randi(d)];
    %     end
    %     est_best_LassoOD_XY(run) = ODLinBAI_XY(theta_s(S_thres), X(S_thres, :), T2);
    % end
    % toc
    % 
    % fprintf('Lasso-BayesGap time: ')
    % tic
    % for run = 1:runtimes
    % 
    %     Y_lasso = X_lasso * theta_s + randn(T1, 1);
    %     [theta_est, ~] = lasso_admm(X_lasso, Y_lasso, lambda_init, 1, 1, 10^3);
    %     theta_est_thresholded = theta_est;
    %     theta_est_thresholded(abs(theta_est_thresholded) < lambda_thres) = 0;
    % 
    %     S_thres = find(theta_est_thresholded ~= 0); 
    %     if isempty(S_thres)
    %         S_thres = [randi(d)];
    %     end
    %     est_best_LassoBayes(run) = BayesGap_Adaptive(theta_s(S_thres), X(S_thres, :), T2);
    % end
    % toc
    % 
    % 
    % fprintf('Lasso-Analytical time: ')
    % tic
    % for run = 1:runtimes
    % 
    %     Y_lasso_A = X_lasso_A * theta_s + randn(T1_analytical, 1);
    %     [theta_est_A, ~] = lasso_admm(X_lasso_A, Y_lasso_A, lambda_init00 / 2, 1, 1, 10^3);
    %     theta_est_thresholded_A = theta_est_A;
    %     theta_est_thresholded_A(abs(theta_est_thresholded_A) < lambda_thres00) = 0;
    % 
    %     S_thres_A = find(theta_est_thresholded_A ~= 0); 
    %     if isempty(S_thres_A)
    %         S_thres_A = [randi(d)];
    %     end
    %     est_best_A(run) = ODLinBAI(theta_s(S_thres_A), X(S_thres_A, :), T2_analytical);
    %     est_prob_set_A(run) = 1 - prod(ismember([1:s], S_thres_A));
    %     size_S_A(run) = length(S_thres_A);
    % end
    % toc

    fprintf('ODLinBAI time: ')
    tic
    for run = 1:runtimes 
        est_best(run) = ODYang(theta_s, X, T);
    end
    toc

    fprintf('GSE time: ')
    tic
    for run = 1:runtimes
        est_best_GSE(run) = GSE(theta_s, X, T);
    end
    toc
    
    % This section is computationally too difficult. 
    fprintf('LinearExploration time: ')
    tic
    for run = 1:10
        est_best_linearexp(run) = LinearExploration(theta_s, X, T);
    end
    toc

    fprintf('Peace time: ')
    tic
    for run = 1:10
        run
        est_best_peace(run) = Peace(theta_s, X, T);
    end
    toc
    
    %% Report empirical error probabilities
    prob_error_PopArt_OD(t_index) = mean(est_best_PopartOD ~= best);
    % prob_error_Lasso_A(t_index) = mean(est_best_A ~= best);
    % prob_error_OD(t_index) = mean(est_best ~= best);
    prob_error_Lasso(t_index) = mean(est_best_LassoOD ~= best);
    % prob_error_Lasso_XY(t_index) = mean(est_best_LassoOD_XY ~= best);
    % prob_error_Lasso_Bayes(t_index) = mean(est_best_LassoBayes ~= best);
    % prob_error_GSE(t_index) = mean(est_best_GSE ~= best);
    % %prob_error_linearexp(t_index) = mean(est_best_linearexp ~= best);
    % %prob_error_Peace(t_index) = mean(est_best_peace ~= best);
    
    std_prob_error_PopArt_OD(t_index) = sqrt(prob_error_PopArt_OD(t_index) * (1 - prob_error_PopArt_OD(t_index)) / runtimes/4);
    % std_prob_error_Lasso_A(t_index) = sqrt(prob_error_Lasso_A(t_index) * (1 - prob_error_Lasso_A(t_index)) / runtimes);
    % std_prob_error_OD(t_index) = sqrt(prob_error_OD(t_index) * (1 - prob_error_OD(t_index)) / (runtimes));
    std_prob_error_Lasso(t_index) = sqrt(prob_error_Lasso(t_index) * (1 - prob_error_Lasso(t_index)) / runtimes);
    % std_prob_error_Lasso_XY(t_index) = sqrt(prob_error_Lasso_XY(t_index) * (1 - prob_error_Lasso_XY(t_index)) / (runtimes));
    % std_prob_error_Lasso_Bayes(t_index) = sqrt(prob_error_Lasso_Bayes(t_index) * (1 - prob_error_Lasso_Bayes(t_index)) / (runtimes));


    %std_prob_error_linearexp(t_index) = sqrt(prob_error_linearexp(t_index) * (1 - prob_error_linearexp(t_index)) / 100);
    %std_prob_error_Peace(t_index) = sqrt(prob_error_Peace(t_index) * (1 - prob_error_Peace(t_index)) / 100);
    %std_prob_error_GSE(t_index) = sqrt(prob_error_GSE(t_index) * (1 - prob_error_GSE(t_index)) / (runtimes));

end

%%Plot the results
figure
% semilogx(T_all(1:length(prob_error_OD)), prob_error_OD, '-r', 'Marker', 'square')
hold on
semilogx(T_all(1:length(prob_error_Lasso)), prob_error_Lasso, '--b', 'Marker', 'diamond')
semilogx(T_all(1:length(prob_error_Lasso)), prob_error_PopArt_OD, 'k', 'Marker', 'pentagram')
% semilogx(T_all(1:length(prob_error_OD)), prob_error_Lasso_A, 'k', 'Marker', 'pentagram')
% semilogx(T_all(1:length(prob_error_OD)), prob_error_Lasso_XY, 'm', 'Marker', '*')
% semilogx(T_all(1:length(prob_error_OD)), prob_error_Lasso_Bayes, '-.g', 'Marker', '>')
% semilogx(T_all(1:length(prob_error_OD)), prob_error_BayesGapAdaptive, 'g', 'Marker', 'o')
% semilogx(T_all(1:length(prob_error_OD)), prob_error_GSE, 'c', 'Marker', 'x')
hold on
errorbar(T_all(1:length(prob_error_Lasso)), prob_error_Lasso, std_prob_error_Lasso, 'b', "LineStyle", "none")
errorbar(T_all(1:length(prob_error_Lasso)), prob_error_PopArt_OD, std_prob_error_PopArt_OD, 'k', "LineStyle", "none")

% errorbar(T_all(1:length(prob_error_OD)), prob_error_Lasso_A, std_prob_error_Lasso_A, 'k', "LineStyle", "none")
% errorbar(T_all(1:length(prob_error_OD)), prob_error_Lasso_XY, std_prob_error_Lasso_XY, 'm', "LineStyle", "none")
% errorbar(T_all(1:length(prob_error_OD)), prob_error_Lasso_Bayes, std_prob_error_Lasso_Bayes, 'g', "LineStyle", "none")
% errorbar(T_all(1:length(prob_error_OD)), prob_error_OD, std_prob_error_OD, '-r', "LineStyle", "none")
% errorbar(T_all(1:length(prob_error_OD)), prob_error_BayesGapAdaptive, std_prob_error_BayesGapAdaptive, '-g', "LineStyle", "none")
% errorbar(T_all(1:length(prob_error_OD)), prob_error_GSE, std_prob_error_BayesGapAdaptive, '-c', "LineStyle", "none")


xlabel('$$T$$', 'interpreter', 'latex')
ylabel('Error probability')
legend('Lasso-OD', 'PopArt-OD')
% legend('OD-LinBAI', 'Lasso-OD', 'Lasso-OD-Analytical', 'Lasso-XY-Allocation', 'Lasso-BayesGap', 'BayesGapAdaptive', 'GSE')
axis([T_all(1) T_all(end) 0 Inf])
grid on