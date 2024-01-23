for t_index = 1:length(T_all)
    
    % Take the tune parameters
    T = T_all(t_index)
    X_pp = X_popart{t_index};

    c0 = 0.25; % T1/T2 ratio for Lasso-OD-CV
    T2 = floor(T / (1 + c0));
    T1 = T - T2;

    runtimes = 1000;
    %% PopArt-OD simulation
    fprintf('PopartOD time: ')
    tic
    for run = 1:runtimes
        Y_pop = X_pp * theta_s + randn(T1, 1);
        [theta_pop, theta_prime] = popart(X_pp, Y_pop, Q_pop, eps_cv(t_index));
        S_pop = find(theta_pop ~= 0);
         
        if isempty(S_pop)
            S_pop = [randi(d)]; % If empty, add a random coordinate
        end

        est_best_PopartOD(run) = ODLinBAI(theta_s(S_pop), X(S_pop, :), T2); % Apply OD-LinBAI on S_thres
        est_prob_pop(run) = 1 - prod(ismember([1:s], S_pop));
        size_pop(run) = length(S_pop);

    end
    toc

    error_prop_popart(t_index) = mean(est_prob_pop);
    mean_size(t_index) = mean(size_pop);


end