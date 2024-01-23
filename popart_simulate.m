clearvars
load d10K50mode3.mat

[v_pop, Q_pop] = vpopcalc(X);

    %% PopArt Tune
    tic
    [X_pop_all, eps_cv] = Popart_tune(X, theta_s, T_all, v_pop, Q_pop);
    eps_fit = fitlm(T_all/5,eps_cv);
    eps_cv = eps_fit.Fitted;
    toc

    for t_index = 1:length(T_all)
    
    % Take the tune parameters
    T = T_all(t_index);
    X_pop = X_pop_all{t_index};

    c0 = 0.25; % T1/T2 ratio for Lasso-OD-CV
    T2 = floor(T / (1 + c0));
    T1 = T - T2;
    runtimes = 4000;

    %% PopArt-OD simulation
    fprintf('PopartOD time: ');
    tic
    for run = 1:runtimes/4
        [X_pop, Y_pop] = generate_pop_pulls(X, theta_s, T1, v_pop);
        % Y_pop = X_pop * theta_s + randn(T1, 1);
        [theta_pop, theta_prime] = popart(X_pop, Y_pop, Q_pop, eps_cv(t_index));
     
        S_pop = find(theta_pop ~= 0);
         
        if isempty(S_pop)
            S_pop = [randi(d)]; % If empty, add a random coordinate
        end

        est_best_PopartOD(run) = ODLinBAI(theta_s(S_pop), X(S_pop, :), T2); % Apply OD-LinBAI on S_thres
        est_prob_pop(run) = 1 - prod(ismember([1:s], S_pop));
        size_pop(run) = length(S_pop);

    end
    toc

    prob_error_PopArt_OD(t_index) = mean(est_best_PopartOD ~= best);
    std_prob_error_PopArt_OD(t_index) = sqrt(prob_error_PopArt_OD(t_index) * (1 - prob_error_PopArt_OD(t_index)) / runtimes/4);

    end

    save('d10K50mode3.mat')
    clear 


load d10K100mode3.mat

[v_pop, Q_pop] = vpopcalc(X);

    %% PopArt Tune
    tic
    [X_pop_all, eps_cv] = Popart_tune(X, theta_s, T_all, v_pop, Q_pop);
    eps_fit = fitlm(T_all/5,eps_cv);
    eps_cv = eps_fit.Fitted;
    toc

    for t_index = 1:length(T_all)
    
    % Take the tune parameters
    T = T_all(t_index);
    X_pop = X_pop_all{t_index};

    c0 = 0.25; % T1/T2 ratio for Lasso-OD-CV
    T2 = floor(T / (1 + c0));
    T1 = T - T2;
    runtimes = 4000;

    %% PopArt-OD simulation
    fprintf('PopartOD time: ');
    tic
    for run = 1:runtimes/4
        [X_pop, Y_pop] = generate_pop_pulls(X, theta_s, T1, v_pop);
        % Y_pop = X_pop * theta_s + randn(T1, 1);
        [theta_pop, theta_prime] = popart(X_pop, Y_pop, Q_pop, eps_cv(t_index));
     
        S_pop = find(theta_pop ~= 0);
         
        if isempty(S_pop)
            S_pop = [randi(d)]; % If empty, add a random coordinate
        end

        est_best_PopartOD(run) = ODLinBAI(theta_s(S_pop), X(S_pop, :), T2); % Apply OD-LinBAI on S_thres
        est_prob_pop(run) = 1 - prod(ismember([1:s], S_pop));
        size_pop(run) = length(S_pop);

    end
    toc

    prob_error_PopArt_OD(t_index) = mean(est_best_PopartOD ~= best);
    std_prob_error_PopArt_OD(t_index) = sqrt(prob_error_PopArt_OD(t_index) * (1 - prob_error_PopArt_OD(t_index)) / runtimes/4);

    end

    save('d10K100mode3.mat')
    clear 
    load d20K50mode3.mat

    [v_pop, Q_pop] = vpopcalc(X);

    %% PopArt Tune
    tic
    [X_pop_all, eps_cv] = Popart_tune(X, theta_s, T_all, v_pop, Q_pop);
    eps_fit = fitlm(T_all/5,eps_cv);
    eps_cv = eps_fit.Fitted;
    toc

    for t_index = 1:length(T_all)
    
    % Take the tune parameters
    T = T_all(t_index);
    X_pop = X_pop_all{t_index};

    c0 = 0.25; % T1/T2 ratio for Lasso-OD-CV
    T2 = floor(T / (1 + c0));
    T1 = T - T2;
    runtimes = 4000;

    %% PopArt-OD simulation
    fprintf('PopartOD time: ');
    tic
    for run = 1:runtimes/4
        [X_pop, Y_pop] = generate_pop_pulls(X, theta_s, T1, v_pop);
        % Y_pop = X_pop * theta_s + randn(T1, 1);
        [theta_pop, theta_prime] = popart(X_pop, Y_pop, Q_pop, eps_cv(t_index));
     
        S_pop = find(theta_pop ~= 0);
         
        if isempty(S_pop)
            S_pop = [randi(d)]; % If empty, add a random coordinate
        end

        est_best_PopartOD(run) = ODLinBAI(theta_s(S_pop), X(S_pop, :), T2); % Apply OD-LinBAI on S_thres
        est_prob_pop(run) = 1 - prod(ismember([1:s], S_pop));
        size_pop(run) = length(S_pop);

    end
    toc

    prob_error_PopArt_OD(t_index) = mean(est_best_PopartOD ~= best);
    std_prob_error_PopArt_OD(t_index) = sqrt(prob_error_PopArt_OD(t_index) * (1 - prob_error_PopArt_OD(t_index)) / runtimes/4);

    end
        
    save('d20K50mode3.mat')
    clear 
    load d20K100mode3.mat

    [v_pop, Q_pop] = vpopcalc(X);

    %% PopArt Tune
    tic
    [X_pop_all, eps_cv] = Popart_tune(X, theta_s, T_all, v_pop, Q_pop);
    eps_fit = fitlm(T_all/5,eps_cv);
    eps_cv = eps_fit.Fitted;
    toc

    for t_index = 1:length(T_all)
    
    % Take the tune parameters
    T = T_all(t_index);
    X_pop = X_pop_all{t_index};

    c0 = 0.25; % T1/T2 ratio for Lasso-OD-CV
    T2 = floor(T / (1 + c0));
    T1 = T - T2;
    runtimes = 4000;

    %% PopArt-OD simulation
    fprintf('PopartOD time: ');
    tic
    for run = 1:runtimes/4
        [X_pop, Y_pop] = generate_pop_pulls(X, theta_s, T1, v_pop);
        % Y_pop = X_pop * theta_s + randn(T1, 1);
        [theta_pop, theta_prime] = popart(X_pop, Y_pop, Q_pop, eps_cv(t_index));
     
        S_pop = find(theta_pop ~= 0);
         
        if isempty(S_pop)
            S_pop = [randi(d)]; % If empty, add a random coordinate
        end

        est_best_PopartOD(run) = ODLinBAI(theta_s(S_pop), X(S_pop, :), T2); % Apply OD-LinBAI on S_thres
        est_prob_pop(run) = 1 - prod(ismember([1:s], S_pop));
        size_pop(run) = length(S_pop);

    end
    toc

    prob_error_PopArt_OD(t_index) = mean(est_best_PopartOD ~= best);
    std_prob_error_PopArt_OD(t_index) = sqrt(prob_error_PopArt_OD(t_index) * (1 - prob_error_PopArt_OD(t_index)) / runtimes/4);

    end
    
    save('d20K100mode3.mat')    
   

