function [X_pop_all, eps_cv] = Popart_tune(X, theta_s, T_all, v_pop, Q_pop)
%This function tunes the Lasso parameters
d = size(X, 1);
K = size(X, 2);
s = sum(theta_s ~= 0);
c0 = 0.25;



for t_index = 1:length(T_all)
    T = T_all(t_index);
    T2 = floor(T / (1 + c0));
    T1 = T - T2;

    v_pop = roundT(v_pop, T1);
    X_pop = zeros(T1, d);
    numTimes = round(v_pop * T1);
    temp_t = 0;
    for i = 1:K 
        if numTimes(i) > 0
            X_pop(temp_t + 1:temp_t + numTimes(i), :) = repmat(X(:, i)', [numTimes(i), 1]);
            temp_t = temp_t + numTimes(i);
        end
    end
    y_pop = X_pop * theta_s + randn(T1, 1);
    
    fprintf('Popart parameter tuning: ')
    tic
    set_eps = linspace(0.1, 3, 50);
    [~, eps_cv(t_index)] = pop_art_cv(X_pop, y_pop, Q_pop, set_eps, 3, s);
    toc

    X_pop_all{t_index} = X_pop;

end

mdl = fitlm(T_all, eps_cv);
eps_cv = mdl.Fitted;

end