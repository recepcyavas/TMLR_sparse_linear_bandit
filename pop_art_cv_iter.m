function [eps_cv] = pop_art_cv_iter(arms, theta_s, T, v_pop, Q_pop, num_rep, s)

log_eps_set = linspace(-2, 0, 40);
set_eps = 10.^log_eps_set * T/10;
log_size = 2;

for r = 1:num_rep
[eps_cv] = pop_art_cv_rand(arms, theta_s, T, v_pop, Q_pop, set_eps, num_rep, s);
init_idx = find(set_eps == eps_cv);
log_size = log_size / 2;
log_eps_set = linspace(log_eps_set(init_idx) - log_size / 2, log_eps_set(init_idx) + log_size / 2, 10);
set_eps =  T/10 .* 10.^(log_eps_set);

end