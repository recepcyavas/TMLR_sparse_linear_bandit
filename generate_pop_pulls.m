function [X, Y_pull] = generate_pop_pulls(arms, theta_s, T, v_pop)

c = cumsum([0,v_pop(:).']);
c = c/c(end); 
[~,index] = histc(rand(1,T),c);
for i = 1:T
    X_pull(:, i) = arms(:, index(i));
end

Y_pull = X_pull' * theta_s + randn(T, 1);

X = X_pull';

end