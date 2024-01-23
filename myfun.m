function obj = myfun(v_var, A)
d = size(A{1}, 1);
K = length(v_var);
B_var = zeros(d, d);
    for i = 1:K
            B_var = B_var + v_var(i) * A{i};
    end

obj = min(eig(B_var)) / sqrt(max(diag(B_var))); 
% obj = min(eig(B_var)); 
end