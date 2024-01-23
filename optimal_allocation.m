function [design, max_rho] = optimal_allocation(X, active)
    X = X';
    [K, d] = size(X);
    design = initwt(X'); %Kumar-Yildirim initialization
    if nargin < 2
        active = [1:1:K];
    end
    L = length(active);
    Yhat = zeros(nchoosek(L, 2), d);
    S = nchoosek(active, 2);
    for i = 1:nchoosek(L, 2)
        Yhat(i, :) = X(S(i, 1), :) - X(S(i, 2), :);
    end
    
    max_iter = 5000;
    
    for count = 1:max_iter
        A_inv = pinv(X' * diag(design) * X);
        [U, D, ~] = svd(A_inv);
        Ainvhalf = U * sqrt(D) * U';
        
        newY = (Yhat * Ainvhalf).^2;
        rho = sum(newY, 2);
        
        [~, idx] = max(rho);
        y = Yhat(idx, :)';
        g = (X * A_inv * y) .* (X * A_inv * y);
        [~, g_idx] = max(g);
        
        gamma = 2 / (count + 2);
        design_update = -gamma * design;
        design_update(g_idx) = design_update(g_idx) + gamma;
        
        relative = norm(design_update) / norm(design);
        design = design + design_update;
        
        if relative < 0.01
            break
        end
    end
    
    max_rho = max(rho);
    design(design < 10^(-5)) = 0;
    design = design / sum(design);
    
end

