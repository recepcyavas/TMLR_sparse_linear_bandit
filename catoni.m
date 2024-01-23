function [y_catoni] = catoni(z, alpha)

y0 = mean(z);
n = length(z);
y = y0;
score = @(y) sum(alpha * phi_func(z - y));
options = optimset('Display','off');
y_catoni = fsolve(score, y0, options);

end

function [phi] = phi_func(x)
phi = sign(x) .* log(1 + abs(x) + x.^2/2);
end
