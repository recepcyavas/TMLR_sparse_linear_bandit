function [v_pop, Q_pop] = vpopcalc(X)
    [d, K] = size(X);
    for k = 1:K
        A{k} = X(:, k) * X(:, k)';
    end

    Imat = eye(d);

    cvx_begin quiet
        variable vpop(K)
        expression Bpop(d, d)
        expression Qdiag(d)
        Bpop = zeros(d, d);
        for i = 1:K
            Bpop = Bpop + vpop(i) * A{i};
        end 
        for i = 1:d
            Qdiag(i) = matrix_frac(Imat(:, i), Bpop);
        end
        minimize (max(Qdiag)) 
            subject to
            sum(vpop) == 1
            vpop >= 0
    cvx_end

    v_pop = vpop .* (vpop > 0);
    Q_pop = pinv(Bpop);
end