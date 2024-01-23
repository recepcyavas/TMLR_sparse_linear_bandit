function [nu_out] = roundT(nu, T)
    zero_idx = nu < 10^(-5);
    nu_nz = nu(~zero_idx);
    
    d = length(nu_nz);
    T_vec = ceil((T - d/2) * nu_nz);
    while sum(T_vec) ~= T
        if sum(T_vec) < T
            j = find((T_vec)./ nu_nz == min((T_vec) ./ nu_nz), 1);
            T_vec(j) = T_vec(j) + 1;
        else
            j =  find((T_vec - 1) ./ nu_nz == max((T_vec - 1) ./ nu_nz), 1);
            T_vec(j) = T_vec(j) - 1;
        end
    end
    
    nu_out = zeros(length(nu), 1);
    nu_out(~zero_idx) = T_vec ./ T;
    nu_out(zero_idx) = 0;        
            
end

