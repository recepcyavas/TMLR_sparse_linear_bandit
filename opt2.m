load Amat.mat
K = 50;
d = 10;

prob = optimproblem('ObjectiveSense','maximize');
% options = optimoptions('fmincon', "Algorithm","interior-point",...
   % "EnableFeasibilityMode",true,...
   % "SubproblemAlgorithm","cg");
options = optimoptions('fmincon', 'Display', 'iter', "Algorithm", "sqp", 'FunctionTolerance', 10^-16, 'MaxFunctionEvaluations', 10^7, 'ConstraintTolerance', 10^-6,'StepTolerance', 10^-16, 'MaxIterations', 10^3);
v = optimvar('v', K, 1,'LowerBound', zeros(K, 1),'UpperBound', ones(K, 1));
funobj = fcn2optimexpr(@(x) myfun(x, A), v);
prob.Objective = funobj;
prob.Constraints = sum(v) == 1;
ms = GlobalSearch(FunctionTolerance=1e-20);
% x0.v = ones(K, 1) * 1/K;
x0.v = v0;
x0.v = sol.v;
[sol,~,exitflag,output] = solve(prob, x0, 'Options',options);
% [sol,~,exitflag,output] = solve(prob, x0, Solver = "ga");
% [sol,~,exitflag,output] = solve(prob, x0, ms);


% options = optimoptions('fmincon', 'Display','iter', 'MaxFunctionEvaluations', 100000, 'MaxIterations', 100000, 'StepTolerance', 10^-20, 'ConstraintTolerance', 10^-20);
% vv(vv == 0) = 10^-6;
% vv = vv / sum(vv);
% f = fmincon(@(x) myfun(x, A), vv, [], [], ones(1, K), 1, zeros(1, K), ones(1, K), [], options);
