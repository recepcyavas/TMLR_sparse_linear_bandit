for i = 1:15
    T_all(i)
    X_lasso = X_lasso_all{i};

[lambda_init(i), lambda_thres(i), score(i)] = Lasso_tune_realdataset(X_lasso, theta_s);

end