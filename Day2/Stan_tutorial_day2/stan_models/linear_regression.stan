data {
    int<lower=0> N;
    int<lower=0> M;
    matrix[N, M] X;
    vector[N] y;
    
    int<lower=0> N_new;
    int<lower=0> M_new;
    matrix[N_new, M_new] X_new;
}
transformed data {
    vector[N] cons;
    vector[N_new] cons_new;

    matrix[N, M + 1] X_full;
    matrix[N_new, M_new + 1] X_new_full;
    
    cons = rep_vector(1, N);
    cons_new = rep_vector(1, N_new);
    
    X_full = append_col(cons, X);
    X_new_full = append_col(cons_new, X_new);
}
parameters {
    vector[M + 1] beta;
    real<lower=0> sigma;
}
model {
    y ~ normal(X_full * beta, sigma);
    // assuming uniform priors for both beta and sigma
}

generated quantities {
  vector[N_new] y_new;
  for (n in 1:N_new)
    y_new[n] = normal_rng(X_new_full[n] * beta,sigma);
}

