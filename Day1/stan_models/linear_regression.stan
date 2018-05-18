data {
    int<lower=0> N;
    int<lower=0> M;
    matrix[N, M] X;
    vector[N] y;
}
transformed data {
    vector[N] cons;
    matrix[N, M + 1] X_full;

    cons = rep_vector(1, N);
    X_full = append_col(cons, X);
}
parameters {
    vector[M + 1] beta;
    real<lower=0> sigma;
}
model {
    y ~ normal(X_full * beta, sigma);
    // assuming uniform priors for both beta and sigma
}

