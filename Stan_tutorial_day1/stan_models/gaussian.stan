data {
    int<lower=0> N;
    int<lower=0> M;
    matrix[N, M] x;
    vector[N] y;
}
transformed data {
    vector[N] cons;
    matrix[N, M+1] X;

    cons = rep_vector(1, N);
    X = append_col(cons, x);
}
parameters {
    vector[K] beta;
    real<lower=0> sigma;
}
model {
    y ~ normal(x * beta, sigma);
}