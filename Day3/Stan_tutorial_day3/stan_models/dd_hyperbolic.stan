data {
  int<lower=1> N;
  int<lower=1> T;
  int<lower=1, upper=T> Tsubj[N];
  real<lower=0> delay_later[N,T];
  real<lower=0> amount_later[N,T];
  real<lower=0> delay_sooner[N,T];
  real<lower=0> amount_sooner[N,T];
  int<lower=0,upper=1> choice[N, T]; // 0 for instant reward, 1 for delayed reward
}
transformed data {
}
parameters {
// Declare all parameters as vectors for vectorizing
  // Hyper(group)-parameters  
  vector[2] mu_p;  
  vector<lower=0>[2] sigma;
    
  // Subject-level raw parameters (for Matt trick)
  vector[N] k_pr;
  vector[N] beta_pr;    
}
transformed parameters {
  // Transform subject-level raw parameters 
  vector<lower=0,upper=1>[N] k;
  vector<lower=0,upper=5>[N] beta;
        
  for (i in 1:N) {
    k[i]    = Phi_approx( mu_p[1] + sigma[1] * k_pr[i] );
    beta[i] = Phi_approx( mu_p[2] + sigma[2] * beta_pr[i] ) * 5;
  }
}
model {
// Hyperbolic function
  // Hyperparameters
  mu_p  ~ normal(0, 1); 
  sigma ~ cauchy(0, 5);
    
  // individual parameters
  k_pr    ~ normal(0, 1);
  beta_pr ~ normal(0, 1);
    
  for (i in 1:N) {
    // Define values
    real ev_later;
    real ev_sooner;
      
    for (t in 1:(Tsubj[i])) {
      ev_later   = amount_later[i,t]  / ( 1 + k[i] * delay_later[i,t] );
      ev_sooner  = amount_sooner[i,t] / ( 1 + k[i] * delay_sooner[i,t] );
      choice[i,t] ~ bernoulli_logit( beta[i] * (ev_later - ev_sooner) );
    }
  }
}

