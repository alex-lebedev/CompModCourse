data {
  int<lower=1> N;      // Number of subjects
  int<lower=0> Nu_max; // Max (across subjects) number of upper boundary responses
  int<lower=0> Nl_max; // Max (across subjects) number of lower boundary responses
  int<lower=0> Nu[N];  // Number of upper boundary responses for each subj
  int<lower=0> Nl[N];  // Number of lower boundary responses for each subj
  real RTu[N,Nu_max];  // upper boundary response times
  real RTl[N,Nl_max];  // lower boundary response times
  real minRT[N];       // minimum RT for each subject of the observed data 
  real RTbound;        // lower bound or RT across all subjects (e.g., 0.1 second)
}
parameters {

  // Hyper(group)-parameters  
  vector[4] mu_p;  
  vector<lower=0>[4] sigma;
    
  // Subject-level raw parameters (for Matt trick)
  vector[N] alpha_pr;
  vector[N] beta_pr;
  vector[N] delta_pr;
  vector[N] tau_pr;
}
transformed parameters {
  // Transform subject-level raw parameters
  vector<lower=0>[N]         alpha; // boundary separation
  vector<lower=0,upper=1>[N] beta;  // initial bias
  vector<lower=0>[N]         delta; // drift rate
  vector<lower=RTbound,upper=max(minRT)>[N] tau; // nondecision time

  for (i in 1:N) {
    beta[i] = Phi_approx( mu_p[2] + sigma[2] * beta_pr[i] );
    tau[i]  = Phi_approx( mu_p[4] + sigma[4] * tau_pr[i] ) * (minRT[N]-RTbound) + RTbound;
  }
  alpha = exp( mu_p[1] + sigma[1] * alpha_pr );
  delta = exp( mu_p[3] + sigma[3] * delta_pr );
}
model {
  // Hyperparameters
  mu_p  ~ normal(0,1);        
  sigma ~ cauchy(0,5);
  
  // Individual parameters for non-centered parameterization
  alpha_pr ~ normal(0, 1);
  beta_pr  ~ normal(0, 1);
  delta_pr ~ normal(0, 1);
  tau_pr   ~ normal(0, 1);

  // Begin subject loop
  for (i in 1:N) {
    // Response time distributed along wiener first passage time distribution
    RTu[i,:Nu[i]] ~ wiener(alpha[i], tau[i], beta[i], delta[i]);
    RTl[i,:Nl[i]] ~ wiener(alpha[i], tau[i], 1-beta[i], -delta[i]);
    
  } // end of subject loop
}
