data {
  int<lower=1> N;
  int<lower=1> T;               
  int<lower=1,upper=T> Tsubj[N];                 
  real<lower=0,upper=1> response[N,T];     
  real outcome[N,T];  // no lower and upper bounds   
}

transformed data {
  vector[2] initV;  // initial values for EV
  initV = rep_vector(0.0, 2);
}

parameters {
  // Declare all parameters as vectors for vectorizing
  // Hyper(group)-parameters  
  vector[2] mu_p;  
  vector<lower=0>[2] sigma;
  
  // Subject-level raw parameters (for Matt trick)
  vector[N] A_pr;    // learning rate
  vector[N] tau_pr;  // inverse temperature
  real<lower=0> gamma;
}

transformed parameters {
  // subject-level parameters
  vector<lower=0,upper=1>[N] A;
  vector<lower=0,upper=5>[N] tau;
  
  for (i in 1:N) {
    A[i]   = Phi_approx( mu_p[1]  + sigma[1]  * A_pr[i] );
    tau[i] = Phi_approx( mu_p[2] + sigma[2] * tau_pr[i] ) * 5;
  }
}

model {
  // Hyperparameters
  mu_p  ~ normal(0, 1); 
  sigma ~ cauchy(0, 5);
  gamma ~ cauchy(0, 1);
  
  // individual parameters
  A_pr   ~ normal(0,1);
  tau_pr ~ normal(0,1);
  
  // subject loop and trial loop
  for (i in 1:N) {
    vector[2] ev; // expected value
    real PE;      // prediction error
    
    ev = initV;
    
    for (t in 1:(Tsubj[i])) {        
      // compute action probabilities
     // response[i,t] ~ normal( tau[i] * ev,gamma);
      
      // prediction error 
      PE = outcome[i,t] - ev;
      
      // value updating (learning) 
      ev[response[i,t]] = ev[response[i,t]] + A[i] * PE; 
    }
  }
}


