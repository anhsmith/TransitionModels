functions {
  real plateau_hof_density(real y, real m0, real m1, real lambda0, real lambda1) {
    real C = m1 - m0;
    return (1 / C) * ( (1 / (1 + exp(-lambda0*(y - m0)))) - (1 / (1 + exp(-lambda1*(y - m1)))) );
  }
}

data {
  int<lower=0> N; // Number of data points
  real y[N];      // Observed values
}

parameters {
  real m0;              // Mean for the left distribution
  real m1;              // Mean for the right distribution
  real<lower=0> lambda0; // Standard deviation for left tail
  real<lower=0> lambda1; // Standard deviation for right tail
}

model {
  // Priors
  m0 ~ normal(1300, 100);
  m1 ~ normal(1600, 100);
  lambda0 ~ exponential(0.5);
  lambda1 ~ exponential(0.5);

  // Likelihood
  for (i in 1:N) {
    target += log(plateau_hof_density(y[i], m0, m1, lambda0, lambda1) + 1e-10);
  }
}
